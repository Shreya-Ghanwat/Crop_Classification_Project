"""
NDVI lifecycle (crop-cycle) detection and feature engineering.

This is a direct port of the exploratory notebook logic into reusable
functions: smoothing -> peak/valley detection -> cycle extraction ->
filtering -> feature engineering for the classifier.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid

SEASONS = {
    "Kharif": [5, 6, 7, 8],
    "Rabi": [10, 11, 12],
    "Zaid": [2, 3, 4],
}

CROP_SEASONS = {
    "Sugarcane": ["All"],
    "Onion": ["Rabi", "Zaid"],
    "Cotton": ["Kharif"],
    "Paddy": ["Kharif"],
}

FEATURE_COLUMNS = [
    "duration", "growth_duration", "decline_duration",
    "peak_ndvi", "mean_ndvi", "std_ndvi", "auc",
    "growth_rate", "decline_rate", "skewness", "kurtosis",
    "sos_doy", "pos_doy", "eos_doy",
]


def smooth_ndvi(farm, window=3):
    farm = farm.copy()
    farm["NDVI_smooth"] = farm["NDVI"].rolling(window=window, center=True, min_periods=1).mean()
    return farm


def compute_iqr(y):
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    return q3 - q1


def dynamic_prominence(iqr):
    return max(0.05, 0.5 * iqr)


def detect_peaks_valleys(farm):
    y = farm["NDVI_smooth"].values
    iqr = compute_iqr(y)
    prominence_threshold = dynamic_prominence(iqr)

    peaks, _ = find_peaks(y, prominence=prominence_threshold)
    valleys, _ = find_peaks(-y, prominence=prominence_threshold)

    window = min(10, len(y))
    left_valley = int(np.argmin(y[:window]))
    right_valley = len(y) - window + int(np.argmin(y[-window:]))

    all_valleys = list(valleys) + [left_valley, right_valley]
    valleys = np.array(sorted(set(all_valleys)))

    return peaks, valleys, iqr, prominence_threshold


def extract_cycles(farm, peaks, valleys):
    cycles = []
    for peak in peaks:
        left_valleys = valleys[valleys < peak]
        right_valleys = valleys[valleys > peak]
        if len(left_valleys) == 0 or len(right_valleys) == 0:
            continue
        cycles.append({
            "start_valley": left_valleys[-1],
            "peak": peak,
            "end_valley": right_valleys[0],
        })
    return cycles


def calculate_cycle_metrics(farm, cycles):
    metrics = []
    for cycle in cycles:
        sv, pk, ev = cycle["start_valley"], cycle["peak"], cycle["end_valley"]

        start_date = farm.iloc[sv]["Date"]
        peak_date = farm.iloc[pk]["Date"]
        end_date = farm.iloc[ev]["Date"]

        start_ndvi = farm.iloc[sv]["NDVI_smooth"]
        peak_ndvi = farm.iloc[pk]["NDVI_smooth"]
        end_ndvi = farm.iloc[ev]["NDVI_smooth"]

        metrics.append({
            "start_valley": sv, "peak": pk, "end_valley": ev,
            "start_date": start_date, "peak_date": peak_date, "end_date": end_date,
            "duration_days": (end_date - start_date).days,
            "growth_days": (peak_date - start_date).days,
            "start_ndvi": start_ndvi, "peak_ndvi": peak_ndvi, "end_ndvi": end_ndvi,
            "absolute_growth": peak_ndvi - start_ndvi,
        })
    return metrics


def filter_cycles(cycle_metrics, min_duration=60, min_growth=0.15):
    return [
        c for c in cycle_metrics
        if c["duration_days"] >= min_duration and c["absolute_growth"] >= min_growth
    ]


def season_filter(cycles, crop_name):
    allowed = CROP_SEASONS.get(crop_name, ["All"])
    if "All" in allowed:
        return cycles
    allowed_months = []
    for season in allowed:
        allowed_months.extend(SEASONS[season])
    return [c for c in cycles if c["start_date"].month in allowed_months]


def extract_features_from_cycle(farm, cycle):
    sv, pk, ev = cycle["start_valley"], cycle["peak"], cycle["end_valley"]

    lifecycle = farm.iloc[sv:ev + 1]
    ndvi = lifecycle["NDVI_smooth"].values

    duration = cycle["duration_days"]
    growth_duration = cycle["growth_days"]
    decline_duration = duration - growth_duration

    peak_ndvi = cycle["peak_ndvi"]
    mean_ndvi = float(np.mean(ndvi))
    std_ndvi = float(np.std(ndvi))
    auc = float(trapezoid(ndvi))

    growth_rate = (cycle["peak_ndvi"] - cycle["start_ndvi"]) / max(growth_duration, 1)
    decline_rate = (cycle["peak_ndvi"] - cycle["end_ndvi"]) / max(decline_duration, 1)

    ndvi_skewness = float(skew(ndvi)) if len(ndvi) > 2 else 0.0
    ndvi_kurtosis = float(kurtosis(ndvi)) if len(ndvi) > 2 else 0.0

    return {
        "duration": duration,
        "growth_duration": growth_duration,
        "decline_duration": decline_duration,
        "peak_ndvi": peak_ndvi,
        "mean_ndvi": mean_ndvi,
        "std_ndvi": std_ndvi,
        "auc": auc,
        "growth_rate": growth_rate,
        "decline_rate": decline_rate,
        "skewness": ndvi_skewness,
        "kurtosis": ndvi_kurtosis,
        "sos_doy": cycle["start_date"].dayofyear,
        "pos_doy": cycle["peak_date"].dayofyear,
        "eos_doy": cycle["end_date"].dayofyear,
    }


def detect_lifecycles(farm_df, min_duration=60, min_growth=0.15, crop_name=None):
    """
    Full pipeline for one farm's NDVI series: smooth -> detect peaks/valleys
    -> extract cycles -> compute metrics -> filter valid ones.

    farm_df must have columns ['Date', 'NDVI'], Date as datetime, sorted ascending.
    crop_name: if provided, also applies the season filter for that crop
               (used during training). Leave None at inference time, since
               the crop is unknown.

    Returns: (farm_smoothed_df, list_of_cycle_metrics_dicts)
    """
    farm = smooth_ndvi(farm_df)

    if len(farm) < 5:
        return farm, []

    peaks, valleys, _, _ = detect_peaks_valleys(farm)
    cycles = extract_cycles(farm, peaks, valleys)
    cycle_metrics = calculate_cycle_metrics(farm, cycles)
    filtered = filter_cycles(cycle_metrics, min_duration=min_duration, min_growth=min_growth)

    if crop_name is not None:
        filtered = season_filter(filtered, crop_name)

    return farm, filtered


def pick_query_cycle(farm, cycles, query_date):
    """
    At inference time we may detect several candidate lifecycles in the
    extraction window. Pick the one whose [start_date, end_date] span
    contains the user's query_date, or -- failing that -- the closest one
    to the query_date.
    """
    if not cycles:
        return None

    query_date = pd.to_datetime(query_date)

    containing = [c for c in cycles if c["start_date"] <= query_date <= c["end_date"]]
    if containing:
        return min(containing, key=lambda c: abs((c["peak_date"] - query_date).days))

    return min(cycles, key=lambda c: abs((c["peak_date"] - query_date).days))
