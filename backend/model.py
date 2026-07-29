"""
Crop classifier: trains a RandomForest on lifecycle features extracted from
historical farm NDVI data, and predicts crop + confidence for new lifecycles.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from lifecycle import detect_lifecycles, extract_features_from_cycle, FEATURE_COLUMNS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_model.pkl")

CROP_SHEETS = ["Sugarcane", "Onion", "Cotton", "Paddy"]


def _load_crop_sheet(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")
    df = df.dropna(subset=["Farm_ID", "Date", "NDVI"])

    farms = {}
    for farm_name, farm in df.groupby("Farm_ID"):
        farms[farm_name] = farm.sort_values("Date").reset_index(drop=True)[["Date", "NDVI"]]
    return farms


def build_training_dataset(excel_path):
    """
    Reproduces the notebook's feature_df: one row per valid lifecycle,
    across all crops/farms in the training workbook.
    """
    rows = []
    for crop_name in CROP_SHEETS:
        farms = _load_crop_sheet(excel_path, crop_name)
        for farm_name, farm_df in farms.items():
            farm_smoothed, cycles = detect_lifecycles(farm_df, crop_name=crop_name)
            for cycle in cycles:
                features = extract_features_from_cycle(farm_smoothed, cycle)
                features["crop"] = crop_name
                rows.append(features)

    return pd.DataFrame(rows)


def train_model(excel_path, n_estimators=50, random_state=42):
    feature_df = build_training_dataset(excel_path)

    if feature_df.empty:
        raise ValueError(
            "No valid lifecycles were extracted from the training data. "
            "Check the Excel file's sheet names/columns match the expected format."
        )

    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["crop"]

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)

    joblib.dump(rf, MODEL_PATH)
    return rf, feature_df


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_crop(model, features_dict):
    """
    features_dict: output of lifecycle.extract_features_from_cycle
    Returns: (predicted_crop, confidence, all_class_probabilities)
    """
    X = pd.DataFrame([features_dict])[FEATURE_COLUMNS]
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    best_idx = probs.argmax()
    predicted_crop = classes[best_idx]
    confidence = float(probs[best_idx])

    all_probs = {cls: float(p) for cls, p in zip(classes, probs)}
    return predicted_crop, confidence, all_probs
