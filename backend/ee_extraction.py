import os
import ee
import pandas as pd

BUFFER_SIZE = 50       # metres, used only to filter which images intersect the point
CLOUD_LIMIT = 20       # max allowed CLOUDY_PIXEL_PERCENTAGE
PIXEL_SIZE = 10        # Sentinel-2 native resolution (m)

# How far back/forward of the observation date to pull imagery.
DEFAULT_MONTHS_WINDOW = 12


def initialize_earth_engine(project=None):
    """
    Authenticate + initialize Earth Engine.

    If no project is passed, automatically read it from the
    GEE_PROJECT environment variable.
    """
    if project is None:
        project = os.environ.get("GEE_PROJECT")

    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def _add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)


def extract_ndvi_timeseries(lat, lon, observation_date, months_window=DEFAULT_MONTHS_WINDOW):
    """
    Pull a real-time Sentinel-2 NDVI time series for a single point.

    Args:
        lat, lon: floats, WGS84 coordinates
        observation_date: 'YYYY-MM-DD' string or pandas Timestamp
        months_window: how many months before/after observation_date to scan

    Returns:
        pandas.DataFrame with columns:
        [Date, NDVI, cloud_percentage, image_id]
    """
    obs_date = pd.to_datetime(observation_date)

    point = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(BUFFER_SIZE)

    start_date = ee.Date(obs_date.strftime('%Y-%m-%d')).advance(-months_window, 'month')
    end_date = ee.Date(obs_date.strftime('%Y-%m-%d')).advance(months_window, 'month')

    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(buffer)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_LIMIT))
        .sort('system:time_start')
        .map(_add_ndvi)
    )

    def _reduce(image):
        ndvi_value = image.select('NDVI').reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=PIXEL_SIZE,
            maxPixels=1e9
        )

        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'NDVI': ndvi_value.get('NDVI'),
            'cloud_percentage': image.get('CLOUDY_PIXEL_PERCENTAGE'),
            'image_id': image.get('system:index'),
        })

    features = collection.map(_reduce)
    result = features.getInfo()

    rows = []

    for f in result['features']:
        props = f['properties']

        if props.get('NDVI') is None:
            continue

        rows.append({
            'Date': props['date'],
            'NDVI': props['NDVI'],
            'cloud_percentage': props.get('cloud_percentage'),
            'image_id': props.get('image_id'),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df['Date'] = pd.to_datetime(df['Date'])
    df = (
        df.sort_values('Date')
          .drop_duplicates(subset='Date')
          .reset_index(drop=True)
    )

    return df