"""
Flask backend for the Crop-from-NDVI website.

Endpoints:
  POST /predict   { lat, lon, date }  -> crop, confidence, NDVI curve, lifecycle window
  GET  /health     simple check that the server + model are ready
"""

import os
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

import ee_extraction
import lifecycle
import model as model_module

app = Flask(__name__)
CORS(app)  # allow the static HTML/JS frontend (served separately) to call this API

# Optional: set your GEE cloud project id here if your account requires one
# (needed for most accounts created after the GEE non-commercial signup changes).
GEE_PROJECT = os.environ.get("GEE_PROJECT")  # e.g. "my-earth-engine-project"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = model_module.load_model()
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": get_model() is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        lat = float(data["lat"])
        lon = float(data["lon"])
        date = data["date"]  # 'YYYY-MM-DD'
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Please provide numeric 'lat', 'lon' and a 'date' (YYYY-MM-DD)."}), 400

    model = get_model()
    if model is None:
        return jsonify({
            "error": "Model not trained yet. Run train_model.py with your FARM_NDVI.xlsx first."
        }), 503

    try:
        # 1. Real-time NDVI extraction from Sentinel-2 via Earth Engine
        ndvi_df = ee_extraction.extract_ndvi_timeseries(lat, lon, date)

        if ndvi_df.empty or len(ndvi_df) < 5:
            return jsonify({
                "error": "Not enough cloud-free Sentinel-2 imagery was found for this location/date. "
                         "Try a different point or date."
            }), 422

        # 2. Detect NDVI lifecycles (peaks/valleys -> candidate crop cycles)
        farm_smoothed, cycles = lifecycle.detect_lifecycles(ndvi_df)

        if not cycles:
            return jsonify({
                "error": "No clear crop growth cycle was detected in the NDVI signal for this period."
            }), 422

        # 3. Pick the cycle that matches the user's query date
        chosen_cycle = lifecycle.pick_query_cycle(farm_smoothed, cycles, date)

        # 4. Feature engineering + classification
        features = lifecycle.extract_features_from_cycle(farm_smoothed, chosen_cycle)
        predicted_crop, confidence, all_probs = model_module.predict_crop(model, features)

        # 5. Package the NDVI curve + lifecycle markers for charting on the frontend
        curve = [
            {"date": d.strftime("%Y-%m-%d"), "ndvi": round(float(v), 4)}
            for d, v in zip(farm_smoothed["Date"], farm_smoothed["NDVI_smooth"])
        ]

        sv, pk, ev = chosen_cycle["start_valley"], chosen_cycle["peak"], chosen_cycle["end_valley"]

        return jsonify({
            "crop": predicted_crop,
            "confidence": round(confidence, 4),
            "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()},
            "ndvi_curve": curve,
            "lifecycle": {
                "start_date": chosen_cycle["start_date"].strftime("%Y-%m-%d"),
                "peak_date": chosen_cycle["peak_date"].strftime("%Y-%m-%d"),
                "end_date": chosen_cycle["end_date"].strftime("%Y-%m-%d"),
                "duration_days": chosen_cycle["duration_days"],
                "start_index": int(sv),
                "peak_index": int(pk),
                "end_index": int(ev),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Something went wrong while processing this request: {str(e)}"}), 500


if __name__ == "__main__":
    print("Initializing Earth Engine...")
    ee_extraction.initialize_earth_engine(project=GEE_PROJECT)
    print("Earth Engine ready.")

    if get_model() is None:
        print("WARNING: No trained model found at crop_model.pkl.")
        print("Run: python train_model.py /path/to/FARM_NDVI.xlsx")
    else:
        print("Model loaded.")

    app.run(debug=True, port=5000)
