# Field NDVI Console — Crop Identification from Satellite Data

Give it a latitude, longitude, and date. It pulls real-time Sentinel-2 imagery
via Google Earth Engine, traces the NDVI growth cycle at that point, and
identifies the crop using a RandomForest classifier trained on historical
lifecycle shapes — with a confidence score.

```
Frontend (HTML/CSS/JS)  --fetch-->  Flask backend  --Earth Engine API-->  Sentinel-2
                                          |
                                          v
                                  trained RandomForest
                                  (crop_model.pkl)
```

## Project structure

```
crop-ndvi-app/
  backend/
    app.py              Flask server, /predict endpoint
    ee_extraction.py     real-time NDVI extraction from Earth Engine
    lifecycle.py          peak/valley detection + feature engineering
    model.py               train/load RandomForest, predict with confidence
    train_model.py       one-time script: trains model from your Excel dataset
    requirements.txt
  frontend/
    index.html
    style.css
    script.js
```

## 1. Set up the backend

```bash
cd crop-ndvi-app/backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Authenticate with Earth Engine (one-time, personal account)

```bash
earthengine authenticate
```

This opens a browser window — log in with the Google account that has Earth
Engine access, approve access, and the credential is cached on your machine.
You won't need to repeat this unless the token expires.

> If your account requires a Cloud project for Earth Engine (most accounts
> created after ~2024 do), set it as an environment variable before running
> the server:
> ```bash
> export GEE_PROJECT=your-project-id      # Windows: set GEE_PROJECT=your-project-id
> ```

## 3. Train the crop classifier

Place your `FARM_NDVI.xlsx` (sheets: Sugarcane, Onion, Cotton, Paddy, Tomato,
each with columns `Farm_ID`, `Date`, `NDVI`) in the `backend/` folder, then:

```bash
python train_model.py FARM_NDVI.xlsx
```

This reproduces the notebook's lifecycle-detection + feature-engineering
pipeline on your historical data and saves the trained model to
`crop_model.pkl`. You only need to do this once (re-run it if you get more
training data later).

## 3b. Check how good the model actually is (precision, recall, etc.)

`train_model.py` trains on 100% of your data to make the best possible model
for the live site — but that means it doesn't tell you how accurate it is.
To actually measure that, run:

```bash
python evaluate_model.py FARM_NDVI.xlsx
```

This holds back 30% of your data as a test set (the model never sees it
during training), then reports **precision, recall, and F1-score for each
crop**, plus a confusion matrix image (`confusion_matrix.png`) and a 5-fold
cross-validation accuracy — same methodology as your notebook. Run this
whenever you want to check quality; it doesn't touch `crop_model.pkl`, so it
won't affect the live site either way.

## 4. Run the backend

```bash
python app.py
```

You should see:
```
Initializing Earth Engine...
Earth Engine ready.
Model loaded.
 * Running on http://127.0.0.1:5000
```

Leave this running.

## 5. Open the frontend

You have **two frontend options** — pick whichever you prefer, or use both:

### Option A: Streamlit (simplest — one command, no separate server)

```bash
cd backend
streamlit run streamlit_app.py
```

This opens a browser tab automatically. Everything — the form, the chart,
the prediction — runs inside this one command. You do **not** need to run
`app.py` separately for this option; `streamlit_app.py` calls the same
extraction/lifecycle/model functions directly.

### Option B: HTML/CSS/JS (the styled "Field Journal" site)

1. Start the API server: `python app.py` (leave it running)
2. Open `frontend/index.html` directly in your browser

This option needs **two things running**: the Flask API (`app.py`) and the
static HTML page. It exists separately because it's a more traditional
website structure (a backend API + a standalone frontend) if that's what
your submission needs to demonstrate.

## How a request flows through the system

1. You enter latitude, longitude, and a date, and hit **Scan field**.
2. `ee_extraction.py` queries Sentinel-2 SR Harmonized imagery within a
   ±12 month window of that date, filters clouds (< 20%), and computes NDVI
   at that exact point for every clear image — the same logic as the
   original GEE script, just scoped to one point instead of a batch table.
3. `lifecycle.py` smooths the NDVI series, detects peaks/valleys, and reduces
   them to candidate growth cycles (sowing → peak → harvest), picking the one
   that contains your query date.
4. `model.py` extracts the same ~13 features used in training (duration,
   growth rate, AUC, skewness, etc.) from that cycle and runs them through the
   trained RandomForest to get a crop name + confidence + full probability
   breakdown.
5. The frontend renders the crop name, a confidence meter, the NDVI curve
   with sowing/peak/harvest markers, and the runner-up crop probabilities.

## Notes / things to watch for

- **Sparse imagery**: if a location has very few cloud-free Sentinel-2 passes
  in the window, the lifecycle detector may not find a clean cycle. The API
  returns a clear error message in that case rather than a bad guess.
- **Training data quality**: the classifier's accuracy is only as good as
  `FARM_NDVI.xlsx` — more farms/seasons per crop will generalize better than
  a handful of lifecycles.
- **CORS**: `flask-cors` is enabled so the static HTML page (opened via
  `file://`) can call the local Flask API without browser errors.
