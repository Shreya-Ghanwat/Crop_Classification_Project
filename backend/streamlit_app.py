"""
Streamlit frontend — Crop identification from NDVI lifecycle.

Unlike the HTML/CSS/JS frontend (which needs app.py running separately as
an API), this file IS the whole application: it directly imports the same
ee_extraction / lifecycle / model functions and renders the UI itself.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import ee_extraction
import lifecycle
import model as model_module

st.set_page_config(
    page_title="Field Journal — Know Your Crop",
    page_icon="🌾",
    layout="centered",
)

# ------------------------------------------------------------------
# One-time setup: Earth Engine auth + load trained model.
# @st.cache_resource makes this run only ONCE per server session,
# not on every button click / page rerun.
# ------------------------------------------------------------------

@st.cache_resource
def setup():
    ee_extraction.initialize_earth_engine()
    model = model_module.load_model()
    return model


with st.spinner("Connecting to Earth Engine…"):
    model = setup()

if model is None:
    st.error(
        "No trained model found (`crop_model.pkl`). "
        "Run `python train_model.py FARM_NDVI.xlsx` first, then restart this app."
    )
    st.stop()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.markdown(
    "<span style='font-family:monospace;font-size:12px;letter-spacing:0.1em;"
    "color:#3a6b3e;background:rgba(58,107,62,0.08);border:1px solid rgba(58,107,62,0.2);"
    "border-radius:20px;padding:4px 12px;'>FROM SATELLITE TO SEASON</span>",
    unsafe_allow_html=True,
)
st.title("🌾 What's growing in that field?")
st.write(
    "Give a coordinate and a date. We'll trace the crop's whole growing season "
    "from real Sentinel-2 satellite imagery — sowing, peak growth, harvest — "
    "and identify the crop."
)

# ------------------------------------------------------------------
# Input form
# ------------------------------------------------------------------

with st.form("query_form"):
    col1, col2 = st.columns(2)
    with col1:
        coords_raw = st.text_input(
            "Coordinates (lat, long)",
            placeholder="18.5204, 73.8567",
        )
    with col2:
        obs_date = st.date_input("Observation date")

    submitted = st.form_submit_button("🌱 Trace the field", use_container_width=True)

# ------------------------------------------------------------------
# On submit: run the pipeline and render results
# ------------------------------------------------------------------

if submitted:
    parts = [p.strip() for p in coords_raw.replace(",", " ").split() if p.strip()]

    if len(parts) != 2:
        st.error('Please enter coordinates as "latitude, longitude", e.g. 18.5204, 73.8567')
        st.stop()

    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        st.error("Coordinates must be numbers.")
        st.stop()

    date_str = obs_date.strftime("%Y-%m-%d")

    with st.spinner("Pulling Sentinel-2 imagery and tracing the growing season…"):
        try:
            ndvi_df = ee_extraction.extract_ndvi_timeseries(lat, lon, date_str)

            if ndvi_df.empty or len(ndvi_df) < 5:
                st.error(
                    "Not enough cloud-free Sentinel-2 imagery was found for this "
                    "location/date. Try a different point or date."
                )
                st.stop()

            farm_smoothed, cycles = lifecycle.detect_lifecycles(ndvi_df)

            if not cycles:
                st.error("No clear crop growth cycle was detected in the NDVI signal for this period.")
                st.stop()

            chosen_cycle = lifecycle.pick_query_cycle(farm_smoothed, cycles, date_str)
            features = lifecycle.extract_features_from_cycle(farm_smoothed, chosen_cycle)
            predicted_crop, confidence, all_probs = model_module.predict_crop(model, features)

        except Exception as e:
            st.error(f"Something went wrong while processing this request: {e}")
            st.stop()

    # ---- Result headline ----
    st.divider()
    st.markdown(f"### 🌾 Identified as: **{predicted_crop}**")
    st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")

    # ---- Season timeline ----
    sv_date = chosen_cycle["start_date"].strftime("%Y-%m-%d")
    pk_date = chosen_cycle["peak_date"].strftime("%Y-%m-%d")
    ev_date = chosen_cycle["end_date"].strftime("%Y-%m-%d")

    c1, c2, c3 = st.columns(3)
    c1.metric("🌱 Sowing", sv_date)
    c2.metric("🌿 Peak growth", pk_date)
    c3.metric("🌾 Harvest window", ev_date)

    # ---- NDVI curve chart ----
    st.markdown("#### Growth curve")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=farm_smoothed["Date"], y=farm_smoothed["NDVI_smooth"],
        mode="lines", name="NDVI",
        line=dict(color="#3a6b3e", width=2.5),
        fill="tozeroy", fillcolor="rgba(58,107,62,0.15)",
    ))

    marker_dates = [chosen_cycle["start_date"], chosen_cycle["peak_date"], chosen_cycle["end_date"]]
    marker_ndvi = [chosen_cycle["start_ndvi"], chosen_cycle["peak_ndvi"], chosen_cycle["end_ndvi"]]
    marker_labels = ["Sowing", "Peak growth", "Harvest window"]
    marker_colors = ["#d9a441", "#3a6b3e", "#b5563c"]

    fig.add_trace(go.Scatter(
        x=marker_dates, y=marker_ndvi,
        mode="markers+text", text=marker_labels, textposition="top center",
        marker=dict(size=10, color=marker_colors, line=dict(width=2, color="white")),
        showlegend=False,
    ))

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None, yaxis_title="NDVI",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Other candidates ----
    st.markdown("#### Other candidates")
    prob_df = (
        pd.DataFrame(list(all_probs.items()), columns=["Crop", "Probability"])
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )
    prob_df["Probability"] = (prob_df["Probability"] * 100).round(1)
    st.dataframe(
        prob_df,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability", format="%.1f%%", min_value=0, max_value=100
            )
        },
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.caption("Sentinel-2 SR Harmonized · Google Earth Engine · RandomForest classifier")
