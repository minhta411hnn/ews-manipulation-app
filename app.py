"""
Early Warning System for Financial Statement Manipulation -- Vietnamese Listed Firms
Streamlit app: type a stock symbol -> see its manipulation-risk score + SHAP explanation.

Before running, place these two files (produced by Day9a_export_artifacts.ipynb) in the
same folder as this script:
    - model_bundle.joblib
    - lookup_snapshot.parquet

Run locally with:  streamlit run app.py
Deploy for free on Streamlit Community Cloud by pushing this folder (app.py, requirements.txt,
model_bundle.joblib, lookup_snapshot.parquet) to a GitHub repo and connecting it there.
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="EWS - Financial Statement Manipulation Risk", layout="wide")

ARTIFACT_DIR = "."  # same folder as this script


@st.cache_resource
def load_model_bundle():
    return joblib.load(f"{ARTIFACT_DIR}/model_bundle.joblib")


@st.cache_data
def load_lookup():
    return pd.read_parquet(f"{ARTIFACT_DIR}/lookup_snapshot.parquet")


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def positive_class_shap(explainer, X):
    """Version-robust extraction of the positive-class SHAP matrix + base value."""
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        matrix = raw[1]
        base = explainer.expected_value[1] if isinstance(
            explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    elif np.ndim(raw) == 3:
        matrix = raw[:, :, 1]
        base = explainer.expected_value[1] if isinstance(
            explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        matrix = raw
        base = explainer.expected_value
    return matrix, base


def risk_band(prob: float) -> tuple[str, str]:
    if prob >= 0.7:
        return "High risk", "🔴"
    if prob >= 0.4:
        return "Moderate risk", "🟠"
    return "Low risk", "🟢"


# ---------------------------------------------------------------------------
bundle = load_model_bundle()
model = bundle["model"]
imputer = bundle["imputer"]
FEATURE_COLS = bundle["feature_cols"]
lookup = load_lookup()
explainer = get_explainer(model)

st.title("📊 Early Warning System — Financial Statement Manipulation Risk")
st.caption(
    "Vietnamese listed firms · Beneish-family features + LightGBM · trained on "
    f"{bundle['train_years'][0]}–{bundle['train_years'][-1]}, held-out tested on {bundle['test_year']}"
)

with st.sidebar:
    st.header("Look up a firm")
    symbols = sorted(lookup["symbol"].dropna().unique().tolist()) if "symbol" in lookup.columns else []
    symbol = st.selectbox("Stock symbol", symbols, index=None, placeholder="e.g. VCA")

    if symbol:
        firm_rows = lookup[lookup["symbol"] == symbol].sort_values("year", ascending=False)
        years_available = firm_rows["year"].tolist()
        year = st.selectbox("Year", years_available, index=0)
    else:
        year = None

    st.divider()
    st.caption(
        "⚠️ Research tool, not investment advice. The risk score reflects statistical similarity "
        "to firms with extreme abnormal accruals in the training data; it is not proof of "
        "manipulation. See the thesis's §5.3 for limitations (including a labeling-construction "
        "caveat on the `TATA` feature)."
    )

if not symbol:
    st.info("⬅️ Select a stock symbol from the sidebar to see its risk score and explanation.")
    st.stop()

row = lookup[(lookup["symbol"] == symbol) & (lookup["year"] == year)].iloc[0]
X_row = pd.DataFrame([row[FEATURE_COLS].values], columns=FEATURE_COLS)
X_row_imputed = pd.DataFrame(imputer.transform(X_row), columns=FEATURE_COLS)

pred_prob = float(model.predict_proba(X_row_imputed)[:, 1][0])
band_label, band_icon = risk_band(pred_prob)

col1, col2, col3 = st.columns(3)
col1.metric("Risk score", f"{pred_prob:.1%}")
col2.metric("Risk band", f"{band_icon} {band_label}")
if "Y" in row and pd.notna(row["Y"]):
    col3.metric("Flagged in labeled data (top-quintile |DAC|)", "Yes" if int(row["Y"]) == 1 else "No")

if row.get("is_test_year", False):
    st.caption(f"ℹ️ {year} is in the held-out test set (model never trained on this year's labels).")
else:
    st.caption(f"ℹ️ {year} is in the training window -- this score is an in-sample fit, not a holdout evaluation.")

st.divider()

# --- SHAP waterfall for this firm-year ---
st.subheader(f"Why this score -- {symbol} ({year})")
shap_matrix, base_value = positive_class_shap(explainer, X_row_imputed)

exp = shap.Explanation(
    values=shap_matrix[0],
    base_values=base_value,
    data=X_row_imputed.iloc[0].values,
    feature_names=FEATURE_COLS,
)

fig, ax = plt.subplots(figsize=(9, 6))
shap.plots.waterfall(exp, show=False, max_display=14)
st.pyplot(fig, clear_figure=True)

st.caption(
    "Reads left to right from the average predicted risk (E[f(x)]) to this firm's score. "
    "Red bars push the risk score up, blue bars pull it down."
)

st.divider()

# --- Raw feature values vs. industry-year median, for context ---
st.subheader("Feature values vs. industry-year peers")
if "industry_name" in lookup.columns:
    peers = lookup[(lookup["industry_name"] == row.get("industry_name")) & (lookup["year"] == year)]
    peer_median = peers[FEATURE_COLS].median()
    compare_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        symbol: row[FEATURE_COLS].values,
        "Industry-year median": peer_median.values,
    }).set_index("Feature")
    st.dataframe(compare_df.style.format("{:.3f}"), use_container_width=True)
else:
    st.dataframe(row[FEATURE_COLS].to_frame(name=symbol), use_container_width=True)
