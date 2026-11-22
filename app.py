"""Streamlit dashboard: streamlit run app.py   (deploy free on Streamlit Community Cloud, see README)"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

R = Path(__file__).parent / "reports"
st.set_page_config(page_title="Candlestick pattern scanner", layout="wide")
st.title("Candlestick pattern scanner")
st.caption("Research tool, not investment advice. Patterns are statistical hints; check the scorecard before trusting one.")

need = [Path(__file__).parent / "models" / "meta.json", R / "latest_signals.csv", R / "pattern_scorecard.csv", R / "feature_importance.csv"]
if not all(p.exists() for p in need):
    st.error("No trained model or reports found yet.")
    st.markdown("Unzip the project to a normal folder, then from inside it run:\n\n"
                "1. `python src/data_collection.py` (downloads prices, needs internet)\n"
                "2. `python src/run_all.py` (trains, evaluates, builds the signals)\n"
                "3. `streamlit run app.py`")
    st.stop()
meta = json.loads(need[0].read_text())
c1, c2 = st.columns(2)
c1.metric("Validation AUC (logistic / GBM)", f'{meta["val_auc"]["logistic"]:.3f} / {meta["val_auc"]["gbm"]:.3f}')
c2.metric("Features", len(meta["features"]))

sig = pd.read_csv(R / "latest_signals.csv")
mm, tm = R / "model_metrics.json", R / "trade_metrics.csv"
if mm.exists() and tm.exists():
    m = json.loads(mm.read_text())
    st.subheader("Test-year performance")
    k = st.columns(4)
    k[0].metric("AUC (test)", f'{m["auc_test"]:.3f}'); k[1].metric("Precision (top 10%)", f'{m["precision"]:.3f}', f'base {m["base_rate"]:.3f}', delta_color="off")
    k[2].metric("Recall", f'{m["recall"]:.3f}'); k[3].metric("F1", f'{m["f1"]:.3f}')
    st.dataframe(pd.read_csv(tm), use_container_width=True)
t1, t2, t3 = st.tabs(["Today's signals", "Pattern scorecard", "Feature importance"])
with t1:
    only = st.checkbox("Only patterns confirmed in the test year", value=False)
    trend = st.multiselect("Trend", sorted(sig["trend"].dropna().unique()), default=list(sig["trend"].dropna().unique()))
    v = sig[(sig["pattern"].notna()) & sig["trend"].isin(trend)]
    st.dataframe(v[v["confirmed"]] if only else v, use_container_width=True)
with t2:
    sc = pd.read_csv(R / "pattern_scorecard.csv")
    st.dataframe(sc[sc["confirmed"]] if st.checkbox("Confirmed only", True) else sc, use_container_width=True)
with t3:
    st.bar_chart(pd.read_csv(R / "feature_importance.csv").head(15).set_index("feature"))
