import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Stack Overflow 2025 — AI Insights", layout="wide")
DATA_PATH = Path(__file__).parent / "clean_data.csv"
@st.cache_data
def load_data(path):
    return pd.read_csv(path, low_memory=False)

df = load_data(DATA_PATH)
st.title("Stack Overflow Developer Survey 2025")
st.subheader("AI tools, AI agents, and perceived productivity")
st.sidebar.header("Dataset and filters")
st.sidebar.write("Public, voluntary 2025 developer survey. Productivity is self-reported; comparisons are descriptive, not causal.")

filtered = df.copy()
if "Country_Short" in filtered.columns:
    countries = sorted(filtered["Country_Short"].dropna().astype(str).unique())
    selected_country = st.sidebar.selectbox("Country", ["All"] + countries)
    if selected_country != "All": filtered = filtered[filtered["Country_Short"].astype(str) == selected_country]
if "AI_Use_Category" in filtered.columns:
    categories = sorted(filtered["AI_Use_Category"].dropna().astype(str).unique())
    selected_ai = st.sidebar.multiselect("AI-use frequency", categories, default=categories)
    filtered = filtered[filtered["AI_Use_Category"].astype(str).isin(selected_ai)]
if "WorkExp" in filtered.columns and filtered["WorkExp"].notna().any():
    lo = int(max(0, filtered["WorkExp"].min()))
    hi = int(max(lo, filtered["WorkExp"].max()))
    if lo < hi:
        exp_range = st.sidebar.slider("Professional experience (years)", lo, hi, (lo, hi))
        filtered = filtered[filtered["WorkExp"].between(exp_range[0], exp_range[1])]

st.header("Filtered dataset")
st.caption(f"{len(filtered):,} respondents match the selected filters.")
preview = [c for c in ["Country", "Age", "DevType", "WorkExp", "YearsCode", "AISelect", "AIAgents", "Perceived_Productivity_Positive"] if c in filtered.columns]
st.dataframe(filtered[preview].head(25) if preview else filtered.head(25), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Summary statistics")
    numeric = [c for c in ["WorkExp", "YearsCode", "ConvertedCompYearly"] if c in filtered.columns]
    if numeric: st.dataframe(filtered[numeric].describe().T, use_container_width=True)
    else: st.info("No selected numeric columns available.")
with right:
    st.subheader("AI-use frequency")
    if "AI_Use_Category" in filtered.columns:
        counts = filtered["AI_Use_Category"].value_counts(dropna=False).rename_axis("AI use").reset_index(name="Respondents")
        st.plotly_chart(px.bar(counts, x="AI use", y="Respondents", title="Respondents by AI-use frequency"), use_container_width=True)

st.subheader("Perceived productivity by AI-agent use")
if {"AIAgents", "Perceived_Productivity_Positive"}.issubset(filtered.columns):
    valid = filtered.dropna(subset=["AIAgents", "Perceived_Productivity_Positive"])
    if len(valid):
        rates = valid.groupby("AIAgents")["Perceived_Productivity_Positive"].agg(["mean", "count"]).reset_index()
        rates["Positive perception (%)"] = rates["mean"] * 100
        st.plotly_chart(px.bar(rates, x="AIAgents", y="Positive perception (%)", hover_data=["count"],
                               title="Share selecting the positive productivity statement"), use_container_width=True)
        st.info("Among respondents with a valid impact response, the overall positive share is "
                f"{valid['Perceived_Productivity_Positive'].mean()*100:.1f}%. This is self-reported association, not causal evidence.")
    else: st.info("No valid productivity-impact answers remain under these filters.")
else: st.info("Required AI-agent response fields are not available in this dataset file.")

if {"WorkExp", "YearsCode"}.issubset(filtered.columns):
    st.subheader("Experience and years coding")
    plot_data = filtered.dropna(subset=["WorkExp", "YearsCode"])
    if len(plot_data) > 5000: plot_data = plot_data.sample(5000, random_state=42)
    st.plotly_chart(px.scatter(plot_data, x="WorkExp", y="YearsCode", color="AI_Use_Category" if "AI_Use_Category" in plot_data else None,
                               opacity=.55, title="Professional experience vs. years coding (sample)"), use_container_width=True)
