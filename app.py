import io
import zipfile

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(page_title="Stack Overflow 2025 — AI Insights", layout="wide")

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "edoardogalli/stack-overflow-annual-developer-survey-2025"
)
TARGET_PHRASE = "AI agents have increased my productivity"

# Only read fields used by this dashboard, plus the AI-agent impact response columns.
# This keeps memory use much lower than loading the full 170-column survey.
REQUIRED_COLUMNS = {
    "ResponseId",
    "Country",
    "Age",
    "DevType",
    "WorkExp",
    "YearsCode",
    "AISelect",
    "AIAgents",
    "ConvertedCompYearly",
    "JobSat",
}


@st.cache_data(ttl=24 * 60 * 60, show_spinner="Downloading and preparing the public survey data…")
def load_and_clean_data():
    """Download the public survey ZIP, load relevant columns, and clean in memory."""
    response = requests.get(DATASET_URL, timeout=(20, 180))
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(
            name for name in archive.namelist()
            if name.endswith("survey_results_public.csv")
        )
        with archive.open(csv_name) as csv_file:
            data = pd.read_csv(
                csv_file,
                low_memory=False,
                usecols=lambda column: (
                    column in REQUIRED_COLUMNS or column.startswith("AIAgentImpact")
                ),
            )

    # Basic cleaning: remove duplicate records and convert numeric fields safely.
    data = data.drop_duplicates().copy()
    if "ResponseId" in data.columns:
        data = data.drop_duplicates(subset=["ResponseId"], keep="first").copy()
    for column in ["WorkExp", "YearsCode", "ConvertedCompYearly", "JobSat"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # In the survey, each statement is stored in the response-category column
    # chosen by the respondent. Agreement is positive; neutral/disagreement is
    # not positive. If the statement was not answered, keep the target missing.
    impact_columns = [
        column for column in data.columns
        if column.startswith("AIAgentImpact")
    ]
    statement_selected = pd.Series(False, index=data.index)
    positive_selected = pd.Series(False, index=data.index)
    agreement_columns = [
        column for column in impact_columns
        if "somewhat agree" in column.lower() or "strongly agree" in column.lower()
    ]
    for column in impact_columns:
        selected = data[column].fillna("").astype(str).str.contains(
            TARGET_PHRASE, case=False, regex=False
        )
        statement_selected |= selected
        if column in agreement_columns:
            positive_selected |= selected

    data["Perceived_Productivity_Positive"] = (
        positive_selected.where(statement_selected, pd.NA).astype("Int64")
    )
    data["Country_Short"] = data.get("Country", pd.Series("Missing", index=data.index)).fillna("Missing")
    data["AI_Use_Category"] = data.get("AISelect", pd.Series("Missing", index=data.index)).fillna("Missing")
    return data.reset_index(drop=True)


st.title("Stack Overflow Developer Survey 2025")
st.subheader("AI tools, AI agents, and perceived productivity")
st.sidebar.header("Dataset and filters")
st.sidebar.write(
    "The app downloads the public 2025 survey and prepares the required fields in memory. "
    "No cleaned CSV needs to be stored in this repository. Productivity is self-reported; "
    "comparisons are descriptive, not causal."
)
st.sidebar.caption("Source: Kaggle public dataset — Stack Overflow Developer Survey 2025")

try:
    df = load_and_clean_data()
except Exception as exc:
    st.error(
        "The survey data could not be downloaded. Check the app's internet connection "
        "and try again. The app retrieves the public dataset from Kaggle at runtime."
    )
    st.exception(exc)
    st.stop()

st.success(f"Loaded and cleaned {len(df):,} survey responses in memory.")
filtered = df

if "Country_Short" in filtered.columns:
    countries = sorted(filtered["Country_Short"].dropna().astype(str).unique())
    selected_country = st.sidebar.selectbox("Country", ["All"] + countries)
    if selected_country != "All":
        filtered = filtered[filtered["Country_Short"].astype(str) == selected_country]

if "AI_Use_Category" in filtered.columns:
    categories = sorted(filtered["AI_Use_Category"].dropna().astype(str).unique())
    selected_ai = st.sidebar.multiselect(
        "AI-use frequency", categories, default=categories
    )
    filtered = filtered[filtered["AI_Use_Category"].astype(str).isin(selected_ai)]

if "WorkExp" in filtered.columns and filtered["WorkExp"].notna().any():
    low = max(0, int(filtered["WorkExp"].min()))
    high = max(low, int(filtered["WorkExp"].max()))
    if low < high:
        experience_range = st.sidebar.slider(
            "Professional experience (years)", low, high, (low, high)
        )
        filtered = filtered[
            filtered["WorkExp"].between(experience_range[0], experience_range[1])
        ]

st.header("Filtered dataset")
st.caption(f"{len(filtered):,} respondents match the selected filters.")
preview_columns = [
    column for column in [
        "Country", "Age", "DevType", "WorkExp", "YearsCode", "AISelect",
        "AIAgents", "Perceived_Productivity_Positive",
    ] if column in filtered.columns
]
st.dataframe(
    filtered[preview_columns].head(25) if preview_columns else filtered.head(25),
    use_container_width=True,
)

left, right = st.columns(2)
with left:
    st.subheader("Summary statistics")
    numeric_columns = [
        column for column in ["WorkExp", "YearsCode", "ConvertedCompYearly", "JobSat"]
        if column in filtered.columns
    ]
    if numeric_columns:
        st.dataframe(filtered[numeric_columns].describe().T, use_container_width=True)
    else:
        st.info("No selected numeric columns are available.")

with right:
    st.subheader("AI-use frequency")
    if "AI_Use_Category" in filtered.columns:
        counts = (
            filtered["AI_Use_Category"].value_counts(dropna=False)
            .rename_axis("AI use").reset_index(name="Respondents")
        )
        st.plotly_chart(
            px.bar(counts, x="AI use", y="Respondents", title="Respondents by AI-use frequency"),
            use_container_width=True,
        )

st.subheader("Perceived productivity by AI-agent use")
if {"AIAgents", "Perceived_Productivity_Positive"}.issubset(filtered.columns):
    valid = filtered.dropna(subset=["AIAgents", "Perceived_Productivity_Positive"])
    if len(valid):
        rates = (
            valid.groupby("AIAgents")["Perceived_Productivity_Positive"]
            .agg(["mean", "count"]).reset_index()
        )
        rates["Positive perception (%)"] = rates["mean"] * 100
        st.plotly_chart(
            px.bar(
                rates, x="AIAgents", y="Positive perception (%)",
                hover_data=["count"],
                title="Share agreeing that AI agents increased productivity",
            ),
            use_container_width=True,
        )
        overall_rate = valid["Perceived_Productivity_Positive"].astype(float).mean() * 100
        st.info(
            f"Among respondents with a valid answer, {overall_rate:.1f}% agreed. "
            "This is self-reported and does not show a causal effect."
        )
    else:
        st.info("No valid productivity-impact answers remain under these filters.")
else:
    st.info("Required AI-agent response fields are not available in this dataset.")

if {"WorkExp", "YearsCode"}.issubset(filtered.columns):
    st.subheader("Experience and years coding")
    plot_data = filtered.dropna(subset=["WorkExp", "YearsCode"])
    if len(plot_data) > 5000:
        plot_data = plot_data.sample(5000, random_state=42)
    st.plotly_chart(
        px.scatter(
            plot_data, x="WorkExp", y="YearsCode",
            color="AI_Use_Category" if "AI_Use_Category" in plot_data else None,
            opacity=0.55,
            title="Professional experience vs. years coding (sample)",
        ),
        use_container_width=True,
    )
