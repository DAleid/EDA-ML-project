from pathlib import Path
import io
import re
import warnings
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SO 2025 · AI Productivity",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#172033; --muted:#68758a; --teal:#0f766e; --teal-light:#dff4f0; --line:#e5eaf1; --panel:#ffffff; --bg:#f6f8fb; }
    .stApp { background: var(--bg); color: var(--ink); }
    .block-container { max-width: 1440px; padding: 2rem 3rem 4rem; }
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-0.02em; }
    h1 { font-size:2.45rem !important; margin-bottom:.25rem; }
    h2 { font-size:1.4rem !important; margin-top:1.5rem; }
    .eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.13em; font-weight:700; font-size:.72rem; margin-bottom:.25rem; }
    .subtitle { color:var(--muted); font-size:1rem; margin-bottom:1.25rem; }
    .hero { background:linear-gradient(135deg,#ffffff 0%,#eefaf7 100%); border:1px solid #d8ebe8; border-radius:18px; padding:1.5rem 1.7rem; margin-bottom:1rem; box-shadow:0 10px 30px rgba(25,55,75,.04); }
    .hero-note { color:#4d5b70; font-size:.92rem; line-height:1.55; max-width:920px; }
    .kpi { background:var(--panel); border:1px solid var(--line); border-radius:15px; padding:1rem 1.1rem; min-height:112px; box-shadow:0 4px 18px rgba(25,55,75,.035); }
    .kpi-label { color:var(--muted); font-size:.77rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
    .kpi-value { color:var(--ink); font-size:1.85rem; font-weight:750; margin:.22rem 0; }
    .kpi-help { color:var(--muted); font-size:.76rem; }
    .insight { background:#f0faf8; border-left:4px solid var(--teal); border-radius:10px; padding:.85rem 1rem; color:#314455; line-height:1.5; margin:.65rem 0 1rem; }
    .warning { background:#fff8e8; border-left:4px solid #d99100; border-radius:10px; padding:.8rem 1rem; color:#5c4a22; line-height:1.5; }
    .section-copy { color:var(--muted); line-height:1.55; margin-top:-.35rem; margin-bottom:1rem; }
    .small-muted { color:var(--muted); font-size:.82rem; }
    .stTabs [data-baseweb="tab-list"] { gap:1.1rem; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { height:48px; padding:0 .25rem; color:var(--muted); }
    .stTabs [aria-selected="true"] { color:var(--teal); border-bottom-color:var(--teal); }
    div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:.9rem; border-radius:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).parent / "survey_results_public.csv"
DATASET_SLUG = "edoardogalli/stack-overflow-annual-developer-survey-2025"

REQUIRED_COLUMNS = [
    "ResponseId", "MainBranch", "Age", "EdLevel", "Employment", "WorkExp", "YearsCode",
    "DevType", "OrgSize", "RemoteWork", "Country", "AIThreat", "AIModelsChoice", "AISelect",
    "AISent", "AIAcc", "AIComplex", "AIAgents", "AIAgentChange", "AIFrustration",
    "AIAgentImpactSomewhat agree", "AIAgentImpactNeutral", "AIAgentImpactSomewhat disagree",
    "AIAgentImpactStrongly agree", "AIAgentImpactStrongly disagree",
]


def tidy_label(value: str) -> str:
    value = str(value).replace("’", "'").strip()
    return value


@st.cache_data(show_spinner="Loading the 2025 survey…")
def load_data(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    header = pd.read_csv(path, nrows=0)
    available = [c for c in REQUIRED_COLUMNS if c in header.columns]
    data = pd.read_csv(path, usecols=available, low_memory=False)
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].map(lambda x: tidy_label(x) if pd.notna(x) else x)
    for col in ["WorkExp", "YearsCode"]:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    impact_cols = [c for c in data.columns if c.startswith("AIAgentImpact")]
    impact_text = data[impact_cols].fillna("").astype(str).agg(" | ".join, axis=1)
    target_phrase = "AI agents have increased my productivity"
    data["Perceived_Productivity_Positive"] = impact_text.str.contains(
        target_phrase, case=False, regex=False
    ).astype("Int64")
    data["AI_Impact_Response_Available"] = impact_text.str.strip().ne("")
    data["Uses_AI_Daily"] = data["AISelect"].eq("Yes, I use AI tools daily").astype(int)
    data["Uses_AI_Weekly_or_more"] = data["AISelect"].isin(
        ["Yes, I use AI tools daily", "Yes, I use AI tools weekly"]
    ).astype(int)
    data["Uses_AI_Agents"] = data["AIAgents"].fillna("").str.startswith("Yes").astype(int)
    data["AI_Agent_Daily"] = data["AIAgents"].eq("Yes, I use AI agents at work daily").astype(int)
    data["AI_Changed_Work"] = data["AIAgentChange"].fillna("").str.startswith("Yes").astype(int)
    data["AI_Frustration_Count"] = data["AIFrustration"].fillna("").str.count(";") + data["AIFrustration"].notna().astype(int)
    data["DevType_Short"] = data["DevType"].fillna("Missing").str.split(",").str[0]
    data["Country_Short"] = data["Country"].fillna("Missing")
    return data


@st.cache_data(show_spinner="Downloading the public 2025 survey CSV…")
def ensure_data_file() -> str:
    if DATA_PATH.exists():
        return str(DATA_PATH)
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith("survey_results_public.csv"))
        temp_path = Path("/tmp/survey_results_public.csv")
        temp_path.write_bytes(archive.read(csv_name))
    return str(temp_path)


@st.cache_resource(show_spinner="Training the classification model…")
def train_model(data: pd.DataFrame):
    model_df = data[data["Perceived_Productivity_Positive"].notna()].copy()
    feature_cols = [
        "AISelect", "AISent", "AIAcc", "AIComplex", "AIAgents", "AIAgentChange", "AIThreat",
        "AIModelsChoice", "Age", "WorkExp", "YearsCode", "DevType_Short", "OrgSize",
        "RemoteWork", "EdLevel", "Employment",
    ]
    feature_cols = [c for c in feature_cols if c in model_df.columns]
    X = model_df[feature_cols].copy()
    y = model_df["Perceived_Productivity_Positive"].astype(int)
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numerical = [c for c in X.columns if c not in categorical]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y
    )
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("preprocess", prep), ("model", LogisticRegression(max_iter=500, class_weight="balanced"))])
    baseline = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    base_pred = baseline.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced accuracy": balanced_accuracy_score(y_test, pred),
        "F1 · positive": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "Baseline accuracy": accuracy_score(y_test, base_pred),
        "Test rows": len(y_test),
        "Positive rate": y.mean(),
    }
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_[0]
    importance = pd.DataFrame({"feature": names, "coefficient": coefs})
    importance["abs_coefficient"] = importance["coefficient"].abs()
    importance = importance.sort_values("abs_coefficient", ascending=False)
    return model, metrics, report, cm, importance, feature_cols


def pct(value, digits=1):
    return f"{value * 100:.{digits}f}%" if pd.notna(value) else "—"


def plotly_defaults(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#172033", size=13),
        title_font=dict(color="#172033", size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#dfe5ec", zeroline=False, color="#172033", tickfont=dict(color="#172033", size=12), title_font=dict(color="#172033", size=13))
    fig.update_yaxes(showgrid=False, zeroline=False, color="#172033", tickfont=dict(color="#172033", size=12), title_font=dict(color="#172033", size=13))
    return fig


def short_ai_label(value):
    replacements = {
        "Yes, I use AI tools daily": "Daily",
        "Yes, I use AI tools weekly": "Weekly",
        "Yes, I use AI tools monthly or infrequently": "Monthly / infrequent",
        "No, and I don't plan to": "No · no plans",
        "No, but I plan to soon": "No · plan soon",
    }
    return replacements.get(str(value), str(value))


try:
    data_file = ensure_data_file()
    df = load_data(data_file)
except Exception as exc:
    st.error(f"The survey data could not be loaded: {exc}")
    st.stop()

# Header
st.markdown('<div class="eyebrow">Stack Overflow Developer Survey · 2025</div>', unsafe_allow_html=True)
st.title("AI adoption & perceived productivity")
st.markdown(
    '<div class="subtitle">A clear, interactive view of how developers use AI tools and agents — and how they describe the effect on their work.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><div class="hero-note"><b>Read this dashboard correctly:</b> the productivity outcome is self-reported perception, not measured output from work logs. Use the charts to understand patterns and associations in the survey, not to claim that AI caused a productivity increase.</div></div>',
    unsafe_allow_html=True,
)

# Sidebar filters
with st.sidebar:
    st.markdown("## Dashboard filters")
    st.caption("All overview charts update to match these selections. The model benchmark remains fixed on the full eligible survey sample for reproducibility.")
    employment_options = sorted(df["Employment"].dropna().unique().tolist())
    selected_employment = st.multiselect("Employment status", employment_options, default=employment_options)
    country_counts = df["Country_Short"].value_counts()
    country_options = country_counts.head(25).index.tolist()
    selected_countries = st.multiselect("Country · top 25 + Missing", country_options, default=[])
    max_exp = int(np.nanmax(df["WorkExp"])) if df["WorkExp"].notna().any() else 50
    exp_range = st.slider("Professional experience (years)", 0, max(1, min(max_exp, 50)), (0, max(1, min(max_exp, 50))))
    ai_options = sorted(df["AISelect"].dropna().unique().tolist())
    selected_ai = st.multiselect("AI use frequency", ai_options, default=ai_options)
    agent_options = sorted(df["AIAgents"].dropna().unique().tolist())
    selected_agents = st.multiselect("AI-agent use", agent_options, default=agent_options)
    if st.button("Reset filters", use_container_width=True):
        st.rerun()
    st.markdown("---")
    st.markdown("**Data note**")
    st.caption(f"{len(df):,} survey responses · {df.shape[1]} analysis fields · official 2025 survey format")

filtered = df.copy()
if selected_employment:
    filtered = filtered[filtered["Employment"].isin(selected_employment)]
if selected_countries:
    filtered = filtered[filtered["Country_Short"].isin(selected_countries)]
filtered = filtered[filtered["WorkExp"].between(exp_range[0], exp_range[1], inclusive="both") | filtered["WorkExp"].isna()]
if selected_ai:
    filtered = filtered[filtered["AISelect"].isin(selected_ai) | filtered["AISelect"].isna()]
if selected_agents:
    filtered = filtered[filtered["AIAgents"].isin(selected_agents) | filtered["AIAgents"].isna()]

model, model_metrics, model_report, cm, importance, model_features = train_model(df)

# KPI strip
eligible = filtered["Perceived_Productivity_Positive"].notna()
kpis = [
    ("Visible respondents", f"{len(filtered):,}", "after sidebar filters"),
    ("Use AI daily", pct(filtered["Uses_AI_Daily"].mean()), "among visible respondents"),
    ("Use AI agents", pct(filtered["Uses_AI_Agents"].mean()), "among visible respondents"),
    ("Positive perception", pct(filtered.loc[eligible, "Perceived_Productivity_Positive"].mean()), "AI agents increased productivity"),
]
cols = st.columns(4)
for col, (label, value, help_text) in zip(cols, kpis):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-help">{help_text}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tabs = st.tabs(["Overview", "AI landscape", "Relationships", "Model results", "Methodology"])

with tabs[0]:
    st.header("Executive overview")
    st.markdown('<div class="section-copy">Start here for the main signal. The sample is large, but the questions describe developers’ own views of their work.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.15, .85])
    with c1:
        counts = filtered["AISelect"].fillna("Missing").map(short_ai_label).value_counts().rename_axis("AI use frequency").reset_index(name="Respondents")
        counts = counts.sort_values("Respondents")
        fig = px.bar(counts, x="Respondents", y="AI use frequency", orientation="h", text="Respondents", color_discrete_sequence=["#147d73"])
        fig.update_traces(textposition="outside", textfont=dict(color="#172033", size=12))
        fig.update_coloraxes(showscale=False)
        fig.update_layout(title="How often do developers use AI?", xaxis_title="Respondents", yaxis_title=None)
        st.plotly_chart(plotly_defaults(fig, 430), use_container_width=True)
        st.caption("Counts show how often visible respondents say they use AI in their development process.")
    with c2:
        positive = filtered[filtered["Perceived_Productivity_Positive"].notna()].copy()
        positive["Outcome"] = np.where(positive["Perceived_Productivity_Positive"].eq(1), "Positive perception", "Not positive")
        outcome = positive["Outcome"].value_counts().rename_axis("Outcome").reset_index(name="Respondents")
        fig = px.pie(outcome, values="Respondents", names="Outcome", hole=.64, color="Outcome", color_discrete_map={"Positive perception":"#147d73", "Not positive":"#aebdce"})
        fig.update_traces(textposition="inside", textinfo="percent", textfont=dict(color="white", size=14))
        fig.update_layout(title="Perceived productivity impact")
        st.plotly_chart(plotly_defaults(fig, 430), use_container_width=True)
        st.caption(f"Eligible base: {len(positive):,} visible respondents with a productivity-impact response.")
    st.markdown('<div class="insight"><b>What stands out:</b> daily AI use is common in this developer sample, while perceived productivity is best read alongside trust, sentiment, and the limitations of self-reporting.</div>', unsafe_allow_html=True)
    st.subheader("Respondent composition")
    comp1, comp2, comp3 = st.columns(3)
    with comp1:
        role = filtered["DevType_Short"].value_counts().head(10).sort_values().rename_axis("Role").reset_index(name="Respondents")
        fig = px.bar(role, x="Respondents", y="Role", orientation="h", color_discrete_sequence=["#5b8def"])
        st.plotly_chart(plotly_defaults(fig, 360), use_container_width=True)
    with comp2:
        exp = filtered["WorkExp"].dropna()
        fig = px.histogram(exp, x="WorkExp", nbins=25, color_discrete_sequence=["#0f766e"])
        fig.update_layout(xaxis_title="Professional experience (years)", yaxis_title="Respondents")
        st.plotly_chart(plotly_defaults(fig, 360), use_container_width=True)
    with comp3:
        country = filtered["Country_Short"].value_counts().head(10).sort_values().rename_axis("Country").reset_index(name="Respondents")
        fig = px.bar(country, x="Respondents", y="Country", orientation="h", color_discrete_sequence=["#d38b36"])
        st.plotly_chart(plotly_defaults(fig, 360), use_container_width=True)

with tabs[1]:
    st.header("AI landscape")
    st.markdown('<div class="section-copy">Explore adoption, agents, sentiment, trust, and the practical parts of development work where respondents use AI.</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        ai = filtered["AISelect"].fillna("Missing").map(short_ai_label).value_counts().rename_axis("Frequency").reset_index(name="Respondents")
        ai["Share"] = ai["Respondents"] / ai["Respondents"].sum()
        ai = ai.sort_values("Respondents")
        fig = px.bar(ai, x="Respondents", y="Frequency", orientation="h", text=ai["Share"].map(lambda x: f"{x:.0%}"), color_discrete_sequence=["#147d73"])
        fig.update_traces(textposition="outside")
        fig.update_layout(title="AI use frequency · share of visible sample", yaxis_title=None)
        st.plotly_chart(plotly_defaults(fig), use_container_width=True)
        st.caption("Percent labels are shares of the filtered response base, including visible missing responses.")
    with right:
        agents = filtered["AIAgents"].fillna("Missing").value_counts().rename_axis("AI-agent use").reset_index(name="Respondents")
        fig = px.bar(agents.sort_values("Respondents"), x="Respondents", y="AI-agent use", orientation="h", color_discrete_sequence=["#5b8def"])
        st.plotly_chart(plotly_defaults(fig), use_container_width=True)
        st.caption("AI agents are autonomous or semi-autonomous systems that operate with limited direct intervention.")
    left, right = st.columns(2)
    with left:
        sent = filtered["AISent"].fillna("Missing").value_counts().rename_axis("Stance").reset_index(name="Respondents")
        fig = px.bar(sent, x="Stance", y="Respondents", color="Respondents", color_continuous_scale=["#e7eef8", "#5b8def"])
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_defaults(fig), use_container_width=True)
    with right:
        trust = filtered["AIAcc"].fillna("Missing").value_counts().rename_axis("Trust in accuracy").reset_index(name="Respondents")
        fig = px.bar(trust, x="Trust in accuracy", y="Respondents", color="Respondents", color_continuous_scale=["#fbe7ca", "#d38b36"])
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_defaults(fig), use_container_width=True)
    st.markdown('<div class="insight"><b>Interpretation:</b> adoption does not automatically mean trust. A useful dashboard read is the combination of frequency, confidence in accuracy, and perceived impact.</div>', unsafe_allow_html=True)

with tabs[2]:
    st.header("Relationships & findings")
    st.markdown('<div class="section-copy">These charts compare groups inside the survey. They are associations, not causal effects.</div>', unsafe_allow_html=True)
    relation = filtered[filtered["Perceived_Productivity_Positive"].notna()].copy()
    relation["Experience band"] = pd.cut(relation["WorkExp"], [-np.inf, 2, 5, 10, 20, np.inf], labels=["0–2", "3–5", "6–10", "11–20", "20+"])
    exp_rate = relation.groupby("Experience band", observed=False)["Perceived_Productivity_Positive"].agg(["count", "mean"]).reset_index()
    exp_rate["Positive perception"] = exp_rate["mean"] * 100
    fig = px.bar(exp_rate, x="Experience band", y="Positive perception", text=exp_rate["Positive perception"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—"), color="Positive perception", color_continuous_scale=["#d9f0ec", "#0f766e"])
    fig.update_coloraxes(showscale=False); fig.update_yaxes(range=[0,100], title="Positive perception (%)")
    st.plotly_chart(plotly_defaults(fig), use_container_width=True)
    st.caption("Experience bands include only respondents with a productivity-impact response and an available work-experience value.")
    col1, col2 = st.columns(2)
    with col1:
        cross = pd.crosstab(relation["AISent"].fillna("Missing"), relation["AIAcc"].fillna("Missing"), values=relation["Perceived_Productivity_Positive"], aggfunc="mean") * 100
        fig = px.imshow(cross.astype(float), text_auto=".1f", color_continuous_scale="YlGnBu", aspect="auto", labels=dict(x="Trust in AI accuracy", y="Stance on AI", color="Positive (%)"))
        st.plotly_chart(plotly_defaults(fig, 420), use_container_width=True)
    with col2:
        group = relation.groupby("AISelect")["Perceived_Productivity_Positive"].agg(["count", "mean"]).reset_index()
        group["Positive perception"] = group["mean"] * 100
        fig = px.scatter(group, x="count", y="Positive perception", size="count", text="AISelect", color="Positive perception", color_continuous_scale="Tealgrn")
        fig.update_traces(textposition="top center")
        fig.update_yaxes(range=[0,100], title="Positive perception (%)")
        fig.update_xaxes(title="Eligible respondents")
        st.plotly_chart(plotly_defaults(fig, 420), use_container_width=True)
    st.subheader("Key findings from the visible sample")
    findings = []
    if len(filtered):
        top_tool = filtered["AISelect"].mode(dropna=True)
        if len(top_tool): findings.append(f"The most common AI-use response is **{top_tool.iloc[0]}** in the current filter context.")
        daily = filtered["Uses_AI_Daily"].mean()
        if pd.notna(daily): findings.append(f"**{daily:.1%}** of visible respondents report daily AI use.")
        agent = filtered["Uses_AI_Agents"].mean()
        if pd.notna(agent): findings.append(f"**{agent:.1%}** of visible respondents report using AI agents in some form.")
        if relation["Perceived_Productivity_Positive"].notna().any():
            findings.append(f"Among eligible visible responses, **{relation['Perceived_Productivity_Positive'].mean():.1%}** include the statement that AI agents increased productivity.")
    for finding in findings:
        st.markdown(f'<div class="insight">{finding}</div>', unsafe_allow_html=True)

with tabs[3]:
    st.header("Model results")
    st.markdown('<div class="section-copy">A Logistic Regression classifier predicts whether a respondent reports that AI agents increased productivity. The benchmark is trained on the full eligible survey sample, not on the filtered slice.</div>', unsafe_allow_html=True)
    if model_metrics["ROC-AUC"] > .9:
        st.markdown('<div class="warning"><b>Important:</b> a very high score here should not be read as proof of real-world predictive power. The target is a survey response, and several predictors describe closely related AI attitudes and usage patterns.</div>', unsafe_allow_html=True)
    metric_cols = st.columns(5)
    for col, (label, key) in zip(metric_cols, [("Accuracy", "Accuracy"), ("Balanced accuracy", "Balanced accuracy"), ("F1 · positive", "F1 · positive"), ("ROC-AUC", "ROC-AUC"), ("Baseline accuracy", "Baseline accuracy")]):
        col.metric(label, f"{model_metrics[key]:.3f}")
    st.caption(f"Test set: {model_metrics['Test rows']:,} rows · positive class rate: {model_metrics['Positive rate']:.1%} · Logistic Regression with imputation, scaling, and one-hot encoding.")
    col1, col2 = st.columns([.85, 1.15])
    with col1:
        cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Predicted · not positive", "Predicted · positive"], y=["Actual · not positive", "Actual · positive"], colorscale="Tealgrn", text=cm, texttemplate="%{text}", hovertemplate="%{z}<extra></extra>"))
        cm_fig.update_layout(title="Confusion matrix", xaxis_title="Prediction", yaxis_title="Actual")
        st.plotly_chart(plotly_defaults(cm_fig, 410), use_container_width=True)
    with col2:
        top = importance.head(18).sort_values("coefficient")
        fig = px.bar(top, x="coefficient", y="feature", orientation="h", color="coefficient", color_continuous_scale=["#d95f02", "#f4f6f8", "#0f766e"], color_continuous_midpoint=0)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_defaults(fig, 410), use_container_width=True)
        st.caption("Positive coefficients push the prediction toward the positive perceived-productivity class; negative coefficients push away from it.")
    st.subheader("Plain-language model read")
    st.markdown('<div class="insight">The model is good at separating the two survey-response classes in this sample. It is not a productivity score, an employee evaluation tool, or evidence that any listed factor causes the reported outcome.</div>', unsafe_allow_html=True)

with tabs[4]:
    st.header("Methodology & limitations")
    st.markdown("""
    **Dataset.** The dashboard uses the Stack Overflow Developer Survey 2025 public CSV. It is a large, voluntary developer survey with many missing values because questions are conditional and not every respondent answered every section.

    **Perceived productivity outcome.** The target is positive when the respondent selected the statement `AI agents have increased my productivity` in the AI-agent impact matrix. This is a perception reported by the respondent, not a measured productivity metric such as tickets resolved per hour.

    **Filters.** Filters affect the overview, EDA, and relationship tabs. They do not retrain the benchmark model so that the model result remains comparable from one filter context to another. Missing experience, AI-use, and agent-use values remain visible in the charts where possible.

    **Model.** The model uses Logistic Regression with median imputation for numeric fields, most-frequent imputation for categorical fields, standardization, and one-hot encoding. It uses a stratified 75/25 split with a fixed random seed. The target-impact matrix is excluded from the features to avoid direct target leakage.

    **Interpretation.** The survey supports descriptive and associational analysis. It cannot by itself establish that AI caused the perceived improvement. Selection bias, self-reporting, different job contexts, and differences in experience may all influence the patterns.
    """)
    st.download_button("Download filtered rows as CSV", filtered.to_csv(index=False).encode("utf-8"), "so2025_filtered.csv", "text/csv")
    st.caption("The downloaded file contains the currently visible rows and derived analysis fields.")
