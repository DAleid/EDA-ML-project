"""
AI Dependency and Academic Vulnerability - Interactive Dashboard
Survey of Bangladeshi university students (Mendeley Data, DOI 10.17632/6s6nvmpgbb.1)

Run with:  streamlit run app.py
Input   :  clean_data.csv (created by the EDA notebook; keep it in the same folder as this file)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Dependency in Higher Education", page_icon="🎓", layout="wide")

DATA_FILE = Path(__file__).parent / "clean_data.csv"

ORDERS = {
    "age_group": ["Under 20", "20-23", "24-26", "Above 26"],
    "cgpa_band": ["Below 2.50", "2.51-3.00", "3.01-3.50", "Above 3.50"],
    "daily_ai_use": ["Less than 1 hour", "1-3 hours", "More than 3 hours"],
    "reliance_level": ["Low", "Moderate", "High"],
}
ITEM_LABELS = {
    "verify_ai_info": "I verify AI information before using it",
    "study_without_ai": "I can study effectively without AI",
    "ask_ai_first": "My first instinct is to ask AI",
    "harder_to_recall": "Harder to recall AI-generated info",
    "speed_over_understanding": "I prioritise speed over understanding",
    "use_unmodified": "I use AI answers without modification",
    "less_confident_without_ai": "Less confident without AI",
    "critical_thinking_weakened": "Critical thinking has weakened",
    "submit_without_understanding": "I submit work I don't fully understand",
}
ITEMS = list(ITEM_LABELS)
LIKERT_LABELS = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
LIKERT_COLORS = ["#B2182B", "#EF8A62", "#D9D9D9", "#67A9CF", "#2166AC"]
MEASURES = {
    "reliance_score": "Reliance on AI score (1-5)",
    "cognitive_concern_score": "Cognitive concern score (1-5)",
    "verify_ai_info": "Verifies AI information (1-5)",
    "daily_ai_use_ord": "Daily AI use (1=<1h, 2=1-3h, 3=>3h)",
    "cgpa_ord": "CGPA band (1=<2.50 ... 4=>3.50)",
}
GROUPINGS = {"daily_ai_use": "Daily AI use", "academic_level": "Academic level", "age_group": "Age group",
             "gender": "Gender", "university_group": "University", "cgpa_band": "CGPA band"}


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading data...")
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col, order in ORDERS.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    for col in ["gender", "academic_level", "university", "university_group"]:
        df[col] = df[col].astype("category")
    df["daily_ai_use_ord"] = df["daily_ai_use"].cat.codes + 1
    df["cgpa_ord"] = df["cgpa_band"].cat.codes + 1
    df["high_cgpa"] = df["cgpa_band"] == "Above 3.50"
    return df


if not DATA_FILE.exists():
    st.error(f"`{DATA_FILE.name}` not found. Run the EDA notebook first and keep `clean_data.csv` next to `app.py`.")
    st.stop()

df = load_data(DATA_FILE)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🎓 AI & Students")
with st.sidebar.expander("📄 Dataset description", expanded=True):
    st.markdown(
        """
**AI Dependency and Academic Vulnerability in Higher Education**
Survey of Bangladeshi university students (Mendeley Data, DOI 10.17632/6s6nvmpgbb.1).

Students reported their gender, age group, university, academic level, **CGPA band** and **daily AI use**,
then rated nine statements (1 = strongly disagree, 5 = strongly agree) about how they use AI.

**Scores built in the analysis**
- *Reliance*: asking AI first, speed over understanding, unmodified use, submitting work not understood, and *not* being able to study without AI.
- *Cognitive concern*: harder recall, less confidence, weaker critical thinking.

⚠️ **Read with care:** self-reported answers from volunteers; one university supplies about 3 in 10 responses;
duplicate responses were removed in cleaning. Results show **associations**, not causes.
        """
    )

st.sidebar.header("Filters")
use_sel = st.sidebar.selectbox("Daily AI use (dropdown)", ["All"] + ORDERS["daily_ai_use"])
level_sel = st.sidebar.selectbox("Academic level (dropdown)", ["All"] + sorted(df["academic_level"].unique().tolist()))
ages = st.sidebar.multiselect("Age group (multiselect)", ORDERS["age_group"], default=ORDERS["age_group"])
unis = st.sidebar.multiselect("University (multiselect)", sorted(df["university_group"].unique().tolist()),
                              default=sorted(df["university_group"].unique().tolist()))
cgpas = st.sidebar.multiselect("CGPA band (multiselect)", ORDERS["cgpa_band"], default=ORDERS["cgpa_band"])
rel_range = st.sidebar.slider("Reliance score (slider)", 1.0, 5.0, (1.0, 5.0), step=0.2)
cog_range = st.sidebar.slider("Cognitive concern score (slider)", 1.0, 5.0, (1.0, 5.0), step=0.2)

mask = (df["age_group"].isin(ages) & df["university_group"].isin(unis) & df["cgpa_band"].isin(cgpas)
        & df["reliance_score"].between(*rel_range) & df["cognitive_concern_score"].between(*cog_range))
if use_sel != "All":
    mask &= df["daily_ai_use"] == use_sel
if level_sel != "All":
    mask &= df["academic_level"] == level_sel
f = df[mask]
st.sidebar.markdown(f"**{len(f):,}** of {len(df):,} respondents selected")

# ------------------------------------------------------------------ header
st.title("Data Insights Dashboard: AI use, dependence and grades")
st.caption("How do students who use AI more describe their dependence, study habits and confidence, and does it relate to CGPA? "
           "Everything here is **self-reported**; relationships are associations, not proof of cause and effect.")

if f.empty:
    st.warning("No respondents match the current filters. Widen the filters in the sidebar.")
    st.stop()
if len(f) < 30:
    st.warning(f"Only {len(f)} respondents selected: treat the results below as very uncertain.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Respondents", f"{len(f):,}")
heavy, heavy_all = (f["daily_ai_use"] == "More than 3 hours").mean() * 100, (df["daily_ai_use"] == "More than 3 hours").mean() * 100
k2.metric("Use AI > 3 h/day", f"{heavy:.0f}%", f"{heavy - heavy_all:+.0f} pts vs all")
k3.metric("Avg reliance (1-5)", f"{f['reliance_score'].mean():.2f}", f"{f['reliance_score'].mean() - df['reliance_score'].mean():+.2f} vs all",
          delta_color="inverse")
hr, hr_all = (f["reliance_level"] == "High").mean() * 100, (df["reliance_level"] == "High").mean() * 100
k4.metric("High reliance", f"{hr:.0f}%", f"{hr - hr_all:+.0f} pts vs all", delta_color="inverse")
hc, hc_all = f["high_cgpa"].mean() * 100, df["high_cgpa"].mean() * 100
k5.metric("CGPA above 3.50", f"{hc:.0f}%", f"{hc - hc_all:+.0f} pts vs all")

tabs = st.tabs(["📋 Data & statistics", "🤖 AI use vs reliance", "🎯 CGPA", "💬 The nine statements",
                "👥 Groups", "🔗 Correlations", "🔍 Explorer"])


def insight(text):
    st.info("💡 **Insight:** " + text)


def rho(a, b):
    s = f[[a, b]].dropna()
    if len(s) < 5 or s[a].nunique() < 2 or s[b].nunique() < 2:
        return np.nan
    return s[a].rank().corr(s[b].rank())


def strength(r):
    if np.isnan(r):
        return "not computable"
    a = abs(r)
    return "negligible" if a < .1 else "weak" if a < .3 else "moderate" if a < .5 else "strong"


def mean_ci(data, by, col):
    g = data.groupby(by, observed=True)[col].agg(["mean", "count", "sem"]).reset_index()
    g["ci"] = g["sem"].fillna(0) * 1.96
    return g


# ---------------------------------------------------------------- tab 1
with tabs[0]:
    st.subheader("Data preview")
    n_rows = st.slider("Rows to show", 5, 100, 10, key="rows")
    st.dataframe(f.drop(columns=["university_group", "daily_ai_use_ord", "cgpa_ord", "high_cgpa"]).head(n_rows), width="stretch")
    st.download_button("⬇️ Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"), "filtered_survey.csv", "text/csv")

    st.subheader("Summary statistics")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(f[["reliance_score", "cognitive_concern_score"] + ITEMS].rename(columns=ITEM_LABELS).describe().T.round(2), width="stretch")
    with c2:
        cat = st.selectbox("Category distribution", list(GROUPINGS), format_func=GROUPINGS.get, key="dist")
        order = ORDERS.get(cat, f[cat].value_counts().index.tolist())
        vc = f[cat].value_counts().reindex(order).fillna(0).reset_index()
        vc.columns = [cat, "respondents"]
        st.plotly_chart(px.bar(vc, x=cat, y="respondents", color_discrete_sequence=["#4C72B0"], labels={cat: GROUPINGS[cat]}),
                        width="stretch")
    top = f["university"].value_counts()
    insight(f"{len(f):,} respondents from {f['university'].nunique()} universities; the largest ({top.index[0]}) supplies "
            f"{top.iloc[0] / len(f) * 100:.0f}% of this selection. Mean reliance score is {f['reliance_score'].mean():.2f} on a 1-5 scale.")

# ---------------------------------------------------------------- tab 2
with tabs[1]:
    st.subheader("How do answers change with daily AI use?")
    measure = st.selectbox("Score", ["reliance_score", "cognitive_concern_score", "verify_ai_info"], format_func=MEASURES.get, key="m_use")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(f, x="daily_ai_use", y=measure, color="daily_ai_use", points="outliers",
                     category_orders={"daily_ai_use": ORDERS["daily_ai_use"]}, labels={"daily_ai_use": "Daily AI use", measure: MEASURES[measure]})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        g = mean_ci(f, "daily_ai_use", measure)
        fig = px.bar(g, x="daily_ai_use", y="mean", error_y="ci", hover_data={"count": True}, color_discrete_sequence=["#C44E52"],
                     category_orders={"daily_ai_use": ORDERS["daily_ai_use"]}, labels={"daily_ai_use": "Daily AI use", "mean": f"Mean {MEASURES[measure]}"})
        st.plotly_chart(fig, width="stretch")
    lv = (pd.crosstab(f["daily_ai_use"], f["reliance_level"], normalize="index") * 100).reset_index().melt(
        id_vars="daily_ai_use", var_name="reliance_level", value_name="percent")
    fig = px.bar(lv, x="daily_ai_use", y="percent", color="reliance_level", barmode="stack",
                 color_discrete_map={"Low": "#67A9CF", "Moderate": "#F2C14E", "High": "#C44E52"},
                 category_orders={"daily_ai_use": ORDERS["daily_ai_use"], "reliance_level": ORDERS["reliance_level"]},
                 labels={"daily_ai_use": "Daily AI use", "percent": "% of students"}, title="Reliance level by daily AI use")
    st.plotly_chart(fig, width="stretch")
    r = rho("daily_ai_use_ord", measure)
    g = mean_ci(f, "daily_ai_use", measure)
    insight(f"Spearman correlation between daily AI use and {MEASURES[measure].lower()}: {r:+.2f} ({strength(r)}). Group means: " +
            ", ".join(f"{k}: {v:.2f}" for k, v in zip(g["daily_ai_use"], g["mean"])) + ".")

# ---------------------------------------------------------------- tab 3
with tabs[2]:
    st.subheader("Is AI use or reliance related to CGPA?")
    c1, c2 = st.columns(2)
    with c1:
        ct = (pd.crosstab(f["daily_ai_use"], f["cgpa_band"], normalize="index") * 100).reset_index().melt(
            id_vars="daily_ai_use", var_name="cgpa_band", value_name="percent")
        fig = px.bar(ct, x="percent", y="daily_ai_use", color="cgpa_band", orientation="h", barmode="stack", text=ct["percent"].round(0).astype(int).astype(str) + "%",
                     category_orders={"daily_ai_use": ORDERS["daily_ai_use"], "cgpa_band": ORDERS["cgpa_band"]},
                     color_discrete_sequence=px.colors.sequential.YlGnBu, labels={"daily_ai_use": "Daily AI use", "percent": "% of students"},
                     title="CGPA band by daily AI use")
        st.plotly_chart(fig, width="stretch")
    with c2:
        g = f.groupby("cgpa_band", observed=True)[["reliance_score", "cognitive_concern_score", "verify_ai_info"]].mean().reset_index().melt(
            id_vars="cgpa_band", var_name="score", value_name="mean")
        g["score"] = g["score"].map(MEASURES)
        fig = px.bar(g, x="cgpa_band", y="mean", color="score", barmode="group", category_orders={"cgpa_band": ORDERS["cgpa_band"]},
                     labels={"cgpa_band": "CGPA band", "mean": "Mean score (1-5)"}, title="Mean scores by CGPA band")
        st.plotly_chart(fig, width="stretch")
    counts = f["cgpa_band"].value_counts().reindex(ORDERS["cgpa_band"]).fillna(0).astype(int)
    r1, r2 = rho("daily_ai_use_ord", "cgpa_ord"), rho("reliance_score", "cgpa_ord")
    share = f.groupby("daily_ai_use", observed=True)["high_cgpa"].mean() * 100
    insight(f"Correlation of daily AI use with CGPA band: {r1:+.2f} ({strength(r1)}); of reliance with CGPA band: {r2:+.2f} ({strength(r2)}). "
            f"Share with CGPA above 3.50: " + ", ".join(f"{k}: {v:.0f}%" for k, v in share.items()) + ". "
            f"Note: CGPA is a self-reported band with few respondents in the lowest band (n = {counts.iloc[0]}).")

# ---------------------------------------------------------------- tab 4
with tabs[3]:
    st.subheader("Responses to the nine statements")
    rows = []
    for c in ITEMS:
        p = f[c].value_counts(normalize=True).reindex(range(1, 6)).fillna(0) * 100
        for i, lab in enumerate(LIKERT_LABELS, 1):
            rows.append((ITEM_LABELS[c], lab, p.loc[i]))
    lk = pd.DataFrame(rows, columns=["statement", "response", "percent"])
    order = lk[lk["response"].isin(["Agree", "Strongly Agree"])].groupby("statement")["percent"].sum().sort_values().index.tolist()
    fig = px.bar(lk, x="percent", y="statement", color="response", orientation="h", barmode="stack",
                 category_orders={"response": LIKERT_LABELS, "statement": order}, color_discrete_sequence=LIKERT_COLORS,
                 labels={"percent": "% of respondents", "statement": ""})
    fig.update_layout(height=480, legend_orientation="h", legend_y=-0.2)
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Mean agreement (1-5) by daily AI use**")
    hm = f.groupby("daily_ai_use", observed=True)[ITEMS].mean().T
    hm.index = [ITEM_LABELS[i] for i in hm.index]
    fig = px.imshow(hm.round(2), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=1, zmax=5, labels={"color": "mean"})
    st.plotly_chart(fig, width="stretch")
    agree = {ITEM_LABELS[c]: (f[c] >= 4).mean() * 100 for c in ITEMS}
    top_item = max(agree, key=agree.get)
    low_item = min(agree, key=agree.get)
    insight(f"Highest agreement: '{top_item}' ({agree[top_item]:.0f}% agree or strongly agree). Lowest: '{low_item}' ({agree[low_item]:.0f}%). "
            f"Items are phrased from the student's own perspective, so they measure perceptions, not tested skills.")

# ---------------------------------------------------------------- tab 5
with tabs[4]:
    st.subheader("Compare groups")
    c1, c2 = st.columns(2)
    grp = c1.selectbox("Group by", list(GROUPINGS), format_func=GROUPINGS.get, key="grp")
    meas = c2.selectbox("Measure", list(MEASURES), format_func=MEASURES.get, key="meas")
    g = mean_ci(f, grp, meas)
    fig = px.bar(g, x=grp, y="mean", error_y="ci", hover_data={"count": True}, color_discrete_sequence=["#4C72B0"],
                 category_orders={grp: ORDERS.get(grp, sorted(f[grp].astype(str).unique()))}, labels={grp: GROUPINGS[grp], "mean": MEASURES[meas]})
    st.plotly_chart(fig, width="stretch")
    st.dataframe(g.rename(columns={"mean": "mean", "count": "n", "ci": "95% CI (±)"}).drop(columns="sem").round(2), width="stretch")
    big = g[g["count"] >= 10]
    if len(big) >= 2:
        hi, lo = big.loc[big["mean"].idxmax()], big.loc[big["mean"].idxmin()]
        insight(f"Among groups with at least 10 respondents, '{hi[grp]}' has the highest mean ({hi['mean']:.2f}) and '{lo[grp]}' the lowest "
                f"({lo['mean']:.2f}). If the error bars overlap, the difference may not be meaningful. Groups with few respondents are unreliable.")

# ---------------------------------------------------------------- tab 6
with tabs[5]:
    st.subheader("Spearman correlations")
    cols = ITEMS + ["daily_ai_use_ord", "cgpa_ord"]
    cm = f[cols].rank().corr()
    names = {**ITEM_LABELS, "daily_ai_use_ord": "Daily AI use (1-3)", "cgpa_ord": "CGPA band (1-4)"}
    cm.index = cm.columns = [names[c] for c in cols]
    fig = px.imshow(cm.round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto", labels={"color": "rho"})
    fig.update_layout(height=680)
    st.plotly_chart(fig, width="stretch")
    cg = cm["CGPA band (1-4)"].drop("CGPA band (1-4)")
    strongest = cg.abs().idxmax()
    insight(f"The nine statements are strongly linked to each other and to daily AI use, but the largest correlation with CGPA band is only "
            f"{cg[strongest]:+.2f} ('{strongest}'). Correlation does not imply causation.")

# ---------------------------------------------------------------- tab 7
with tabs[6]:
    st.subheader("Build your own chart")
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Chart type", ["Box", "Bar (mean)", "Histogram", "Heatmap (% of row)"])
    x = c2.selectbox("Group (X)", list(GROUPINGS), format_func=GROUPINGS.get, key="ex_x")
    y = c3.selectbox("Score (Y)", list(MEASURES), format_func=MEASURES.get, key="ex_y")
    color = c4.selectbox("Color by", ["(none)"] + list(GROUPINGS), format_func=lambda c: GROUPINGS.get(c, c), key="ex_c")
    col = None if color == "(none)" else color
    if kind == "Box":
        fig = px.box(f, x=x, y=y, color=col, category_orders={x: ORDERS.get(x, sorted(f[x].astype(str).unique()))})
    elif kind == "Bar (mean)":
        g = f.groupby([x] + ([col] if col and col != x else []), observed=True)[y].mean().reset_index()
        fig = px.bar(g, x=x, y=y, color=col if col and col != x else None, barmode="group", category_orders={x: ORDERS.get(x, sorted(f[x].astype(str).unique()))})
    elif kind == "Histogram":
        fig = px.histogram(f, x=y, color=col, nbins=20, barmode="overlay", opacity=0.7)
    else:
        other = col if col and col != x else ("cgpa_band" if x != "cgpa_band" else "daily_ai_use")
        ct = pd.crosstab(f[x], f[other], normalize="index") * 100
        fig = px.imshow(ct.round(0), text_auto=True, aspect="auto", color_continuous_scale="Blues", labels={"color": "% of row", "x": GROUPINGS[other], "y": GROUPINGS[x]})
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ overall insights
st.divider()
st.subheader("Key insights (update with your filters)")
a, b, c_, d = rho("daily_ai_use_ord", "reliance_score"), rho("daily_ai_use_ord", "verify_ai_info"), rho("daily_ai_use_ord", "cognitive_concern_score"), rho("daily_ai_use_ord", "cgpa_ord")
st.markdown(
    f"""
- **Daily AI use and reliance:** Spearman rho = **{a:+.2f}** ({strength(a)}).
- **Daily AI use and verifying AI output:** rho = **{b:+.2f}** ({strength(b)}).
- **Daily AI use and feeling less capable (memory, confidence, critical thinking):** rho = **{c_:+.2f}** ({strength(c_)}).
- **Daily AI use and CGPA band:** rho = **{d:+.2f}** ({strength(d)}).
- **Caution:** self-reported survey of volunteers (one university dominates). These are associations, not causes.
"""
)
st.caption("Final Project - Unit 3 | Built with Streamlit, pandas and Plotly | Data: Mendeley Data, DOI 10.17632/6s6nvmpgbb.1")
