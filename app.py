"""
Impact of AI on Students - Interactive Dashboard
Run with:  streamlit run app.py
Input   :  clean_data.csv (produced by the EDA notebook, same folder as this file)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Impact on Students", page_icon="🎓", layout="wide")

DATA_FILE = Path(__file__).parent / "clean_data.csv"

ORDERS = {
    "year_of_study": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
    "prompt_engineering_skill": ["Beginner", "Intermediate", "Advanced"],
    "burnout_risk_level": ["Low", "Medium", "High"],
    "ai_usage_group": ["0-2h", "2-5h", "5-10h", "10-20h", "20-40h"],
}
LABELS = {
    "gpa_change": "GPA change (post - pre)",
    "skill_retention_score": "Skill retention score",
    "anxiety_level_during_exams": "Exam anxiety (1-10)",
    "burnout_score": "Burnout score (0=Low, 2=High)",
    "weekly_genai_hours": "Weekly GenAI hours",
    "traditional_study_hours": "Traditional study hours",
    "perceived_ai_dependency": "Perceived AI dependency (1-10)",
    "pre_semester_gpa": "Pre-semester GPA",
    "post_semester_gpa": "Post-semester GPA",
    "tool_diversity": "Number of AI tools",
    "ai_share": "AI share of study time",
    "total_study_hours": "Total study hours",
}
OUTCOMES = ["gpa_change", "skill_retention_score", "anxiety_level_during_exams", "burnout_score"]
GROUPINGS = ["major_category", "year_of_study", "primary_use_case", "prompt_engineering_skill",
             "institutional_policy", "paid_subscription", "ai_usage_group"]
NUMERIC = ["pre_semester_gpa", "post_semester_gpa", "gpa_change", "weekly_genai_hours", "traditional_study_hours",
           "tool_diversity", "perceived_ai_dependency", "anxiety_level_during_exams", "skill_retention_score",
           "burnout_score", "ai_share", "total_study_hours"]
BURNOUT_COLORS = {"Low": "#55A868", "Medium": "#F2A541", "High": "#C44E52"}


def label(c):
    return LABELS.get(c, c.replace("_", " ").capitalize())


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading data...")
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col, order in ORDERS.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    for col in ["major_category", "primary_use_case", "institutional_policy"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    if "paid_subscription" in df.columns:
        df["paid_subscription"] = df["paid_subscription"].astype(str).map(
            {"True": "Paid", "False": "Free", "1": "Paid", "0": "Free"}).astype("category")
    if "gpa_change" not in df.columns:
        df["gpa_change"] = df["post_semester_gpa"] - df["pre_semester_gpa"]
    if "burnout_score" not in df.columns:
        df["burnout_score"] = df["burnout_risk_level"].cat.codes
    if "total_study_hours" not in df.columns:
        df["total_study_hours"] = df["weekly_genai_hours"] + df["traditional_study_hours"]
    if "ai_share" not in df.columns:
        df["ai_share"] = (df["weekly_genai_hours"] / df["total_study_hours"].replace(0, np.nan)).fillna(0)
    if "ai_usage_group" not in df.columns:
        df["ai_usage_group"] = pd.cut(df["weekly_genai_hours"], bins=[-0.01, 2, 5, 10, 20, 40.01],
                                      labels=ORDERS["ai_usage_group"])
    return df


if not DATA_FILE.exists():
    st.error(f"`{DATA_FILE.name}` not found. Run the EDA notebook first and keep `clean_data.csv` next to `app.py`.")
    st.stop()

df = load_data(DATA_FILE)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🎓 AI Impact on Students")
with st.sidebar.expander("📄 Dataset description", expanded=True):
    st.markdown(
        """
**Impact of AI on Students** (Kaggle, `laveshjadon`)

Student-level records on generative-AI usage, traditional study time, GPA before and after a semester,
skill retention, exam anxiety and burnout risk, plus the institution's AI policy.

⚠️ *The dataset page does not document how the data was collected and the data looks very clean and balanced.
Treat it as possibly **simulated**: patterns describe this dataset, not necessarily real students.*
        """
    )

st.sidebar.header("Filters")

major = st.sidebar.selectbox("Major (dropdown)", ["All"] + sorted(df["major_category"].dropna().unique().tolist()))
use_case = st.sidebar.selectbox("Primary AI use case (dropdown)",
                                ["All"] + sorted(df["primary_use_case"].dropna().unique().tolist()))
paid = st.sidebar.selectbox("AI subscription (dropdown)", ["All", "Paid", "Free"])

years = st.sidebar.multiselect("Year of study (multiselect)", ORDERS["year_of_study"], default=ORDERS["year_of_study"])
policies = st.sidebar.multiselect("Institutional policy (multiselect)",
                                  sorted(df["institutional_policy"].dropna().unique().tolist()),
                                  default=sorted(df["institutional_policy"].dropna().unique().tolist()))
skills = st.sidebar.multiselect("Prompt skill (multiselect)", ORDERS["prompt_engineering_skill"],
                                default=ORDERS["prompt_engineering_skill"])

ai_min, ai_max = float(df["weekly_genai_hours"].min()), float(df["weekly_genai_hours"].max())
ai_range = st.sidebar.slider("Weekly GenAI hours (slider)", ai_min, ai_max, (ai_min, ai_max), step=0.5)
tr_min, tr_max = float(df["traditional_study_hours"].min()), float(df["traditional_study_hours"].max())
tr_range = st.sidebar.slider("Traditional study hours (slider)", tr_min, tr_max, (tr_min, tr_max), step=0.5)
g_min, g_max = float(df["pre_semester_gpa"].min()), float(df["pre_semester_gpa"].max())
gpa_range = st.sidebar.slider("Pre-semester GPA (slider)", round(g_min, 2), round(g_max, 2),
                              (round(g_min, 2), round(g_max, 2)), step=0.05)

mask = (
    df["year_of_study"].isin(years)
    & df["institutional_policy"].isin(policies)
    & df["prompt_engineering_skill"].isin(skills)
    & df["weekly_genai_hours"].between(*ai_range)
    & df["traditional_study_hours"].between(*tr_range)
    & df["pre_semester_gpa"].between(*gpa_range)
)
if major != "All":
    mask &= df["major_category"] == major
if use_case != "All":
    mask &= df["primary_use_case"] == use_case
if paid != "All":
    mask &= df["paid_subscription"] == paid
f = df[mask]

st.sidebar.markdown(f"**{len(f):,}** of {len(df):,} students selected")

# ------------------------------------------------------------------ header
st.title("Data Insights Dashboard: Is AI a tutor or a shortcut?")
st.caption("Explore how generative-AI use relates to grades, skill retention, anxiety and burnout. "
           "Relationships shown are **associations**, not proof of cause and effect.")

if f.empty:
    st.warning("No students match the current filters. Widen the filters in the sidebar.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Students", f"{len(f):,}")
k2.metric("Avg weekly AI hours", f"{f['weekly_genai_hours'].mean():.1f}",
          f"{f['weekly_genai_hours'].mean() - df['weekly_genai_hours'].mean():+.1f} vs all")
k3.metric("Avg GPA change", f"{f['gpa_change'].mean():+.2f}",
          f"{f['gpa_change'].mean() - df['gpa_change'].mean():+.2f} vs all")
k4.metric("Avg skill retention", f"{f['skill_retention_score'].mean():.1f}",
          f"{f['skill_retention_score'].mean() - df['skill_retention_score'].mean():+.1f} vs all")
hb, hb_all = (f["burnout_risk_level"] == "High").mean() * 100, (df["burnout_risk_level"] == "High").mean() * 100
k5.metric("High burnout risk", f"{hb:.0f}%", f"{hb - hb_all:+.0f} pts vs all", delta_color="inverse")

tabs = st.tabs(["📋 Data & statistics", "🤖 AI usage vs outcomes", "🧠 Burnout & well-being",
                "🏛️ Groups & policy", "🔗 Correlations", "🔍 Explorer"])


def insight(text):
    st.info("💡 **Insight:** " + text)


def safe_corr(a, b, method="spearman"):
    s = f[[a, b]].dropna()
    if len(s) < 3 or s[a].nunique() < 2 or s[b].nunique() < 2:
        return np.nan
    return s[a].rank().corr(s[b].rank()) if method == "spearman" else s[a].corr(s[b])


# ---------------------------------------------------------------- tab 1
with tabs[0]:
    st.subheader("Data preview")
    n_rows = st.slider("Rows to show", 5, 100, 10, key="rows")
    st.dataframe(f.head(n_rows), width="stretch")
    st.download_button("⬇️ Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"),
                       "filtered_students.csv", "text/csv")

    st.subheader("Summary statistics")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(f[[c for c in NUMERIC if c in f.columns and c != "burnout_score"]].describe().T.round(2),
                     width="stretch")
    with c2:
        cat_pick = st.selectbox("Category distribution", ["major_category", "year_of_study", "primary_use_case",
                                                          "prompt_engineering_skill", "institutional_policy",
                                                          "burnout_risk_level"], format_func=label)
        vc = f[cat_pick].value_counts().reindex(
            ORDERS.get(cat_pick, f[cat_pick].cat.categories.tolist())).reset_index()
        vc.columns = [cat_pick, "students"]
        st.plotly_chart(px.bar(vc, x=cat_pick, y="students", color_discrete_sequence=["#4C72B0"],
                               labels={cat_pick: label(cat_pick)}), width="stretch")
    insight(f"The current selection contains {len(f):,} students. "
            f"Average post-semester GPA is {f['post_semester_gpa'].mean():.2f} "
            f"(pre: {f['pre_semester_gpa'].mean():.2f}), and {f['gpa_improved'].mean() * 100:.0f}% of students improved."
            if "gpa_improved" in f.columns else
            f"The current selection contains {len(f):,} students.")

# ---------------------------------------------------------------- tab 2
with tabs[1]:
    st.subheader("How do outcomes change with weekly AI hours?")
    outcome = st.selectbox("Outcome", OUTCOMES[:3], format_func=label, key="out_ai")
    c1, c2 = st.columns(2)
    by_group = f.groupby("ai_usage_group", observed=True)[outcome].agg(["mean", "count", "sem"]).reset_index()
    with c1:
        fig = px.bar(by_group, x="ai_usage_group", y="mean", error_y=by_group["sem"] * 1.96,
                     color_discrete_sequence=["#C44E52"], hover_data={"count": True},
                     labels={"ai_usage_group": "Weekly AI hours", "mean": f"Mean {label(outcome)}"})
        st.plotly_chart(fig, width="stretch")
    with c2:
        smp = f.sample(min(len(f), 4000), random_state=42)
        fig = px.scatter(smp, x="weekly_genai_hours", y=outcome, opacity=0.35, color="prompt_engineering_skill",
                         category_orders={"prompt_engineering_skill": ORDERS["prompt_engineering_skill"]},
                         labels={"weekly_genai_hours": label("weekly_genai_hours"), outcome: label(outcome),
                                 "prompt_engineering_skill": "Prompt skill"})
        st.plotly_chart(fig, width="stretch")
    if len(by_group) >= 2:
        lo, hi = by_group.iloc[0], by_group.iloc[-1]
        rho = safe_corr("weekly_genai_hours", outcome)
        insight(f"Mean {label(outcome).lower()} is {lo['mean']:.2f} in the '{lo['ai_usage_group']}' group and "
                f"{hi['mean']:.2f} in the '{hi['ai_usage_group']}' group. Spearman correlation with AI hours: "
                f"{rho:+.2f}. (Association only: students who struggle may also use AI more.)")

# ---------------------------------------------------------------- tab 3
with tabs[2]:
    st.subheader("Burnout risk and well-being")
    c1, c2 = st.columns(2)
    with c1:
        ct = (pd.crosstab(f["ai_usage_group"], f["burnout_risk_level"], normalize="index") * 100).reset_index()
        ct = ct.melt(id_vars="ai_usage_group", var_name="burnout_risk_level", value_name="percent")
        fig = px.bar(ct, x="ai_usage_group", y="percent", color="burnout_risk_level", color_discrete_map=BURNOUT_COLORS,
                     category_orders={"burnout_risk_level": ORDERS["burnout_risk_level"],
                                      "ai_usage_group": ORDERS["ai_usage_group"]},
                     labels={"ai_usage_group": "Weekly AI hours", "percent": "% of students",
                             "burnout_risk_level": "Burnout"}, title="Burnout mix by AI usage")
        st.plotly_chart(fig, width="stretch")
    with c2:
        hm = (pd.crosstab(f["perceived_ai_dependency"], f["burnout_risk_level"], normalize="index") * 100)
        fig = px.imshow(hm.round(0), text_auto=True, aspect="auto", color_continuous_scale="Reds",
                        labels={"x": "Burnout", "y": "Perceived AI dependency", "color": "%"},
                        title="Burnout level (%) by perceived AI dependency")
        st.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(f, x="burnout_risk_level", y="anxiety_level_during_exams", color="burnout_risk_level",
                     color_discrete_map=BURNOUT_COLORS, category_orders={"burnout_risk_level": ORDERS["burnout_risk_level"]},
                     title="Exam anxiety by burnout level", labels={"anxiety_level_during_exams": "Exam anxiety"})
        st.plotly_chart(fig, width="stretch")
    with c4:
        fig = px.box(f, x="burnout_risk_level", y="weekly_genai_hours", color="burnout_risk_level",
                     color_discrete_map=BURNOUT_COLORS, category_orders={"burnout_risk_level": ORDERS["burnout_risk_level"]},
                     title="Weekly AI hours by burnout level", labels={"weekly_genai_hours": "Weekly AI hours"})
        st.plotly_chart(fig, width="stretch")
    hi_share = (f["burnout_risk_level"] == "High").groupby(f["ai_usage_group"], observed=True).mean() * 100
    if len(hi_share) >= 2:
        insight("Share of high burnout risk by AI usage: " +
                ", ".join(f"{k}: {v:.0f}%" for k, v in hi_share.items()) +
                f". Correlation between AI dependency and anxiety: {safe_corr('perceived_ai_dependency', 'anxiety_level_during_exams'):+.2f}.")

# ---------------------------------------------------------------- tab 4
with tabs[3]:
    st.subheader("Compare groups")
    c1, c2 = st.columns(2)
    grp = c1.selectbox("Group by", GROUPINGS, format_func=label, key="grp")
    out2 = c2.selectbox("Outcome", OUTCOMES[:3], format_func=label, key="out_grp")
    g = f.groupby(grp, observed=True)[out2].agg(["mean", "count", "sem"]).reset_index()
    fig = px.bar(g, x=grp, y="mean", error_y=g["sem"] * 1.96, color_discrete_sequence=["#4C72B0"],
                 hover_data={"count": True}, labels={grp: label(grp), "mean": f"Mean {label(out2)}"})
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Two-way view:** mean outcome by AI usage and another variable")
    other = st.selectbox("Second variable", [x for x in GROUPINGS if x != "ai_usage_group"], format_func=label, key="oth")
    pv = f.pivot_table(index="ai_usage_group", columns=other, values=out2, aggfunc="mean", observed=True)
    fig = px.imshow(pv.round(2), text_auto=True, aspect="auto", color_continuous_scale="RdYlGn",
                    labels={"color": label(out2), "y": "Weekly AI hours", "x": label(other)})
    st.plotly_chart(fig, width="stretch")
    if len(g) >= 2:
        best, worst = g.loc[g["mean"].idxmax()], g.loc[g["mean"].idxmin()]
        insight(f"For {label(out2).lower()}, '{best[grp]}' scores highest ({best['mean']:.2f}) and "
                f"'{worst[grp]}' lowest ({worst['mean']:.2f}), a gap of {best['mean'] - worst['mean']:.2f}. "
                f"Check the error bars: small gaps may not be meaningful.")

# ---------------------------------------------------------------- tab 5
with tabs[4]:
    st.subheader("Correlation heatmap")
    method = st.radio("Method", ["spearman", "pearson"], horizontal=True)
    cols = [c for c in NUMERIC if c in f.columns]
    cm = f[cols].rank().corr() if method == "spearman" else f[cols].corr()
    fig = px.imshow(cm.round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
                    labels={"color": "r"})
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")
    pairs = cm.where(np.triu(np.ones(cm.shape, dtype=bool), 1)).stack().reset_index()
    pairs.columns = ["var_1", "var_2", "r"]
    # skip trivially related pairs built from the same variables
    trivial = {frozenset(p) for p in [("pre_semester_gpa", "gpa_change"), ("post_semester_gpa", "gpa_change"),
               ("weekly_genai_hours", "ai_share"), ("weekly_genai_hours", "total_study_hours"),
               ("traditional_study_hours", "total_study_hours"), ("traditional_study_hours", "ai_share"),
               ("ai_share", "total_study_hours"), ("pre_semester_gpa", "post_semester_gpa")]}
    pairs = pairs[~pairs.apply(lambda r: frozenset((r.var_1, r.var_2)) in trivial, axis=1)]
    top = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).head(3)
    insight("Strongest non-trivial relationships: " +
            "; ".join(f"{label(a)} and {label(b)} ({r:+.2f})" for a, b, r in top.itertuples(index=False)) +
            ". Correlation does not imply causation.")

# ---------------------------------------------------------------- tab 6
with tabs[5]:
    st.subheader("Build your own chart")
    num_cols = [c for c in NUMERIC if c in f.columns]
    cat_cols = ["(none)"] + GROUPINGS + ["burnout_risk_level"]
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Chart type", ["Scatter", "Histogram", "Box"])
    x = c2.selectbox("X", num_cols, index=num_cols.index("weekly_genai_hours"), format_func=label)
    y = c3.selectbox("Y (scatter/box)", num_cols, index=num_cols.index("skill_retention_score"), format_func=label)
    color = c4.selectbox("Color by", cat_cols, format_func=lambda c: label(c) if c != "(none)" else c)
    col = None if color == "(none)" else color
    if kind == "Scatter":
        fig = px.scatter(f.sample(min(len(f), 4000), random_state=1), x=x, y=y, color=col, opacity=0.4,
                         labels={x: label(x), y: label(y)})
    elif kind == "Histogram":
        fig = px.histogram(f, x=x, color=col, nbins=40, barmode="overlay", opacity=0.7, labels={x: label(x)})
    else:
        fig = px.box(f, x=col, y=y, color=col, labels={y: label(y)}) if col else px.box(f, y=y, labels={y: label(y)})
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ overall insights
st.divider()
st.subheader("Key insights (update with your filters)")
r1 = safe_corr("weekly_genai_hours", "skill_retention_score")
r2 = safe_corr("weekly_genai_hours", "gpa_change")
r3 = safe_corr("perceived_ai_dependency", "burnout_score")
st.markdown(
    f"""
- **Skill retention:** Spearman correlation with weekly AI hours is **{r1:+.2f}**.
- **GPA change:** correlation with weekly AI hours is **{r2:+.2f}**.
- **Burnout:** correlation between perceived AI dependency and burnout level is **{r3:+.2f}**.
- **Caution:** these are associations in observational (and possibly simulated) data. They do not show that AI *causes* these outcomes.
"""
)
st.caption("Final Project - Unit 3 | Built with Streamlit, pandas and Plotly.")
