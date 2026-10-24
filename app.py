"""
AI Dependency and Academic Vulnerability - Interactive Dashboard (redesigned)
Survey of Bangladeshi university students (Mendeley Data, DOI 10.17632/6s6nvmpgbb.1)

Run with:  streamlit run app.py
Input   :  clean_data.csv (created by the EDA notebook; keep it next to this file)
Theme   :  put .streamlit/config.toml (provided) next to this file for dark widgets
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="AI Dependency in Higher Education", page_icon="🧠", layout="wide")

DATA_FILE = Path(__file__).parent / "clean_data.csv"
RAW_N = 1104  # rows in the raw Mendeley file (before removing duplicates)

# ------------------------------------------------------------------ design tokens
INK, PANEL, PANEL2, LINE = "#0A0F1F", "#111A33", "#17223F", "#26345C"
TEXT, MUTED, GRID = "#E8EDFB", "#93A0C4", "rgba(147,160,196,.14)"
CYAN, VIOLET, CORAL, AMBER = "#3DD6F5", "#8B7CFF", "#FF6B81", "#FFC857"
FONT = "Manrope, 'Segoe UI', sans-serif"

pio.templates["ai"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=TEXT, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[CYAN, VIOLET, CORAL, AMBER, "#5EEAD4", "#F0ABFC"],
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, ticks="", automargin=True, title=dict(font=dict(size=12, color=MUTED))),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, ticks="", automargin=True, title=dict(font=dict(size=12, color=MUTED))),
    hoverlabel=dict(bgcolor=PANEL2, bordercolor=LINE, font=dict(color=TEXT, family=FONT)),
    legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, font=dict(size=12, color=MUTED)),
    margin=dict(l=8, r=8, t=16, b=8),
))
pio.templates.default = "ai"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');
:root{--ink:#0A0F1F;--panel:#111A33;--panel2:#17223F;--line:#26345C;--text:#E8EDFB;--muted:#93A0C4;
--cyan:#3DD6F5;--violet:#8B7CFF;--coral:#FF6B81;--amber:#FFC857;}
html,body,.stApp,[data-testid="stAppViewContainer"]{background-color:var(--ink)!important;color:var(--text);font-family:Manrope,sans-serif;}
.stApp{background-image:radial-gradient(rgba(139,124,255,.15) 1px,transparent 1px),radial-gradient(900px 420px at 88% -6%,rgba(61,214,245,.10),transparent 70%);
background-size:28px 28px,100% 100%;background-attachment:fixed;}
[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1300px;}
h1,h2,h3,h4{font-family:Sora,sans-serif!important;color:var(--text)!important;letter-spacing:-.01em;}
[data-testid="stSidebar"]{background:#0C1330;border-right:1px solid var(--line);}
[data-testid="stSidebar"] *{font-family:Manrope,sans-serif;}

.hero{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:28px;align-items:center;
border:1px solid var(--line);border-radius:18px;padding:30px 34px;margin-bottom:18px;
background:linear-gradient(135deg,rgba(23,34,63,.9),rgba(17,26,51,.75));}
.hero h1{font-size:2.35rem;line-height:1.12;margin:0 0 12px 0;font-weight:700;}
.hero p{color:var(--muted);font-size:1.02rem;line-height:1.6;margin:0 0 16px 0;max-width:60ch;}
.tags span{display:inline-block;font-size:.78rem;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:4px 12px;margin:0 8px 6px 0;}
.hero svg{width:100%;height:auto;}
.maplegend{font-size:.78rem;color:var(--muted);text-align:center;margin-top:2px;}
@media(max-width:900px){.hero{grid-template-columns:1fr;padding:22px;}.hero h1{font-size:1.8rem;}}

.kpi{border:1px solid var(--line);border-radius:14px;padding:16px 18px;background:var(--panel);height:100%;position:relative;}
.kpi:before{content:"";position:absolute;left:0;top:16px;bottom:16px;width:3px;border-radius:3px;background:var(--accent);}
.kpi .l{color:var(--muted);font-size:.82rem;margin-bottom:6px;}
.kpi .v{font-family:Sora,sans-serif;font-size:1.9rem;font-weight:600;line-height:1.1;}
.kpi .d{color:var(--muted);font-size:.78rem;margin-top:6px;}

[class*="st-key-panel"]{background:var(--panel)!important;border:1px solid var(--line)!important;border-radius:14px!important;padding:6px 8px;}
.q{font-family:Sora,sans-serif;font-size:1.08rem;font-weight:600;margin:6px 0 2px 0;}
.qs{color:var(--muted);font-size:.86rem;margin-bottom:6px;line-height:1.5;}
.answer{border-left:3px solid var(--cyan);background:rgba(61,214,245,.07);border-radius:0 10px 10px 0;padding:12px 16px;margin:6px 0 4px 0;
font-size:.93rem;line-height:1.6;}
.answer b.t{color:var(--cyan);font-family:Sora,sans-serif;font-weight:600;margin-right:6px;}
.answer.warn{border-left-color:var(--amber);background:rgba(255,200,87,.07);}
.answer.warn b.t{color:var(--amber);}
.note{color:var(--muted);font-size:.8rem;}

.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{color:var(--muted);font-weight:600;padding:10px 16px;}
.stTabs [data-baseweb="tab"]:hover,.stTabs [data-baseweb="tab"]:hover *{color:var(--cyan)!important;}
.stTabs [aria-selected="true"],.stTabs [aria-selected="true"] *{color:var(--cyan)!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-highlight"] *{background:var(--cyan)!important;background-color:var(--cyan)!important;}
[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"]{background:var(--cyan)!important;}
.stTabs [data-baseweb="tab-border"]{display:none;}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;}
.stDownloadButton button{background:var(--panel2);border:1px solid var(--line);color:var(--text);}
.stDownloadButton button:hover{border-color:var(--cyan);color:var(--cyan);}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:var(--panel);}
"""
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

# ------------------------------------------------------------------ constants
ORDERS = {
    "age_group": ["Under 20", "20-23", "24-26", "Above 26"],
    "cgpa_band": ["Below 2.50", "2.51-3.00", "3.01-3.50", "Above 3.50"],
    "daily_ai_use": ["Less than 1 hour", "1-3 hours", "More than 3 hours"],
    "reliance_level": ["Low", "Moderate", "High"],
}
USE_COLORS = {"Less than 1 hour": CYAN, "1-3 hours": VIOLET, "More than 3 hours": CORAL}
LEVEL_COLORS = {"Low": CYAN, "Moderate": AMBER, "High": CORAL}
CGPA_COLORS = ["#34407A", "#5663C7", "#38A8E8", "#9CF0FF"]
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
PROTECTIVE = {"verify_ai_info", "study_without_ai"}
COGNITIVE = ["harder_to_recall", "less_confident_without_ai", "critical_thinking_weakened"]
LIKERT_LABELS = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
LIKERT_COLORS = ["#E11D48", "#FB7185", "#4A5680", "#67E8F9", "#06B6D4"]
MEASURES = {
    "reliance_score": "Reliance on AI (1-5)",
    "cognitive_concern_score": "Cognitive concern (1-5)",
    "verify_ai_info": "Verifies AI information (1-5)",
    "daily_ai_use_ord": "Daily AI use (1=<1h, 3=>3h)",
    "cgpa_ord": "CGPA band (1=<2.50, 4=>3.50)",
}
GROUPINGS = {"daily_ai_use": "Daily AI use", "academic_level": "Academic level", "age_group": "Age group",
             "gender": "Gender", "university_group": "University", "cgpa_band": "CGPA band"}


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner="Loading data...")
def load_data(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    for col, order in ORDERS.items():
        if col in d.columns:
            d[col] = pd.Categorical(d[col], categories=order, ordered=True)
    for col in ["gender", "academic_level", "university", "university_group"]:
        d[col] = d[col].astype("category")
    d["daily_ai_use_ord"] = d["daily_ai_use"].cat.codes + 1
    d["cgpa_ord"] = d["cgpa_band"].cat.codes + 1
    d["high_cgpa"] = d["cgpa_band"] == "Above 3.50"
    d["high_reliance"] = d["reliance_level"] == "High"
    return d


if not DATA_FILE.exists():
    st.error(f"`{DATA_FILE.name}` not found. Run the EDA notebook first and keep `clean_data.csv` next to `app.py`.")
    st.stop()
df = load_data(DATA_FILE)


# ------------------------------------------------------------------ helpers
def html(s: str):
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def panel(key: str, question: str, sub: str = ""):
    box = st.container(border=True, key="panel_" + key)
    with box:
        html(f'<div class="q">{question}</div>' + (f'<div class="qs">{sub}</div>' if sub else ""))
    return box


def answer(text: str, warn: bool = False):
    html(f'<div class="answer{" warn" if warn else ""}"><b class="t">{"Read with care" if warn else "What the data says"}</b>{text}</div>')


def show(fig, height=340):
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p * 100, (c - h) * 100, (c + h) * 100


def mean_ci(data, by, col):
    g = data.groupby(by, observed=True)[col].agg(["mean", "count", "sem"]).reset_index()
    g["ci"] = g["sem"].fillna(0) * 1.96
    return g


def strength(r):
    if np.isnan(r):
        return "not computable"
    a = abs(r)
    return "negligible" if a < .1 else "weak" if a < .3 else "moderate" if a < .5 else "strong"


def link(r):
    if np.isnan(r):
        return "cannot be computed for this selection"
    return "no meaningful link" if abs(r) < .1 else f"a {strength(r)} {'positive' if r > 0 else 'negative'} link"


def rho_ci(a, b):
    s = f[[a, b]].dropna()
    n = len(s)
    if n < 10 or s[a].nunique() < 2 or s[b].nunique() < 2:
        return np.nan, np.nan, np.nan
    r = s[a].rank().corr(s[b].rank())
    z, se = np.arctanh(np.clip(r, -.999, .999)), 1.06 / np.sqrt(n - 3)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def pct_agree(data, col):
    return (data[col] >= 4).mean() * 100 if len(data) else np.nan


def fmt(v, suffix="%"):
    return "n/a" if pd.isna(v) else f"{v:.0f}{suffix}"


def stacked(frame, group, cat, order_g, order_c, colors, horizontal=False, height=300):
    ct = (pd.crosstab(frame[group], frame[cat], normalize="index") * 100).reindex(index=order_g, columns=order_c).fillna(0)
    fig = go.Figure()
    for c in order_c:
        bright = colors[c] in (CYAN, AMBER, CORAL, "#9CF0FF", "#38A8E8")
        v = ct[c]
        kw = dict(y=ct.index, x=v, orientation="h") if horizontal else dict(x=ct.index, y=v)
        fig.add_bar(name=str(c), marker_color=colors[c], text=[f"{x:.0f}%" if x >= 6 else "" for x in v], textposition="inside",
                    insidetextanchor="middle", textfont=dict(color=INK if bright else "#fff", size=12),
                    hovertemplate="%{" + ("x" if horizontal else "y") + ":.1f}% " + str(c) + "<extra></extra>", **kw)
    fig.update_layout(barmode="stack", bargap=.35, legend=dict(traceorder="normal", y=-0.3 if horizontal else -0.14))
    if horizontal:
        fig.update_xaxes(range=[0, 100], ticksuffix="%", side="top")
        fig.update_yaxes(autorange="reversed")
    else:
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    html('<div style="font-family:Sora;font-size:1.25rem;font-weight:700;margin:4px 0 2px 0;">AI &amp; Students</div>'
         '<div class="note" style="margin-bottom:10px;">Filter the survey. Every chart updates.</div>')
    use_sel = st.selectbox("Daily AI use", ["All"] + ORDERS["daily_ai_use"])
    level_sel = st.selectbox("Academic level", ["All"] + sorted(df["academic_level"].unique().tolist()))
    ages = st.multiselect("Age group", ORDERS["age_group"], default=ORDERS["age_group"])
    uni_all = sorted(df["university_group"].unique().tolist())
    unis = st.multiselect("University", uni_all, default=uni_all)
    cgpas = st.multiselect("CGPA band", ORDERS["cgpa_band"], default=ORDERS["cgpa_band"])
    with st.expander("Score ranges"):
        rel_range = st.slider("Reliance score", 1.0, 5.0, (1.0, 5.0), step=0.2)
        cog_range = st.slider("Cognitive concern score", 1.0, 5.0, (1.0, 5.0), step=0.2)

mask = (df["age_group"].isin(ages) & df["university_group"].isin(unis) & df["cgpa_band"].isin(cgpas)
        & df["reliance_score"].between(*rel_range) & df["cognitive_concern_score"].between(*cog_range))
if use_sel != "All":
    mask &= df["daily_ai_use"] == use_sel
if level_sel != "All":
    mask &= df["academic_level"] == level_sel
f = df[mask]
with st.sidebar:
    st.progress(len(f) / len(df))
    html(f'<div class="note"><b style="color:{TEXT}">{len(f):,}</b> of {len(df):,} respondents selected</div>')

# ------------------------------------------------------------------ hero
def node(x, y, c, r=4.5):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>'


_in = [(30, 20 + 25 * i) for i in range(9)]
_mid = {"rel": (330, 100), "cog": (330, 175)}
_edges = "".join(f'<line x1="{x}" y1="{y}" x2="{mx}" y2="{my}" stroke="{CYAN}" stroke-opacity=".16"/>' for x, y in _in for mx, my in _mid.values())
_edges += "".join(f'<line x1="{mx}" y1="{my}" x2="565" y2="138" stroke="{AMBER}" stroke-width="1.6" stroke-dasharray="5 5" stroke-opacity=".8"/>' for mx, my in _mid.values())
_edges += f'<line x1="330" y1="28" x2="330" y2="100" stroke="{VIOLET}" stroke-width="2"/>'
_edges += f'<path d="M330 28 C 410 62, 410 142, 330 175" fill="none" stroke="{VIOLET}" stroke-width="2"/>'
_lab = "font-size:12px;fill:#93A0C4;paint-order:stroke;stroke:#111A33;stroke-width:5px;font-family:Manrope,sans-serif"
HERO_SVG = (
    '<svg viewBox="0 0 600 250" role="img" aria-label="Study map: nine statements feed two scores, which are compared with CGPA">'
    + _edges + "".join(node(x, y, CYAN) for x, y in _in)
    + node(330, 28, VIOLET, 8) + node(*_mid["rel"], CORAL, 9) + node(*_mid["cog"], CORAL, 9) + node(565, 138, AMBER, 10)
    + f'<text x="344" y="32" style="{_lab}">Daily AI use</text><text x="346" y="104" style="{_lab}">Reliance</text>'
    + f'<text x="346" y="179" style="{_lab}">Cognitive concern</text><text x="565" y="166" text-anchor="middle" style="{_lab}">CGPA</text>'
    + f'<text x="30" y="246" style="{_lab}">9 survey statements</text></svg>'
)
html(f"""
<div class="hero"><div>
<h1>AI dependency and academic vulnerability</h1>
<p>{len(df):,} Bangladeshi university students described how much they use AI, how much they lean on it, and how they feel about their own thinking.
This dashboard asks whether heavier use goes with more reliance, and whether grades tell the same story.</p>
<div class="tags"><span>Self-reported survey</span><span>Mendeley Data, DOI 10.17632/6s6nvmpgbb.1</span><span>Associations, not causes</span></div>
</div><div>{HERO_SVG}<div class="maplegend">Study map: statements form two scores, then compared with grades (dashed = tested link)</div></div></div>
""")

if f.empty:
    st.warning("No respondents match the current filters. Widen the filters in the sidebar.")
    st.stop()
if len(f) < 30:
    st.warning(f"Only {len(f)} respondents selected: treat the results below as very uncertain.")

# ------------------------------------------------------------------ KPI row
def kpi(col, label, value, now, base, accent, is_pct=True):
    same = len(f) == len(df)
    delta = "All respondents" if same else (f"{'▲' if now >= base else '▼'} {abs(now - base):.0f} pts vs all" if is_pct
                                           else f"{'▲' if now >= base else '▼'} {abs(now - base):.2f} vs all")
    col.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="l">{label}</div><div class="v">{value}</div><div class="d">{delta}</div></div>',
                 unsafe_allow_html=True)


c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(f'<div class="kpi" style="--accent:{CYAN}"><div class="l">Respondents</div><div class="v">{len(f):,}</div>'
            f'<div class="d">of {len(df):,} after cleaning</div></div>', unsafe_allow_html=True)
for col, lab, a, b, acc in [
    (c2, "Use AI more than 3 hours a day", (f["daily_ai_use"] == "More than 3 hours").mean() * 100, (df["daily_ai_use"] == "More than 3 hours").mean() * 100, VIOLET),
    (c3, "High reliance on AI", f["high_reliance"].mean() * 100, df["high_reliance"].mean() * 100, CORAL),
    (c4, "Say their critical thinking weakened", pct_agree(f, "critical_thinking_weakened"), pct_agree(df, "critical_thinking_weakened"), AMBER),
    (c5, "CGPA above 3.50", f["high_cgpa"].mean() * 100, df["high_cgpa"].mean() * 100, CYAN),
]:
    kpi(col, lab, f"{a:.0f}%", a, b, acc)
st.write("")

tabs = st.tabs(["Overview", "Habits", "Reliance and thinking", "Grades", "Who is most reliant", "Data and method"])

# ================================================================== Overview
with tabs[0]:
    left, right = st.columns([3, 2])
    pairs = [("daily_ai_use_ord", "reliance_score", "More daily AI use and higher reliance"),
             ("daily_ai_use_ord", "cognitive_concern_score", "More daily AI use and feeling thinking weaken"),
             ("daily_ai_use_ord", "verify_ai_info", "More daily AI use and verifying AI answers"),
             ("reliance_score", "cognitive_concern_score", "Higher reliance and feeling thinking weaken"),
             ("daily_ai_use_ord", "cgpa_ord", "More daily AI use and higher CGPA"),
             ("reliance_score", "cgpa_ord", "Higher reliance and higher CGPA"),
             ("cognitive_concern_score", "cgpa_ord", "Feeling thinking weaken and higher CGPA")]
    rows = [(lab, *rho_ci(a, b)) for a, b, lab in pairs]
    fr = pd.DataFrame(rows, columns=["label", "rho", "lo", "hi"]).dropna()
    with left:
        with panel("forest", "What is connected to what?",
                   "Each dot is a Spearman correlation (-1 to +1) with its 95% confidence range. Dots inside the shaded band are negligible."):
            if fr.empty:
                st.info("Not enough respondents to compute correlations.")
            else:
                fr = fr.iloc[::-1]
                col_ = [MUTED if abs(r) < .1 else CYAN if r > 0 else CORAL for r in fr["rho"]]
                fig = go.Figure()
                fig.add_vrect(x0=-.1, x1=.1, fillcolor=MUTED, opacity=.14, line_width=0)
                fig.add_scatter(x=fr["rho"], y=fr["label"], mode="markers+text", text=[f"{r:+.2f}" for r in fr["rho"]], textposition="top center",
                                textfont=dict(color=TEXT, size=12), marker=dict(size=13, color=col_, line=dict(color=INK, width=2)),
                                error_x=dict(type="data", symmetric=False, array=fr["hi"] - fr["rho"], arrayminus=fr["rho"] - fr["lo"], color=MUTED, thickness=1.6, width=5),
                                hovertemplate="%{y}<br>rho %{x:+.2f}<extra></extra>")
                fig.update_xaxes(range=[-1, 1], zeroline=True, title="Spearman correlation")
                fig.update_layout(showlegend=False)
                show(fig, 380)
        r_rel, r_cg = rho_ci("daily_ai_use_ord", "reliance_score")[0], rho_ci("reliance_score", "cgpa_ord")[0]
        answer(f"Daily AI use has <b>{link(r_rel)}</b> with reliance (rho {r_rel:+.2f}), while reliance has <b>{link(r_cg)}</b> with CGPA (rho {r_cg:+.2f}). "
               "Compare the dots near the top with the CGPA rows: attitudes move together, grades much less.")
    with right:
        with panel("donut", "How much do students use AI each day?", "Share of the selected respondents."):
            vc = f["daily_ai_use"].value_counts().reindex(ORDERS["daily_ai_use"]).fillna(0)
            fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=.62, sort=False, marker=dict(colors=[USE_COLORS[k] for k in vc.index], line=dict(color=PANEL, width=3)),
                                   textinfo="percent", textfont=dict(size=13, color=INK), hovertemplate="%{label}: %{value} students<extra></extra>"))
            fig.update_layout(showlegend=True, annotations=[dict(text=f"<b>{len(f):,}</b><br>students", showarrow=False, font=dict(size=15, color=TEXT))])
            show(fig, 380)

    with panel("who", "Who answered the survey?", "Counts for the current selection. Small groups give unreliable averages later on."):
        cols = st.columns(3)
        for col, key, title in zip(cols, ["academic_level", "age_group", "university_group"], ["Academic level", "Age group", "University"]):
            vc = f[key].value_counts()
            vc = vc.reindex(ORDERS[key]) if key in ORDERS else vc
            vc = vc.fillna(0).sort_values() if key != "age_group" else vc.fillna(0)[::-1]
            fig = go.Figure(go.Bar(x=vc.values, y=vc.index.astype(str), orientation="h", marker_color=VIOLET, text=vc.values.astype(int), textposition="outside",
                                   textfont=dict(color=TEXT, size=12), cliponaxis=False, hovertemplate="%{y}: %{x} students<extra></extra>"))
            fig.update_xaxes(visible=False)
            fig.update_layout(title=dict(text=title, font=dict(size=13, color=MUTED), x=0), bargap=.4, margin=dict(l=8, r=30, t=34, b=8))
            with col:
                show(fig, 260)
    top = f["university"].value_counts()
    answer(f"One university ({top.index[0]}) supplies {top.iloc[0] / len(f) * 100:.0f}% of this selection, so the results describe this sample rather than all students in Bangladesh.", warn=True)

# ================================================================== Habits
with tabs[1]:
    with panel("likert", "What do students admit about their own AI habits?",
               "Each bar splits the answers from strongly disagree (left, red) to strongly agree (right, blue). Sorted by agreement."):
        pct = pd.DataFrame({ITEM_LABELS[c]: f[c].value_counts(normalize=True).reindex(range(1, 6)).fillna(0) * 100 for c in ITEMS}).T
        pct = pct.loc[(pct[4] + pct[5]).sort_values().index]
        left_ = -(pct[1] + pct[2] + pct[3] / 2)
        fig = go.Figure()
        for k, lab, colr in zip(range(1, 6), LIKERT_LABELS, LIKERT_COLORS):
            fig.add_bar(y=pct.index, x=pct[k], base=left_, orientation="h", name=lab, marker=dict(color=colr, line=dict(color=PANEL, width=1.5)),
                        text=[f"{v:.0f}%" if v >= 7 else "" for v in pct[k]], textposition="inside", insidetextanchor="middle",
                        textfont=dict(color=INK if k in (4, 5) else "#fff", size=12), hovertemplate="%{y}<br>" + lab + ": %{x:.1f}%<extra></extra>")
            left_ = left_ + pct[k]
        fig.add_vline(x=0, line_color=TEXT, line_width=1, opacity=.5)
        fig.update_layout(barmode="overlay", bargap=.3, legend=dict(orientation="h", y=-0.12))
        fig.update_xaxes(tickvals=[-50, -25, 0, 25, 50, 75, 100], ticktext=["50%", "25%", "0", "25%", "50%", "75%", "100%"], title="Disagree  <  >  Agree")
        show(fig, 470)
    agree = {ITEM_LABELS[c]: pct_agree(f, c) for c in ITEMS}
    hi_i, lo_i = max(agree, key=agree.get), min(agree, key=agree.get)
    answer(f"Most agreed: <b>{hi_i}</b> ({agree[hi_i]:.0f}% agree). Least agreed: <b>{lo_i}</b> ({agree[lo_i]:.0f}%). "
           "These are opinions about oneself, not tested skills, and some wordings are leading.")

    with panel("dumb", "What changes when students use AI for longer?",
               "Share of students who agree (4 or 5) with each statement, by daily AI use. A tick marks a protective habit, where higher is better. Longest gaps on top."):
        groups = [g for g in ORDERS["daily_ai_use"] if (f["daily_ai_use"] == g).sum() >= 5]
        if len(groups) < 2:
            st.info("Select at least two daily-use levels (set the Daily AI use filter to All) to compare them.")
        else:
            ag = {g: {c: pct_agree(f[f["daily_ai_use"] == g], c) for c in ITEMS} for g in groups}
            lo_g, hi_g = groups[0], groups[-1]
            order = sorted(ITEMS, key=lambda c: abs(ag[hi_g][c] - ag[lo_g][c]))
            lab = {c: ("✓ " if c in PROTECTIVE else "") + ITEM_LABELS[c] for c in ITEMS}
            xs, ys = [], []
            for c in order:
                xs += [ag[lo_g][c], ag[hi_g][c], None]
                ys += [lab[c], lab[c], None]
            fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#3A4A7D", width=4), hoverinfo="skip", showlegend=False))
            for g in groups:
                fig.add_scatter(x=[ag[g][c] for c in order], y=[lab[c] for c in order], mode="markers", name=g,
                                marker=dict(size=14, color=USE_COLORS[g], line=dict(color=INK, width=2)), hovertemplate=g + ": %{x:.0f}% agree<extra></extra>")
            fig.update_xaxes(range=[0, 100], ticksuffix="%", title="% who agree or strongly agree")
            show(fig, 460)
            big = max(ITEMS, key=lambda c: abs(ag[hi_g][c] - ag[lo_g][c]))
            answer(f"The biggest shift from '{lo_g}' to '{hi_g}' users is for <b>{ITEM_LABELS[big]}</b>: {ag[lo_g][big]:.0f}% to {ag[hi_g][big]:.0f}% agree "
                   f"({ag[hi_g][big] - ag[lo_g][big]:+.0f} points).")

# ================================================================== Reliance and thinking
with tabs[2]:
    a, b = st.columns(2)
    with a:
        with panel("relmix", "Does heavier use mean more dependence?", "Share of students in each reliance level, by daily AI use."):
            show(stacked(f, "daily_ai_use", "reliance_level", ORDERS["daily_ai_use"], ORDERS["reliance_level"], LEVEL_COLORS), 340)
    with b:
        with panel("relmean", "How high is the average reliance score?", "Mean on the 1-5 scale with a 95% confidence range."):
            g = mean_ci(f, "daily_ai_use", "reliance_score")
            fig = go.Figure(go.Scatter(x=g["daily_ai_use"].astype(str), y=g["mean"], mode="lines+markers+text", text=[f"{v:.2f}" for v in g["mean"]], textposition="top center",
                                       textfont=dict(color=TEXT, size=13), line=dict(color=VIOLET, width=3), marker=dict(size=14, color=CORAL, line=dict(color=INK, width=2)),
                                       error_y=dict(type="data", array=g["ci"], color=MUTED, thickness=1.6, width=6), customdata=g["count"],
                                       hovertemplate="%{x}<br>mean %{y:.2f} (n=%{customdata})<extra></extra>"))
            fig.update_yaxes(range=[1, 5], dtick=1, title="Reliance score (1 = low, 5 = high)")
            fig.update_xaxes(categoryorder="array", categoryarray=ORDERS["daily_ai_use"])
            show(fig, 340)
    r, lo, hi = rho_ci("daily_ai_use_ord", "reliance_score")
    mm = f.groupby("daily_ai_use", observed=True)["reliance_score"].mean()
    answer(f"Daily AI use and reliance show <b>{link(r)}</b> (rho {r:+.2f}, 95% range {lo:+.2f} to {hi:+.2f}). Mean reliance: "
           + ", ".join(f"{k} {v:.2f}" for k, v in mm.items()) + ".")

    with panel("cog", "Do students who rely more on AI feel their thinking weakening?",
               "Share who agree with each statement about memory, confidence and critical thinking, by reliance level."):
        short = {"harder_to_recall": "Harder to recall AI-generated info", "less_confident_without_ai": "Less confident without AI",
                 "critical_thinking_weakened": "Critical thinking has weakened"}
        fig = go.Figure()
        for c, colr in zip(COGNITIVE, [CYAN, VIOLET, CORAL]):
            vals = [pct_agree(f[f["reliance_level"] == lv], c) for lv in ORDERS["reliance_level"]]
            fig.add_bar(x=ORDERS["reliance_level"], y=vals, name=short[c], marker_color=colr, text=[fmt(v) for v in vals], textposition="outside",
                        textfont=dict(color=TEXT, size=12), cliponaxis=False, hovertemplate="%{x} reliance: %{y:.0f}% agree<extra>" + short[c] + "</extra>")
        fig.update_layout(barmode="group", bargap=.28, bargroupgap=.06)
        fig.update_yaxes(range=[0, 108], ticksuffix="%", title="% who agree")
        fig.update_xaxes(title="Reliance level")
        show(fig, 360)
    r2, lo2, hi2 = rho_ci("reliance_score", "cognitive_concern_score")
    cl = pct_agree(f[f["reliance_level"] == "Low"], "critical_thinking_weakened")
    ch = pct_agree(f[f["reliance_level"] == "High"], "critical_thinking_weakened")
    answer(f"Reliance and cognitive concern show <b>{link(r2)}</b> (rho {r2:+.2f}). Saying 'my critical thinking has weakened': {fmt(cl)} of low-reliance vs {fmt(ch)} of high-reliance students. "
           "Both are self-ratings by the same people, so part of this link may be a shared response style.", warn=False)

# ================================================================== Grades
with tabs[3]:
    a, b = st.columns(2)
    with a:
        with panel("cgpamix", "Do heavy AI users report lower CGPA?", "CGPA bands within each daily-use group (100% bars)."):
            show(stacked(f, "daily_ai_use", "cgpa_band", ORDERS["daily_ai_use"], ORDERS["cgpa_band"], dict(zip(ORDERS["cgpa_band"], CGPA_COLORS)), horizontal=True), 330)
    with b:
        with panel("cgpahigh", "Is the share of top grades different?", "Students with CGPA above 3.50. Whiskers are 95% confidence ranges: if they overlap, the difference may be chance."):
            fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=.06, subplot_titles=("By daily AI use", "By reliance level"))
            SHORT = {"Less than 1 hour": "Under 1 h", "1-3 hours": "1-3 h", "More than 3 hours": "Over 3 h", "Low": "Low", "Moderate": "Moderate", "High": "High"}
            ymax = 10
            for i, (col, order, cm) in enumerate([("daily_ai_use", ORDERS["daily_ai_use"], USE_COLORS), ("reliance_level", ORDERS["reliance_level"], LEVEL_COLORS)], 1):
                st_ = [wilson(int(f.loc[f[col] == o, "high_cgpa"].sum()), int((f[col] == o).sum())) for o in order]
                p_ = np.array([s_[0] for s_ in st_])
                ymax = max(ymax, np.nanmax([s_[2] for s_ in st_]) + 8)
                fig.add_trace(go.Bar(x=[SHORT[o] for o in order], y=p_, marker_color=[cm[o] for o in order],
                                     text=[fmt(v) for v in p_], textposition="inside", insidetextanchor="start", textfont=dict(color=INK, size=14),
                                     error_y=dict(type="data", symmetric=False, array=[s_[2] - s_[0] for s_ in st_], arrayminus=[s_[0] - s_[1] for s_ in st_], color=TEXT, thickness=1.6, width=6),
                                     hovertemplate="%{x}: %{y:.0f}%<extra></extra>", showlegend=False), row=1, col=i)
            fig.update_yaxes(range=[0, min(100, ymax)], ticksuffix="%")
            fig.update_layout(bargap=.35)
            fig.update_annotations(font=dict(size=12, color=MUTED))
            show(fig, 330)
    r1, l1, h1 = rho_ci("daily_ai_use_ord", "cgpa_ord")
    r2_, l2_, h2_ = rho_ci("reliance_score", "cgpa_ord")
    share = f.groupby("daily_ai_use", observed=True)["high_cgpa"].mean() * 100
    answer(f"Daily AI use has <b>{link(r1)}</b> with CGPA band (rho {r1:+.2f}, range {l1:+.2f} to {h1:+.2f}); reliance has <b>{link(r2_)}</b> with CGPA band (rho {r2_:+.2f}). "
           "Share with CGPA above 3.50: " + ", ".join(f"{k} {v:.0f}%" for k, v in share.items()) + ".")
    n_low = int((f["cgpa_band"] == "Below 2.50").sum())
    answer(f"CGPA is a self-reported band with only 4 levels, and just {n_low} respondents are in the lowest band. A weak link here does not prove that AI use is harmless for learning.", warn=True)

# ================================================================== Who is most reliant
with tabs[4]:
    a, b = st.columns([1, 1])
    with a:
        with panel("heat", "Which combinations of use and level are the most reliant?", "Mean reliance score (1-5). Cells with fewer than 10 students are left blank."):
            pv = f.pivot_table(index="daily_ai_use", columns="academic_level", values="reliance_score", aggfunc="mean", observed=True)
            cn = f.pivot_table(index="daily_ai_use", columns="academic_level", values="reliance_score", aggfunc="count", observed=True)
            pv = pv.where(cn >= 10).dropna(how="all").dropna(axis=1, how="all")
            if pv.empty:
                st.info("Not enough respondents per cell under the current filters.")
            else:
                pv = pv.reindex([u for u in ORDERS["daily_ai_use"] if u in pv.index])
                txt = [[("" if pd.isna(v) else f"{v:.2f}") for v in r] for r in pv.values]
                fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns.astype(str), y=pv.index.astype(str), text=txt, texttemplate="%{text}", textfont=dict(size=14, color="#fff"),
                                           colorscale=[[0, "#1B2A5E"], [.5, VIOLET], [1, CORAL]], xgap=4, ygap=4, hoverongaps=False,
                                           colorbar=dict(title=dict(text="mean", font=dict(color=MUTED, size=11)), thickness=10, tickfont=dict(color=MUTED)),
                                           hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
                fig.update_yaxes(autorange="reversed")
                show(fig, 330)
                s_ = pv.stack()
                answer(f"Highest cell: <b>{s_.idxmax()[0]}</b> students at the <b>{s_.idxmax()[1]}</b> level (mean {s_.max():.2f}); lowest: {s_.idxmin()[0]} / {s_.idxmin()[1]} ({s_.min():.2f}).")
    with b:
        with panel("risk", "Which groups have the most students with high reliance?", "Share with a high reliance score (above 3.5). Whiskers are 95% confidence ranges. Coral = above the average of all selected students; groups under 10 are hidden."):
            risk_opts = [k for k in GROUPINGS if k != "daily_ai_use"]
            grp = st.selectbox("Compare by", risk_opts, index=risk_opts.index("university_group"), format_func=GROUPINGS.get, key="risk_grp", label_visibility="collapsed")
            rows = []
            for name, sub in f.groupby(grp, observed=True):
                if len(sub) >= 10:
                    p_, l_, h_ = wilson(int(sub["high_reliance"].sum()), len(sub))
                    rows.append((f"{name} (n={len(sub)})", p_, p_ - l_, h_ - p_))
            if len(rows) < 2:
                st.info("Not enough groups with 10 or more students under the current filters.")
            else:
                rk = pd.DataFrame(rows, columns=["g", "p", "lo", "hi"]).sort_values("p")
                avg = f["high_reliance"].mean() * 100
                fig = go.Figure(go.Bar(x=rk["p"], y=rk["g"], orientation="h", marker_color=[CORAL if v > avg else CYAN for v in rk["p"]], text=[fmt(v) for v in rk["p"]],
                                       textposition="inside", insidetextanchor="start", textfont=dict(color=INK, size=13),
                                       error_x=dict(type="data", symmetric=False, array=rk["hi"], arrayminus=rk["lo"], color=MUTED, thickness=1.4, width=4),
                                       hovertemplate="%{y}: %{x:.0f}% high reliance<extra></extra>"))
                fig.add_vline(x=avg, line_dash="dash", line_color=AMBER, annotation_text=f"all: {avg:.0f}%", annotation_font_color=AMBER, annotation_position="top")
                fig.update_xaxes(range=[0, 100], ticksuffix="%")
                fig.update_layout(bargap=.35, margin=dict(l=8, r=36, t=28, b=8))
                show(fig, 300)
                answer(f"<b>{rk.iloc[-1]['g']}</b> has the highest share ({rk.iloc[-1]['p']:.0f}%), <b>{rk.iloc[0]['g']}</b> the lowest ({rk.iloc[0]['p']:.0f}%). "
                       "If the grey ranges of two groups overlap, the difference may be chance.")

# ================================================================== Data and method
with tabs[5]:
    top = df["university"].value_counts()
    cards = [(f"{RAW_N:,}", "raw responses", MUTED), (f"{RAW_N - len(df):,}", "duplicate responses removed", CORAL), (f"{len(df):,}", "responses analysed", CYAN),
             (f"{top.iloc[0] / len(df) * 100:.0f}%", f"from one university ({top.index[0]})", AMBER), (f"{int((top < 10).sum())}", f"of {len(top)} universities have under 10 students", VIOLET)]
    for col, (v, l, acc) in zip(st.columns(5), cards):
        col.markdown(f'<div class="kpi" style="--accent:{acc}"><div class="v">{v}</div><div class="d" style="font-size:.84rem">{l}</div></div>', unsafe_allow_html=True)
    st.write("")
    with st.expander("About the dataset and the two scores", expanded=False):
        st.markdown(
            "**AI Dependency and Academic Vulnerability in Higher Education** - survey of Bangladeshi university students "
            "(Mendeley Data, DOI 10.17632/6s6nvmpgbb.1). Students reported gender, age group, university, academic level, CGPA band and daily AI use, "
            "then rated nine statements from 1 (strongly disagree) to 5 (strongly agree).\n\n"
            "- **Reliance score:** asking AI first, speed over understanding, using answers unmodified, submitting work not understood, and *not* being able to study without AI (reverse-coded).\n"
            "- **Cognitive concern score:** harder recall, less confidence without AI, weaker critical thinking.\n\n"
            "Self-reported answers from volunteers; one university supplies about 3 in 10 responses. Results show **associations**, not causes.")
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        st.markdown("##### Summary statistics")
        st.dataframe(f[["reliance_score", "cognitive_concern_score"] + ITEMS].rename(columns=ITEM_LABELS).describe().T.round(2), width="stretch")
    with cc2:
        st.markdown("##### Distribution")
        cat = st.selectbox("Category", list(GROUPINGS), format_func=GROUPINGS.get, key="dist", label_visibility="collapsed")
        order = ORDERS.get(cat, f[cat].value_counts().index.tolist())
        vc = f[cat].value_counts().reindex(order).fillna(0)
        fig = go.Figure(go.Bar(x=vc.index.astype(str), y=vc.values, marker_color=VIOLET, hovertemplate="%{x}: %{y}<extra></extra>"))
        fig.update_layout(bargap=.4)
        show(fig, 260)
    st.markdown("##### Data preview")
    n_rows = st.slider("Rows to show", 5, 100, 10, key="rows")
    st.dataframe(f.drop(columns=["university_group", "daily_ai_use_ord", "cgpa_ord", "high_cgpa", "high_reliance"]).head(n_rows), width="stretch")
    st.download_button("Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"), "filtered_survey.csv", "text/csv")

    with st.expander("Build your own chart"):
        e1, e2, e3, e4 = st.columns(4)
        kind = e1.selectbox("Chart type", ["Box", "Bar (mean)", "Histogram", "Heatmap (% of row)"])
        x = e2.selectbox("Group (X)", list(GROUPINGS), format_func=GROUPINGS.get, key="ex_x")
        y = e3.selectbox("Score (Y)", list(MEASURES), format_func=MEASURES.get, key="ex_y")
        color = e4.selectbox("Color by", ["(none)"] + list(GROUPINGS), format_func=lambda c: GROUPINGS.get(c, c), key="ex_c")
        col = None if color == "(none)" else color
        xo = {x: ORDERS.get(x, sorted(f[x].astype(str).unique()))}
        if kind == "Box":
            fig = px.box(f, x=x, y=y, color=col, category_orders=xo)
        elif kind == "Bar (mean)":
            gg = f.groupby([x] + ([col] if col and col != x else []), observed=True)[y].mean().reset_index()
            fig = px.bar(gg, x=x, y=y, color=col if col and col != x else None, barmode="group", category_orders=xo)
        elif kind == "Histogram":
            fig = px.histogram(f, x=y, color=col, nbins=20, barmode="overlay", opacity=0.75)
        else:
            other = col if col and col != x else ("cgpa_band" if x != "cgpa_band" else "daily_ai_use")
            ct = pd.crosstab(f[x], f[other], normalize="index") * 100
            fig = px.imshow(ct.round(0), text_auto=True, aspect="auto", color_continuous_scale=[[0, "#1B2A5E"], [1, CYAN]], labels={"color": "% of row"})
        fig.update_layout(legend_title_text="")
        show(fig, 380)

html(f'<div class="note" style="text-align:center;margin-top:22px;">Final Project - Unit 3 &nbsp;|&nbsp; Built with Streamlit, pandas and Plotly &nbsp;|&nbsp; Data: Mendeley Data, DOI 10.17632/6s6nvmpgbb.1</div>')
