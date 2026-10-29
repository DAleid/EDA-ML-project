import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="InsightLab | Superstore Analytics", page_icon="◍", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- Visual identity (light, green, rounded cards) ----------
GREEN_DK, GREEN, GREEN_MID, GREEN_LT, GREEN_XLT = "#0F4D32", "#1F7A4D", "#3FA06B", "#8ED1A9", "#E6F4EC"
TEXT, MUTED, PANEL, BG = "#14201A", "#7B857F", "#FFFFFF", "#EEF0EE"
GRID, LINE = "#EDF0ED", "#E3E8E3"
AMBER, RED = "#E8A33D", "#D64545"
TEAL, BLUE, PURPLE, ORANGE = GREEN_MID, GREEN_DK, GREEN_LT, AMBER  # roles used by the charts
PALETTE = [GREEN_DK, GREEN_MID, GREEN_LT, AMBER, "#5B8DEF", RED, "#B6C2BA", "#7A5CFA"]
DIVERGING = [RED, "#F1F3F1", GREEN_MID]
SEQ_GREEN = ["#E8F5EC", GREEN_LT, GREEN_DK]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root { color-scheme: light; }
html, body, .stApp, .stMarkdown, p, label, input, textarea, button { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: __BG__; color: __TEXT__; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.3rem; padding-bottom: 3rem; max-width: 1500px; }
footer { visibility: hidden; }

/* sidebar */
[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid __LINE__; }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
.brand { display:flex; align-items:center; gap:10px; font-weight:800; font-size:21px; color:__TEXT__; margin-bottom:8px; }
.brand .logo { width:34px; height:34px; border-radius:50%; border:3px solid __GREEN__; display:flex; align-items:center; justify-content:center; }
.brand .logo:after { content:''; width:11px; height:11px; border-radius:50%; background:__GREEN__; }
.nav-label { color:__MUTED__; font-size:11px; letter-spacing:1.5px; font-weight:700; margin:18px 0 6px 6px; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] div[role="radiogroup"] label { padding: 10px 14px; border-radius: 12px; width: 100%; cursor: pointer; transition: background .15s; }
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] label p { font-size: 14.5px; color: #8A938D; font-weight: 500; margin: 0; }
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: #F4F7F4; }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background: __XLT__; box-shadow: inset 3px 0 0 __GREEN__; }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color: __TEXT__; font-weight: 700; }
.side-card { margin-top: 26px; border-radius: 20px; padding: 18px; color: #fff;
  background: radial-gradient(circle at 85% 15%, #2f8f5e 0%, #0d3f29 58%, #082b1c 100%); }
.side-card .t { font-weight:700; font-size:15px; margin-bottom:4px; }
.side-card .s { font-size:12px; color:rgba(255,255,255,.72); line-height:1.5; }

/* header + buttons */
.page-title { font-size:36px; font-weight:800; letter-spacing:-.8px; margin:0; line-height:1.1; color:__TEXT__; }
.page-sub { color:__MUTED__; font-size:13.5px; margin-top:6px; }
.st-key-btn_filters button, .st-key-btn_import button, .stDownloadButton button { border-radius: 999px; padding: .55rem 1.3rem; font-weight: 600; }
.st-key-btn_filters button { background: linear-gradient(135deg, __GREEN__, __GREEN_DK__); border: none; }
.st-key-btn_filters button p, .st-key-btn_filters button span { color: #fff !important; }
.st-key-btn_import button { background: #fff; border: 1.5px solid __GREEN__; }
.st-key-btn_import button p, .st-key-btn_import button span { color: __GREEN__ !important; }
.stDownloadButton button { background: #fff; border: 1.5px solid __GREEN__; color: __GREEN__; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 14px; }
.chip-f { background:__XLT__; color:__GREEN__; border-radius:999px; padding:4px 12px; font-size:12px; font-weight:600; }

/* cards */
.kpi { background:#fff; border:1px solid __LINE__; border-radius:22px; padding:18px 20px; min-height:138px; box-shadow:0 1px 2px rgba(20,32,26,.03); }
.kpi-top { display:flex; justify-content:space-between; align-items:center; }
.kpi-label { font-size:14px; font-weight:600; color:__TEXT__; }
.kpi-arrow { width:28px; height:28px; border-radius:50%; border:1.5px solid __TEXT__; display:flex; align-items:center; justify-content:center; font-size:13px; }
.kpi-value { font-size:44px; font-weight:700; letter-spacing:-1.5px; margin:10px 0 8px; line-height:1; color:__TEXT__; }
.kpi-note { color:__MUTED__; font-size:11.5px; }
.chip { border-radius:6px; padding:2px 7px; font-weight:700; margin-right:4px; }
.chip.up { color:__GREEN__; background:__XLT__; }
.chip.down { color:__RED__; background:#FBE9E9; }
.kpi.dark { background: linear-gradient(145deg, #1F7A4D 0%, #0F4D32 100%); border-color:#0F4D32; }
.kpi.dark .kpi-label, .kpi.dark .kpi-value { color:#fff; }
.kpi.dark .kpi-arrow { background:#fff; color:__GREEN_DK__; border-color:#fff; }
.kpi.dark .kpi-note { color:rgba(255,255,255,.78); }
.kpi.dark .chip.up { background:rgba(255,255,255,.18); color:#C6F3DA; }
.kpi.dark .chip.down { background:rgba(255,255,255,.18); color:#FFC2C2; }
.card { background:#fff; border:1px solid __LINE__; border-radius:22px; padding:18px 20px; }
.card h4 { margin:0 0 12px; font-size:16px; font-weight:700; }
.sig { display:flex; gap:10px; padding:10px 0; border-bottom:1px solid #F0F2F0; font-size:13px; line-height:1.5; }
.sig:last-child { border-bottom:none; }
.sig .dot { width:9px; height:9px; border-radius:50%; background:__GREEN_MID__; margin-top:6px; flex:none; }
.sig strong { color:__GREEN__; }
.hero { padding: 34px 36px; border-radius: 26px; color:#fff; margin-bottom: 22px;
  background: radial-gradient(circle at 90% 10%, #2f8f5e 0%, #0d3f29 60%, #082b1c 100%); }
.hero .eyebrow { letter-spacing:2px; font-size:11px; font-weight:700; color:#9BE3BC; }
.hero h1 { font-size:38px; margin:10px 0 8px; color:#fff; }
.hero p { color:rgba(255,255,255,.78); max-width:760px; margin:0; font-size:14px; }
div[data-testid="stPlotlyChart"] { background:#fff; border:1px solid __LINE__; border-radius:22px; padding:10px 8px 0; }
div[data-testid="stMetric"] { background:#fff; border:1px solid __LINE__; padding:14px 16px; border-radius:18px; }
div[data-testid="stDataFrame"] { border:1px solid __LINE__; border-radius:16px; overflow:hidden; }
hr { border-color: __LINE__; }

/* top navigation (works even when the sidebar is collapsed) */
.st-key-topnav div[role="radiogroup"] { gap: 8px; flex-wrap: wrap; }
.st-key-topnav div[role="radiogroup"] label { background:#fff; border:1px solid __LINE__; border-radius:999px; padding:7px 16px; cursor:pointer; }
.st-key-topnav div[role="radiogroup"] label > div:first-child { display:none; }
.st-key-topnav div[role="radiogroup"] label p { font-size:13.5px; font-weight:600; color:#5F6B64; margin:0; }
.st-key-topnav div[role="radiogroup"] label:has(input:checked) { background:__GREEN__; border-color:__GREEN__; }
.st-key-topnav div[role="radiogroup"] label:has(input:checked) p { color:#fff; }

/* keep every widget readable and light, even if the browser/system is in dark mode */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp [data-testid="stWidgetLabel"] p { color: __TEXT__; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: __MUTED__ !important; }
[data-testid="stMetricLabel"] p { color: __MUTED__ !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: __TEXT__ !important; }
[data-testid="stPopoverBody"] { background:#fff !important; border:1px solid __LINE__; border-radius:18px; }
[data-testid="stPopoverBody"] p, [data-testid="stPopoverBody"] label, [data-testid="stPopoverBody"] span, [data-testid="stPopoverBody"] small { color: __TEXT__ !important; }
[data-testid="stPopoverBody"] div[data-baseweb="input"], [data-testid="stPopoverBody"] div[data-baseweb="select"] > div,
[data-testid="stPopoverBody"] div[data-baseweb="base-input"] { background:#F4F7F4 !important; border-color:__LINE__ !important; }
[data-testid="stPopoverBody"] input { color:__TEXT__ !important; -webkit-text-fill-color:__TEXT__ !important; }
[data-testid="stFileUploaderDropzone"] { background:#F4F7F4 !important; border:1px dashed #B9C6BE !important; }
[data-testid="stPopoverBody"] button { background:#fff !important; border:1.5px solid __GREEN__ !important; border-radius:999px !important; }
[data-testid="stPopoverBody"] button p, [data-testid="stPopoverBody"] button span { color:__GREEN__ !important; }
span[data-baseweb="tag"] { background:__XLT__ !important; border-radius:8px !important; }
span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg { color:__GREEN_DK__ !important; fill:__GREEN_DK__ !important; }
div[data-baseweb="popover"] ul[role="listbox"], div[data-baseweb="popover"] ul[role="listbox"] li { background:#fff !important; color:__TEXT__ !important; }
div[data-baseweb="popover"] ul[role="listbox"] li:hover { background:__XLT__ !important; }
div[data-baseweb="calendar"] { background:#fff !important; color:__TEXT__ !important; }
div[data-baseweb="calendar"] div { color:__TEXT__; }
</style>
"""
for k, v in {"__BG__": BG, "__TEXT__": TEXT, "__MUTED__": MUTED, "__LINE__": LINE, "__GREEN__": GREEN, "__GREEN_DK__": GREEN_DK,
             "__GREEN_MID__": GREEN_MID, "__XLT__": GREEN_XLT, "__RED__": RED}.items():
    CSS = CSS.replace(k, v)
st.markdown(CSS, unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(family="Plus Jakarta Sans", color=TEXT, size=12),
    margin=dict(l=18, r=18, t=52, b=18), title_font=dict(size=15, color=TEXT),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE),
)

_ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}


# ---------- Helpers ----------
def money(value):
    if pd.isna(value):
        return "—"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if v >= 1_000_000:
        return f"{sign}${v / 1_000_000:,.2f}M"
    if v >= 1_000:
        return f"{sign}${v / 1_000:,.1f}K"
    return f"{sign}${v:,.0f}"


def fmt_int(value):
    return "—" if pd.isna(value) else f"{value:,.0f}"


def has(df, *cols):
    return all(c in df.columns for c in cols)


def plot(fig, height=350):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    st.plotly_chart(fig, theme=None, **STRETCH)


def table(data):
    st.dataframe(data, hide_index=True, **STRETCH)


def hbar(data, x, y, title, color=GREEN_MID, height=380, **kw):
    plot(px.bar(data, x=x, y=y, orientation="h", title=title, color_discrete_sequence=[color], **kw), height)


def popover(label):
    try:
        return st.popover(label, **STRETCH)
    except TypeError:
        return st.popover(label)


def short(s, n=38):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


CANON = ["Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Customer Name",
         "Segment", "Country", "City", "State", "Postal Code", "Region", "Product ID", "Category",
         "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit"]


def _key(c):
    return re.sub(r"[\s_\-]+", "", str(c).lower())


CANON_MAP = {_key(c): c for c in CANON}


def clean_columns(df):
    df.columns = [CANON_MAP.get(_key(c), str(c).strip()) for c in df.columns]
    return df


def smart_to_datetime(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    s = series.astype("string").str.strip()
    parts = s.dropna().head(3000).str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.]\d{2,4}").dropna().astype(int)
    dayfirst = bool(len(parts)) and bool((parts[0] > 12).any())
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


@st.cache_data(show_spinner=False)
def prepare_data(raw):
    df = clean_columns(raw.copy())
    for col in ["Order Date", "Ship Date"]:
        if col in df:
            df[col] = smart_to_datetime(df[col])
    for col in ["Sales", "Profit", "Quantity", "Discount", "Postal Code"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if has(df, "Order Date", "Ship Date"):
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    if "Discount" in df:
        df["Discount Band"] = pd.cut(df["Discount"], bins=[-0.001, 0, 0.10, 0.20, 0.30, 0.50, 1.0],
                                     labels=["0%", "≤10%", "10–20%", "20–30%", "30–50%", ">50%"])
    return df


@st.cache_data(show_spinner="Reading file…")
def read_bytes(name, data):
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    try:
        return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="latin-1")


@st.cache_data(show_spinner="Reading file…")
def read_local(path, mtime):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_raw(uploaded):
    if uploaded is not None:
        try:
            return read_bytes(uploaded.name, uploaded.getvalue())
        except Exception as e:
            st.error(f"تعذّر قراءة الملف: {e}")
            st.stop()
    for name in ["Superstore_Cleaned.csv", "Superstore.csv", "train.csv"]:
        p = Path(name)
        if p.exists():
            try:
                return read_local(str(p), p.stat().st_mtime)
            except Exception:
                continue
    return None


def kpi_block(d):
    sales = d["Sales"].sum() if "Sales" in d else np.nan
    profit = d["Profit"].sum() if "Profit" in d else np.nan
    orders = d["Order ID"].nunique() if "Order ID" in d else len(d)
    return dict(
        sales=sales, profit=profit, orders=orders,
        customers=d["Customer ID"].nunique() if "Customer ID" in d else np.nan,
        units=d["Quantity"].sum() if "Quantity" in d else np.nan,
        aov=(sales / orders) if orders else np.nan,
        margin=(profit / sales * 100) if pd.notna(profit) and pd.notna(sales) and sales != 0 else np.nan,
    )


def delta_html(cur, prev):
    if prev is None or pd.isna(cur) or pd.isna(prev) or prev == 0:
        return None
    diff = cur - prev
    cls, arrow = ("up", "▲") if diff >= 0 else ("down", "▼")
    return f'<span class="chip {cls}">{arrow} {diff / abs(prev) * 100:+.1f}%</span>vs previous period'


# ---------- Navigation ----------
P_DASH, P_PROD, P_CUST = "◧  Dashboard", "▦  Products", "◎  Customers"
P_OPS, P_FC, P_DATA = "↗  Operations", "◔  Forecast", "⌘  Data quality"
TITLES = {P_DASH: ("Dashboard", "Track sales momentum, seasonality and the signals worth investigating."),
          P_PROD: ("Products", "Which sub-categories and products drive revenue?"),
          P_CUST: ("Customers & Regions", "Where do sales come from, and who are the key customers?"),
          P_OPS: ("Operations", "Shipping performance and discount behaviour."),
          P_FC: ("Forecast", "What do the next months look like, and how reliable is the model?"),
          P_DATA: ("Data quality", "Check the dataset health and download the filtered records.")}

PAGES = [P_DASH, P_PROD, P_CUST, P_OPS, P_FC, P_DATA]


def _side_changed():
    st.session_state["nav_top"] = st.session_state["page"]


def _top_changed():
    st.session_state["page"] = st.session_state["nav_top"]


with st.sidebar:
    st.markdown('<div class="brand"><span class="logo"></span>InsightLab</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-label">MENU</div>', unsafe_allow_html=True)
    st.session_state.setdefault("page", P_DASH)
    st.session_state.setdefault("nav_top", st.session_state["page"])
    page = st.radio("Navigation", PAGES, label_visibility="collapsed", key="page", on_change=_side_changed)
    info_slot = st.container()

# ---------- Header ----------
head_l, head_f, head_i = st.columns([3.3, 1, 1.25], vertical_alignment="center")
with head_l:
    t, sub = TITLES[page]
    st.markdown(f'<div class="page-title">{t}</div><div class="page-sub">{sub}</div>', unsafe_allow_html=True)
with head_i:
    with st.container(key="btn_import"):
        with popover("⬆ Import data"):
            uploaded = st.file_uploader("Upload Superstore data (CSV or Excel)", type=["csv", "xlsx", "xls"])
            st.caption("Expected fields include Sales, Order Date, Category, Region and Segment. "
                       "If no file is uploaded, `Superstore_Cleaned.csv` next to app.py is used. "
                       "The notebook saves the cleaned data and the forecast together in this one file.")

with st.container(key="topnav"):
    st.radio("Go to", PAGES, key="nav_top", horizontal=True, label_visibility="collapsed", on_change=_top_changed)
try:
    if st.context.theme.type == "dark":
        st.warning("الواجهة مصمّمة للثيم الفاتح، وجهازك على الوضع الداكن فقد تظهر بعض العناصر (مثل الجداول) بألوان غير متناسقة. "
                   "للإصلاح: ضع مجلد `.streamlit` (فيه ملف `config.toml`) بجانب `app.py` وأعد تشغيل التطبيق، "
                   "أو من القائمة ⋮ ← Settings ← Theme ← Light.")
except Exception:
    pass

raw = load_raw(uploaded)
fc_rows = None
if raw is not None and "Record Type" in raw.columns:
    # The notebook saves transactions and forecast rows in one file: separate them here.
    # Parse dates on the whole file first: the date format is detected from the order rows (day > 12),
    # so forecast months (always the 1st) are read correctly even if Excel re-saved the file as dd/mm/yyyy.
    if "Order Date" in raw.columns:
        raw = raw.assign(**{"Order Date": smart_to_datetime(raw["Order Date"])})
    is_fc = raw["Record Type"].astype(str).str.strip().str.lower().eq("forecast")
    fc_rows = raw[is_fc].copy() if is_fc.any() else None
    raw = raw[~is_fc].drop(columns=[c for c in ["Record Type", "Period", "Forecast", "Reference", "Low", "High"] if c in raw.columns])
    raw = raw.reset_index(drop=True)
    if "Row ID" in raw and raw["Row ID"].notna().all() and (raw["Row ID"] % 1 == 0).all():
        raw["Row ID"] = raw["Row ID"].astype("int64")
if raw is None:
    st.markdown("""
    <div class="hero"><div class="eyebrow">DATA INTELLIGENCE / LEARNING STUDIO</div><h1>InsightLab</h1>
    <p>An interactive analytics workspace for exploring Superstore sales, customers and operations.</p></div>
    """, unsafe_allow_html=True)
    st.info("ابدأ برفع ملف البيانات من زر Import data في أعلى الصفحة. يدعم التطبيق ملفات CSV وExcel، ويمكنه أيضًا قراءة `Superstore_Cleaned.csv` أو `Superstore.csv` إذا كان موجودًا بجانب `app.py`.")
    st.stop()

df = prepare_data(raw)
if df.empty:
    st.error("الملف لا يحتوي على صفوف بيانات.")
    st.stop()
if "Sales" not in df.columns:
    st.error("لم أجد عمود `Sales` في الملف. الأعمدة الموجودة: " + ", ".join(map(str, df.columns)))
    st.stop()

has_dates = "Order Date" in df and df["Order Date"].notna().any()
if has_dates:
    min_date, max_date = df["Order Date"].min().date(), df["Order Date"].max().date()
with info_slot:
    rng = f"{min_date:%b %Y} – {max_date:%b %Y}" if has_dates else "dates unavailable"
    st.markdown(f'<div class="side-card"><div class="t">Dataset</div><div class="s">{len(df):,} rows · {len(df.columns)} fields<br>{rng}<br>'
                f'Use <b>Filters</b> at the top to narrow the view.</div></div>', unsafe_allow_html=True)

# ---------- Filters (hidden behind a button) ----------
region_opts = sorted(df["Region"].dropna().astype(str).unique()) if "Region" in df else []
cat_opts = sorted(df["Category"].dropna().astype(str).unique()) if "Category" in df else []
seg_opts = sorted(df["Segment"].dropna().astype(str).unique()) if "Segment" in df else []
LIST_FILTERS = [("f_region", "Region", region_opts), ("f_cat", "Category", cat_opts), ("f_seg", "Customer segment", seg_opts)]

for key, _, opts in LIST_FILTERS:      # keep saved selections valid when a new file is loaded
    cur = st.session_state.get(key)
    if cur is None or any(x not in opts for x in cur):
        st.session_state[key] = list(opts)
if has_dates:
    cur = st.session_state.get("f_date")
    if not (isinstance(cur, (tuple, list)) and len(cur) >= 1 and all(min_date <= x <= max_date for x in cur)):
        st.session_state["f_date"] = (min_date, max_date)


def reset_filters():
    for k, _, o in LIST_FILTERS:
        st.session_state[k] = list(o)
    if has_dates:
        st.session_state["f_date"] = (min_date, max_date)


date_sel = st.session_state.get("f_date", ())
date_active = has_dates and isinstance(date_sel, (tuple, list)) and len(date_sel) == 2 and (date_sel[0], date_sel[1]) != (min_date, max_date)
chips = [f"{lab}: {', '.join(st.session_state[k]) or 'none'}" for k, lab, o in LIST_FILTERS if o and set(st.session_state[k]) != set(o)]
if date_active:
    chips.append(f"Dates: {date_sel[0]:%d %b %Y} → {date_sel[1]:%d %b %Y}")

with head_f:
    with st.container(key="btn_filters"):
        pop = popover(f"☰ Filters ({len(chips)})" if chips else "☰ Filters")
with pop:
    if has_dates:
        st.date_input("Order date range", min_value=min_date, max_value=max_date, key="f_date")
        if not (isinstance(st.session_state.get("f_date"), (tuple, list)) and len(st.session_state["f_date"]) == 2):
            st.caption("Select both a start and an end date.")
    for key, lab, opts in LIST_FILTERS:
        if opts:
            st.multiselect(lab, opts, key=key)
    st.button("Reset filters", on_click=reset_filters, **STRETCH)

base = df
for key, col, opts in [("f_region", "Region", region_opts), ("f_cat", "Category", cat_opts), ("f_seg", "Segment", seg_opts)]:
    if opts:
        base = base[base[col].astype(str).isin(st.session_state[key])]

filtered, prev_df = base, None
sel = st.session_state.get("f_date", ())
if has_dates and isinstance(sel, (tuple, list)) and len(sel) == 2:
    start, end_excl = pd.Timestamp(sel[0]), pd.Timestamp(sel[1]) + pd.Timedelta(days=1)
    filtered = base[(base["Order Date"] >= start) & (base["Order Date"] < end_excl)]
    span = end_excl - start
    prev_df = base[(base["Order Date"] >= start - span) & (base["Order Date"] < start)]
    prev_df = prev_df if not prev_df.empty else None

if chips:
    st.markdown('<div class="chips">' + "".join(f'<span class="chip-f">{c}</span>' for c in chips) + "</div>", unsafe_allow_html=True)
if filtered.empty:
    st.warning("لا توجد سجلات تطابق الفلاتر المختارة. افتح Filters ووسّع النطاق أو اضغط Reset filters.")
    st.stop()

has_od = "Order Date" in filtered and filtered["Order Date"].notna().any()
has_profit = "Profit" in filtered and filtered["Profit"].notna().any()

# =====================================================================
#  DASHBOARD
# =====================================================================
if page == P_DASH:
    cur = kpi_block(filtered)
    prv = kpi_block(prev_df) if prev_df is not None else None
    P = lambda k: prv[k] if prv else None
    cand = [("TOTAL SALES", money(cur["sales"]), delta_html(cur["sales"], P("sales")), "Revenue in selection")]
    if has_profit:
        cand.append(("NET PROFIT", money(cur["profit"]), delta_html(cur["profit"], P("profit")),
                     f"{cur['margin']:.1f}% profit margin" if pd.notna(cur["margin"]) else "Profitability"))
    cand += [("ORDERS", fmt_int(cur["orders"]), delta_html(cur["orders"], P("orders")), "Distinct orders"),
             ("CUSTOMERS", fmt_int(cur["customers"]), delta_html(cur["customers"], P("customers")), "Distinct customers")]
    if "Quantity" in filtered:
        cand.append(("UNITS SOLD", fmt_int(cur["units"]), delta_html(cur["units"], P("units")), "Total quantity"))
    cand.append(("AVG ORDER VALUE", money(cur["aov"]), delta_html(cur["aov"], P("aov")), "Sales per order"))
    for i, (col, (label, value, note, fb)) in enumerate(zip(st.columns(4), cand[:4])):
        with col:
            st.markdown(f'<div class="kpi {"dark" if i == 0 else ""}"><div class="kpi-top"><span class="kpi-label">{label}</span>'
                        f'<span class="kpi-arrow">↗</span></div><div class="kpi-value">{value}</div>'
                        f'<div class="kpi-note">{note or fb}</div></div>', unsafe_allow_html=True)
    st.write("")

    # signals
    insights = []
    if has(filtered, "Category"):
        cs = filtered.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        insights.append(f"<strong>Sales leader:</strong> {cs.index[0]} brings {money(cs.iloc[0])} ({cs.iloc[0] / cs.sum() * 100:.0f}% of sales).")
    if has(filtered, "Sub-Category"):
        ss = filtered.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
        insights.append(f"<strong>Top sub-category:</strong> {ss.index[0]} ({money(ss.iloc[0])}); the top 3 make up {ss.head(3).sum() / ss.sum() * 100:.0f}% of sales.")
    if has(filtered, "Region"):
        rs = filtered.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        insights.append(f"<strong>Strongest region:</strong> {rs.index[0]} ({rs.iloc[0] / rs.sum() * 100:.0f}% of sales); weakest is {rs.index[-1]} ({rs.iloc[-1] / rs.sum() * 100:.0f}%).")
    if has_od:
        mo = filtered.groupby(filtered["Order Date"].dt.month)["Sales"].sum()
        if len(mo) >= 6:
            insights.append(f"<strong>Peak month:</strong> {pd.Timestamp(2000, int(mo.idxmax()), 1):%B} has the highest cumulative sales; {pd.Timestamp(2000, int(mo.idxmin()), 1):%B} the lowest.")
    if has(filtered, "Customer ID"):
        cu = filtered.groupby("Customer ID")["Sales"].sum().sort_values(ascending=False)
        n10 = max(1, int(len(cu) * 0.1))
        insights.append(f"<strong>Customer concentration:</strong> the top 10% of customers generate {cu.head(n10).sum() / cu.sum() * 100:.0f}% of sales.")
    if has_profit:
        loss = (filtered["Profit"] < 0).mean() * 100
        insights.append(f"<strong>Loss-making lines:</strong> {loss:.1f}% of order lines lose money ({money(filtered.loc[filtered['Profit'] < 0, 'Profit'].sum())}).")
    if has(filtered, "Discount", "Profit"):
        cd = filtered[["Discount", "Profit"]].dropna()
        if len(cd) > 2 and cd["Discount"].nunique() > 1:
            insights.append(f"<strong>Discount signal:</strong> correlation with profit is {cd['Discount'].corr(cd['Profit']):.2f} (association, not causation).")

    left, right = st.columns([1.6, 1])
    with left:
        if has_od:
            monthly = filtered.dropna(subset=["Order Date"]).set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
            monthly["3-month avg"] = monthly["Sales"].rolling(3, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["Order Date"], y=monthly["Sales"], name="Monthly sales", mode="lines",
                                     line=dict(color=GREEN, width=2.8), fill="tozeroy", fillcolor="rgba(63,160,107,0.14)"))
            fig.add_trace(go.Scatter(x=monthly["Order Date"], y=monthly["3-month avg"], name="3-month average",
                                     mode="lines", line=dict(color=AMBER, width=2, dash="dash")))
            fig.update_layout(title="Sales analytics: growing or slowing?", legend=dict(orientation="h", y=1.13, x=0))
            fig.update_xaxes(dtick="M3", tickformat="%b\n%Y")
            plot(fig, 400)
        elif has(filtered, "Category"):
            plot(px.bar(filtered.groupby("Category", as_index=False)["Sales"].sum(), x="Category", y="Sales",
                        title="Sales by category", color="Category", color_discrete_sequence=PALETTE), 400)
    with right:
        items = "".join(f'<div class="sig"><span class="dot"></span><div>{i}</div></div>' for i in insights[:6])
        st.markdown(f'<div class="card" style="min-height:420px"><h4>Signals worth investigating</h4>{items}'
                    f'<div style="color:{MUTED};font-size:11px;margin-top:10px">Associations from the current selection, not causal claims.</div></div>',
                    unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if has(filtered, "Category"):
            tmp = filtered.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
            fig = px.pie(tmp, names="Category", values="Sales", hole=.66, title="Which categories drive revenue?", color_discrete_sequence=PALETTE)
            fig.update_traces(textinfo="percent", marker=dict(line=dict(color=PANEL, width=3)))
            fig.update_layout(legend=dict(orientation="h", y=-0.05))
            plot(fig, 340)
    with c2:
        if has_od:
            d = filtered.dropna(subset=["Order Date"])
            dow = d.groupby(d["Order Date"].dt.dayofweek)["Sales"].sum().reindex(range(7), fill_value=0)
            top = int(dow.idxmax())
            fig = go.Figure(go.Bar(
                x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=dow.values,
                marker=dict(color=[GREEN_DK if i == top else "#9DBBA9" for i in range(7)],
                            pattern=dict(shape=["" if i == top else "/" for i in range(7)], bgcolor="#FFFFFF", size=6, solidity=0.35)),
                text=[f"{v / dow.sum() * 100:.0f}%" if i == top else "" for i, v in enumerate(dow.values)], textposition="outside",
                hovertemplate="%{x}: %{y:$,.0f}<extra></extra>"))
            fig.update_layout(title="Which weekday sells the most?", showlegend=False)
            plot(fig, 340)
    with c3:
        if has_od:
            yr = filtered.dropna(subset=["Order Date"]).groupby(filtered["Order Date"].dt.year).agg(
                **({"Sales": ("Sales", "sum")} | ({"Profit": ("Profit", "sum")} if has_profit else {}))).reset_index()
            yr.columns = ["Year"] + list(yr.columns[1:])
            yr["Year"] = yr["Year"].astype(int).astype(str)
            fig = px.bar(yr, x="Year", y=[c for c in ["Sales", "Profit"] if c in yr], barmode="group", title="Year over year",
                         color_discrete_map={"Sales": GREEN_DK, "Profit": GREEN_LT})
            fig.update_layout(legend_title_text="", legend=dict(orientation="h", y=-0.05))
            plot(fig, 340)

    left, right = st.columns([1.6, 1])
    with left:
        if has_od:
            d = filtered.dropna(subset=["Order Date"])
            pv = d.pivot_table(index=d["Order Date"].dt.year, columns=d["Order Date"].dt.month, values="Sales", aggfunc="sum").reindex(columns=range(1, 13))
            pv.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            pv.index = pv.index.astype(int).astype(str)
            plot(px.imshow(pv, aspect="auto", color_continuous_scale=SEQ_GREEN, title="Seasonality: strongest months each year", labels=dict(color="Sales")), 320)
    with right:
        if has(filtered, "Sub-Category"):
            ts = filtered.groupby("Sub-Category", as_index=False)["Sales"].sum().nlargest(6, "Sales").sort_values("Sales")
            hbar(ts, "Sales", "Sub-Category", "Top sub-categories by sales", GREEN_MID, 320)

    if has_profit:
        left, right = st.columns(2)
        with left:
            n = min(len(filtered), 3000)
            samp = filtered.sample(n, random_state=0) if len(filtered) > n else filtered
            fig = px.scatter(samp, x="Sales", y="Profit", color="Category" if "Category" in samp else None,
                             hover_data=[c for c in ["Product Name", "Sub-Category", "Discount", "Region"] if c in samp],
                             title="Do high-sales transactions also deliver profit?", color_discrete_sequence=PALETTE, opacity=.7, render_mode="webgl")
            fig.add_hline(y=0, line_dash="dash", line_color=RED, opacity=.8)
            plot(fig, 360)
            st.caption(f"Showing {n:,} of {len(filtered):,} lines" + (" (random sample for speed)." if n < len(filtered) else "."))
        with right:
            if "Discount Band" in filtered:
                dd = filtered.dropna(subset=["Discount Band"])
                fig = px.box(dd, x="Discount Band", y="Profit", points=False, title="Profit distribution by discount level",
                             color="Discount Band", color_discrete_sequence=PALETTE,
                             category_orders={"Discount Band": ["0%", "≤10%", "10–20%", "20–30%", "30–50%", ">50%"]})
                fig.update_layout(showlegend=False)
                plot(fig, 360)

# =====================================================================
#  PRODUCTS
# =====================================================================
elif page == P_PROD:
    if has(filtered, "Sub-Category"):
        agg = {"Sales": ("Sales", "sum")} | ({"Profit": ("Profit", "sum")} if has_profit else {})
        pp = filtered.groupby("Sub-Category", as_index=False).agg(**agg)
        pp["Share %"] = pp["Sales"] / pp["Sales"].sum() * 100
        if has_profit:
            pp["Margin %"] = np.where(pp["Sales"] != 0, pp["Profit"] / pp["Sales"] * 100, np.nan)
            q = pp.sort_values("Sales")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=q["Sub-Category"], x=q["Sales"], name="Sales", orientation="h", marker_color=GREEN_DK))
            fig.add_trace(go.Bar(y=q["Sub-Category"], x=q["Profit"], name="Profit", orientation="h", marker_color=GREEN_LT))
            fig.update_layout(title="Which sub-categories generate sales — and which generate profit?", barmode="group")
            plot(fig, max(430, len(q) * 27))
            left, right = st.columns(2)
            with left:
                m = pp.sort_values("Margin %")
                fig = px.bar(m, x="Margin %", y="Sub-Category", orientation="h", color="Margin %", color_continuous_scale=DIVERGING,
                             color_continuous_midpoint=0, title="Profit margin by sub-category")
                fig.update_layout(coloraxis_showscale=False)
                plot(fig, max(380, len(m) * 26))
            with right:
                fig = px.scatter(pp, x="Sales", y="Profit", text="Sub-Category", size=pp["Sales"].abs(), color="Margin %",
                                 color_continuous_scale=DIVERGING, color_continuous_midpoint=0, title="Portfolio map: scale vs. profitability")
                fig.add_hline(y=0, line_dash="dash", line_color=RED)
                fig.update_traces(textposition="top center", textfont=dict(size=10))
                plot(fig, max(380, len(m) * 26))
        else:
            left, right = st.columns([1.15, 1])
            with left:
                hbar(pp.sort_values("Sales"), "Sales", "Sub-Category", "Sales by sub-category", GREEN_MID, max(430, len(pp) * 27),
                     hover_data={"Share %": ":.1f"})
            with right:
                pa = pp.sort_values("Sales", ascending=False).reset_index(drop=True)
                pa["Cumulative %"] = pa["Sales"].cumsum() / pa["Sales"].sum() * 100
                fig = go.Figure()
                fig.add_trace(go.Bar(x=pa["Sub-Category"], y=pa["Sales"], name="Sales", marker_color=GREEN_MID))
                fig.add_trace(go.Scatter(x=pa["Sub-Category"], y=pa["Cumulative %"], name="Cumulative share", mode="lines+markers",
                                         line=dict(color=AMBER, width=3), yaxis="y2"))
                fig.update_layout(title="Pareto: how concentrated is revenue?", legend=dict(orientation="h", y=1.12, x=0),
                                  yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False, ticksuffix="%"))
                fig.update_xaxes(tickangle=-45)
                plot(fig, max(430, len(pp) * 27))
    if has(filtered, "Product Name"):
        prod = filtered.groupby("Product Name", as_index=False).agg(Sales=("Sales", "sum"), **({"Profit": ("Profit", "sum")} if has_profit else {}))
        prod["Label"] = prod["Product Name"].map(short)
        key = "Profit" if has_profit else "Sales"
        left, right = st.columns(2)
        with left:
            hbar(prod.nlargest(10, key).sort_values(key), key, "Label", f"Top 10 products by {key.lower()}", GREEN_MID, 400,
                 hover_data={"Product Name": True, "Label": False})
        with right:
            hbar(prod.nsmallest(10, key).sort_values(key, ascending=False), key, "Label",
                 f"Bottom 10 products by {key.lower()}", RED if has_profit else GREEN_LT, 400, hover_data={"Product Name": True, "Label": False})
    if has(filtered, "Sub-Category"):
        st.markdown("#### Sub-category detail")
        fmt = {"Sales": "${:,.2f}", "Share %": "{:.1f}%"} | ({"Profit": "${:,.2f}", "Margin %": "{:.1f}%"} if has_profit else {})
        table(pp.sort_values("Profit" if has_profit else "Sales", ascending=has_profit).style.format(fmt))
    else:
        st.info("This page requires a Sub-Category column.")

# =====================================================================
#  CUSTOMERS & REGIONS
# =====================================================================
elif page == P_CUST:
    mcols = ["Sales"] + (["Profit"] if has_profit else [])
    left, right = st.columns(2)
    with left:
        if "Region" in filtered:
            reg = filtered.groupby("Region", as_index=False)[mcols].sum()
            fig = px.bar(reg, x="Region", y=mcols, barmode="group", title="Which regions contribute most?",
                         color_discrete_map={"Sales": GREEN_DK, "Profit": GREEN_LT})
            fig.update_layout(legend_title_text="")
            plot(fig, 370)
    with right:
        if "Segment" in filtered:
            seg = filtered.groupby("Segment", as_index=False)[mcols].sum()
            fig = px.bar(seg, x="Segment", y=mcols, barmode="group", title="Which customer segment is most valuable?",
                         color_discrete_map={"Sales": GREEN_DK, "Profit": GREEN_LT})
            fig.update_layout(legend_title_text="")
            plot(fig, 370)
    if has(filtered, "Region", "Category"):
        if has_profit:
            g = filtered.groupby(["Region", "Category"]).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
            g["v"] = np.where(g["Sales"] != 0, g["Profit"] / g["Sales"] * 100, np.nan)
            fig = px.imshow(g.pivot(index="Region", columns="Category", values="v"), text_auto=".1f", aspect="auto",
                            color_continuous_scale=DIVERGING, color_continuous_midpoint=0, title="Profit margin (%) by region and category")
        else:
            g = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
            fig = px.imshow(g.pivot(index="Region", columns="Category", values="Sales") / 1000, text_auto=".0f", aspect="auto",
                            color_continuous_scale=SEQ_GREEN, title="Sales ($K) by region and category", labels=dict(color="$K"))
        plot(fig, 330)
    if "State" in filtered:
        stt = filtered.groupby("State", as_index=False)[mcols].sum()
        key = "Profit" if has_profit else "Sales"
        left, right = st.columns(2)
        with left:
            hbar(stt.nlargest(10, key).sort_values(key), key, "State", f"Top 10 states by {key.lower()}", GREEN_MID, 380)
        with right:
            if has_profit:
                hbar(stt.nsmallest(10, key).sort_values(key, ascending=False), key, "State", "Bottom 10 states by profit", RED, 380)
            elif "City" in filtered:
                hbar(filtered.groupby("City", as_index=False)["Sales"].sum().nlargest(10, "Sales").sort_values("Sales"),
                     "Sales", "City", "Top 10 cities by sales", GREEN_DK, 380)
    if has(filtered, "Customer ID"):
        cust = filtered.groupby("Customer ID", as_index=False)[mcols].sum()
        if "Customer Name" in filtered:
            nm = filtered.drop_duplicates("Customer ID").set_index("Customer ID")["Customer Name"]
            cust["Customer"] = cust["Customer ID"].map(nm).fillna(cust["Customer ID"])
        else:
            cust["Customer"] = cust["Customer ID"]
        cust = cust.nlargest(15, "Sales").sort_values("Sales")
        if has_profit:
            fig = px.bar(cust, x="Sales", y="Customer", orientation="h", color="Profit", color_continuous_scale=DIVERGING,
                         color_continuous_midpoint=0, title="Top 15 customers by sales — and are they profitable?")
            plot(fig, 480)
        else:
            hbar(cust, "Sales", "Customer", "Top 15 customers by sales", GREEN_MID, 480)
    if has(filtered, "Segment", "Order ID"):
        o = filtered.groupby("Segment").agg(Orders=("Order ID", "nunique")).reset_index()
        fig = px.pie(o, names="Segment", values="Orders", hole=.65, title="Share of distinct orders by customer segment", color_discrete_sequence=PALETTE)
        fig.update_traces(textinfo="label+percent", marker=dict(line=dict(color=PANEL, width=3)))
        plot(fig, 330)

# =====================================================================
#  OPERATIONS
# =====================================================================
elif page == P_OPS:
    left, right = st.columns(2)
    with left:
        if has(filtered, "Ship Mode"):
            agg = {"Sales": ("Sales", "sum")} | ({"Avg days": ("Shipping Days", "mean")} if "Shipping Days" in filtered else {})
            ship = filtered.groupby("Ship Mode", as_index=False).agg(**agg)
            fig = px.bar(ship.sort_values("Sales"), x="Sales", y="Ship Mode", orientation="h", title="Which shipping modes carry the most sales?",
                         color="Avg days" if "Avg days" in ship else "Sales",
                         color_continuous_scale=[GREEN_LT, AMBER, RED] if "Avg days" in ship else [GREEN_XLT, GREEN_DK])
            plot(fig, 360)
            if "Avg days" in ship:
                st.caption("Bar color shows the average shipping time (days) for each mode.")
    with right:
        if "Shipping Days" in filtered:
            sd = filtered["Shipping Days"].dropna()
            neg = int((sd < 0).sum())
            sd = sd[sd >= 0]
            if not sd.empty:
                hist = sd.value_counts().sort_index().rename_axis("Shipping Days").reset_index(name="Order lines")
                plot(px.bar(hist, x="Shipping Days", y="Order lines", title="How long does shipping take?", color_discrete_sequence=[GREEN_MID]), 360)
                st.caption(f"Median shipping time: {sd.median():.1f} days." + (f" {neg:,} rows with negative durations were excluded." if neg else ""))
    if has(filtered, "Ship Mode", "Order ID") and "Shipping Days" in filtered:
        sm = filtered.groupby("Ship Mode").agg(Orders=("Order ID", "nunique"), Sales=("Sales", "sum"),
                                              **{"Avg days": ("Shipping Days", "mean")}).reset_index()
        table(sm.style.format({"Orders": "{:,.0f}", "Sales": "${:,.0f}", "Avg days": "{:.2f}"}))
    if has_profit and has(filtered, "Discount Band"):
        db = filtered.groupby("Discount Band", observed=True).agg(
            Lines=("Profit", "size"), Sales=("Sales", "sum"), Profit=("Profit", "sum"),
            Loss_Share=("Profit", lambda s: (s < 0).mean() * 100)).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=db["Discount Band"].astype(str), y=db["Profit"], name="Total profit",
                             marker_color=[GREEN_MID if v >= 0 else RED for v in db["Profit"]]))
        fig.add_trace(go.Scatter(x=db["Discount Band"].astype(str), y=db["Loss_Share"], name="% of lines losing money",
                                 mode="lines+markers", line=dict(color=AMBER, width=3), yaxis="y2"))
        fig.update_layout(title="Discount bands: total profit vs. share of loss-making lines",
                          yaxis2=dict(title="% loss-making lines", overlaying="y", side="right", showgrid=False, range=[0, 105]),
                          legend=dict(orientation="h", y=1.12, x=0))
        plot(fig, 400)
        st.caption("Diagnostic, not a recommendation to remove discounts. Check product, segment and region mix before acting.")
    elif "Discount" not in filtered:
        st.info("Discount and profit analysis needs `Discount` and `Profit` columns, which this dataset doesn't include.")

# =====================================================================
#  FORECAST
# =====================================================================
elif page == P_FC:
    fcm = None
    if fc_rows is not None:
        fcm = fc_rows.assign(Month=pd.to_datetime(fc_rows["Order Date"], errors="coerce"))
        for c in ["Forecast", "Reference", "Low", "High"]:
            fcm[c] = pd.to_numeric(fcm[c], errors="coerce") if c in fcm else np.nan
        fcm["Period"] = fcm["Period"].astype(str).str.strip().str.title() if "Period" in fcm else ""
        fcm = fcm.dropna(subset=["Month"]).sort_values("Month")
    if fcm is None or fcm[fcm["Period"] == "Future"].empty:
        st.info("هذا الملف لا يحتوي على صفوف تنبؤ. شغّل القسم 17 في الـ Notebook؛ فهو يكتب `Superstore_Cleaned.csv` "
                "(البيانات النظيفة + التنبؤ في ملف واحد). ضعه بجانب `app.py` أو ارفعه من زر Import data.")
    else:
        actual = df.dropna(subset=["Order Date"]).set_index("Order Date")["Sales"].resample("MS").sum() if has_dates else pd.Series(dtype=float)
        test = fcm[(fcm["Period"] == "Test") & fcm["Forecast"].notna()].copy()
        test["Actual"] = actual.reindex(test["Month"]).values
        test = test.dropna(subset=["Actual"])
        fut = fcm[fcm["Period"] == "Future"].reset_index(drop=True)
        st.caption("Gradient Boosting forecast saved by the notebook in the same file as the cleaned data. It covers **all** regions, "
                   "categories and segments, so the filters above do not change it.")
        if chips:
            st.info("الفلاتر الحالية لا تؤثر على هذه الصفحة: التنبؤ المحفوظ يغطي كل المناطق والفئات والشرائح.")
        if has_dates and not test.empty and test["Month"].max().to_period("M") != df["Order Date"].max().to_period("M"):
            st.warning(f"التنبؤ المحفوظ مبني على بيانات حتى {test['Month'].max():%b %Y} بينما بياناتك تنتهي عند "
                       f"{df['Order Date'].max():%b %Y}. أعد تشغيل القسم 17 لتحديثه.")
        horizon = len(fut)
        if len(fut) > 3:
            horizon = st.slider("Months to show", 3, len(fut), len(fut))
        f = fut.head(horizon)
        last_same = actual.reindex(f["Month"] - pd.DateOffset(months=12)).sum()   # same calendar months, one year earlier
        growth = (f["Forecast"].sum() / last_same - 1) * 100 if last_same else np.nan
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Forecast sales · next {horizon} mo", money(f["Forecast"].sum()))
        k2.metric("vs same months last year", f"{growth:+.1f}%" if pd.notna(growth) and last_same else "—")
        k3.metric("Expected peak month", f"{f.loc[f['Forecast'].idxmax(), 'Month']:%b %Y}")
        k4.metric("80% range (total)", f"{money(f['Low'].sum())} – {money(f['High'].sum())}" if f["Low"].notna().all() else "—")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual.index, y=actual.values, name="History", mode="lines", line=dict(color=GREEN, width=2.8)))
        if f["Low"].notna().all() and f["High"].notna().all():
            fig.add_trace(go.Scatter(x=list(f["Month"]) + list(f["Month"][::-1]), y=list(f["High"]) + list(f["Low"][::-1]), fill="toself",
                                     fillcolor="rgba(232,163,61,0.20)", line=dict(width=0), name="80% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[actual.index[-1]] + list(f["Month"]), y=[actual.values[-1]] + list(f["Forecast"]), name="Forecast",
                                 mode="lines+markers", line=dict(color=AMBER, width=2.8, dash="dash")))
        fig.update_layout(title="Monthly sales: history and forecast", legend=dict(orientation="h", y=1.12, x=0))
        plot(fig, 400)

        out_tbl = pd.DataFrame({"Month": f["Month"].dt.strftime("%b %Y"), "Forecast": f["Forecast"].round(0),
                                "Low (10%)": f["Low"].round(0), "High (90%)": f["High"].round(0)})
        left, right = st.columns([1, 1.4])
        with left:
            table(out_tbl.style.format({"Forecast": "${:,.0f}", "Low (10%)": "${:,.0f}", "High (90%)": "${:,.0f}"}))
            st.download_button("Download forecast as CSV", out_tbl.to_csv(index=False).encode("utf-8-sig"),
                               file_name="sales_forecast.csv", mime="text/csv")
        with right:
            if not test.empty:
                wape = lambda p: float(np.abs(test["Actual"] - p).sum() / test["Actual"].sum() * 100)
                b1, b2 = st.columns(2)
                b1.metric("Holdout error · Gradient Boosting", f"{wape(test['Forecast']):.1f}%")
                if test["Reference"].notna().all():
                    b2.metric("Reference · same month last year", f"{wape(test['Reference']):.1f}%", help="A no-model baseline. The model should beat it.")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test["Month"], y=test["Actual"], name="Actual", line=dict(color=TEXT, width=2.5)))
                fig.add_trace(go.Scatter(x=test["Month"], y=test["Forecast"], name="Gradient Boosting", line=dict(color=AMBER)))
                if test["Reference"].notna().all():
                    fig.add_trace(go.Scatter(x=test["Month"], y=test["Reference"], name="Same month last year", line=dict(color=GREEN_MID, dash="dot")))
                fig.update_layout(title="How accurate was it on the last 12 months?", legend=dict(orientation="h", y=1.15, x=0))
                plot(fig, 300)
        st.caption("Error = weighted absolute % error on the last 12 known months (model trained on earlier data only). "
                   "The 80% range is built from the model's own past errors, so it is wide on purpose.")

# =====================================================================
#  DATA QUALITY
# =====================================================================
else:
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rows", f"{len(df):,}")
    q2.metric("Columns", f"{len(df.columns):,}")
    q3.metric("Exact duplicate rows", f"{df.duplicated().sum():,}")
    q4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    checks = []
    if "Row ID" in df:
        checks.append(("Duplicate Row ID values", int(df["Row ID"].duplicated().sum())))
    if "Order Date" in df:
        checks.append(("Unparsed / missing Order Date", int(df["Order Date"].isna().sum())))
    if "Shipping Days" in df:
        checks.append(("Ship date before order date", int((df["Shipping Days"] < 0).sum())))
    checks.append(("Sales ≤ 0", int((df["Sales"] <= 0).sum())))
    if "Discount" in df:
        checks.append(("Discount outside 0–1", int(((df["Discount"] < 0) | (df["Discount"] > 1)).sum())))
    if "Quantity" in df:
        checks.append(("Quantity ≤ 0", int((df["Quantity"] <= 0).sum())))
    chk = pd.DataFrame(checks, columns=["Check", "Rows affected"])
    chk["Status"] = np.where(chk["Rows affected"] == 0, "✅ OK", "⚠️ Review")
    table(chk)
    miss = df.isna().sum().rename("Missing values").to_frame()
    miss["Missing %"] = (df.isna().mean() * 100).round(2)
    miss = miss[miss["Missing values"] > 0].sort_values("Missing values", ascending=False)
    if not miss.empty:
        fig = px.bar(miss.reset_index(names="Column").head(20), x="Missing values", y="Column", orientation="h",
                     title="Which columns need data-quality attention?", color="Missing %", color_continuous_scale=[GREEN_LT, AMBER, RED])
        plot(fig, max(320, min(600, 30 * len(miss.head(20)))))
    else:
        st.success("No missing values detected in the loaded dataset.")
    st.markdown("#### Preview filtered records")
    st.dataframe(filtered.head(500), hide_index=True, **STRETCH)
    st.download_button("Download filtered data as CSV", data=filtered.to_csv(index=False).encode("utf-8-sig"),
                       file_name="superstore_filtered.csv", mime="text/csv")

st.markdown("---")
st.markdown(f'<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:{MUTED};font-size:11px">'
            f'<span>◍ InsightLab · Analytical learning workspace</span>'
            f'<span>{len(filtered):,} rows in current view · {len(df.columns)} fields</span></div>', unsafe_allow_html=True)
