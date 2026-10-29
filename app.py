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

st.set_page_config(
    page_title="InsightLab | Superstore Analytics",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual identity ----------
BG, PANEL, PANEL_2 = "#0B1020", "#121A2E", "#17213A"
TEXT, MUTED = "#F2F5FF", "#98A6C7"
TEAL, BLUE, PURPLE, ORANGE, RED = "#45E0C1", "#6D9EFF", "#B59CFF", "#FFB86B", "#FF718B"
GRID = "#26324D"
PALETTE = [TEAL, BLUE, PURPLE, ORANGE, RED, "#7BD3FF", "#D6E685", "#F28BCB"]
DIVERGING = [RED, "#2A3555", TEAL]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {{ color-scheme: dark; }}
html, body, .stApp, .stMarkdown, p, label, input, textarea, button {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background: radial-gradient(ellipse at 12% 0%, #17254a 0%, {BG} 44%); color: {TEXT}; }}
[data-testid="stSidebar"] {{ background: #0D1426; border-right: 1px solid #24304D; }}
[data-testid="stHeader"] {{ background: rgba(11,16,32,.82); }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }}
.hero {{ padding: 28px 32px; border: 1px solid #29395D; border-radius: 24px;
background: linear-gradient(115deg, rgba(29,48,87,.95), rgba(18,26,46,.95) 58%, rgba(40,31,78,.85));
margin-bottom: 22px; position: relative; overflow: hidden; }}
.hero:after {{ content:'✦'; position:absolute; right:35px; top:-22px; font-size:150px; color:rgba(181,156,255,.10); }}
.eyebrow {{ color:{TEAL}; letter-spacing:2px; font-size:11px; font-weight:700; text-transform:uppercase; }}
.hero h1 {{ font-family:'Space Grotesk',sans-serif; color:{TEXT}; font-size:35px; line-height:1.15; margin:10px 0 8px; }}
.hero p {{ color:{MUTED}; font-size:14px; margin:0; max-width:780px; }}
.section-title {{ font-family:'Space Grotesk',sans-serif; font-size:21px; font-weight:700; margin:26px 0 3px; color:{TEXT}; }}
.section-sub {{ color:{MUTED}; font-size:12px; margin-bottom:14px; }}
.kpi {{ background:linear-gradient(145deg, {PANEL_2}, {PANEL}); border:1px solid #273553; border-radius:18px; padding:18px 19px; min-height:120px; }}
.kpi-label {{ color:{MUTED}; font-size:12px; font-weight:600; }}
.kpi-value {{ font-family:'Space Grotesk',sans-serif; color:{TEXT}; font-size:28px; font-weight:700; margin-top:8px; letter-spacing:-.6px; }}
.kpi-note {{ color:{MUTED}; font-size:11px; margin-top:5px; }}
.kpi-note .up {{ color:{TEAL}; font-weight:700; }}
.kpi-note .down {{ color:{RED}; font-weight:700; }}
.insight {{ background:linear-gradient(120deg, rgba(69,224,193,.09), rgba(109,158,255,.06)); border:1px solid #28445A; border-radius:15px; padding:15px 17px; margin:7px 0; }}
.insight strong {{ color:{TEAL}; }}
div[data-testid="stPlotlyChart"] {{ background:{PANEL}; border:1px solid #24314E; border-radius:16px; padding:9px 8px 0; }}
.stTabs [data-baseweb="tab-list"] {{ gap:8px; }}
.stTabs [data-baseweb="tab"] {{ background:{PANEL}; border-radius:10px 10px 0 0; padding:10px 16px; }}
.stTabs [aria-selected="true"] {{ border-bottom:2px solid {TEAL}; }}
div[data-testid="stMetric"] {{ background:{PANEL}; border:1px solid #273553; padding:14px; border-radius:14px; }}
.stButton button {{ border-radius:10px; }}
hr {{ border-color:#273553; }}
footer {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(family="DM Sans", color=TEXT, size=12),
    margin=dict(l=18, r=18, t=50, b=18), title_font=dict(size=15, color=TEXT),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
)

# Streamlit renamed use_container_width -> width="stretch"; support both.
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
    st.plotly_chart(fig, **STRETCH)


def table(data):
    st.dataframe(data, hide_index=True, **STRETCH)


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
    """Strip spaces and map variants (order_date, Sub Category, ...) to canonical names."""
    df.columns = [CANON_MAP.get(_key(c), str(c).strip()) for c in df.columns]
    return df


def smart_to_datetime(series):
    """Parse dates and detect dd/mm/yyyy vs mm/dd/yyyy from the data instead of guessing."""
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
    if "Order Date" in df:
        df["Year"] = df["Order Date"].dt.year
    if has(df, "Order Date", "Ship Date"):
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    if "Discount" in df:
        df["Discount Band"] = pd.cut(
            df["Discount"], bins=[-0.001, 0, 0.10, 0.20, 0.30, 0.50, 1.0],
            labels=["0%", "≤10%", "10–20%", "20–30%", "30–50%", ">50%"],
        )
    return df


@st.cache_data(show_spinner="Reading file…")
def read_bytes(name, data):
    bio = io.BytesIO(data)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
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
    return dict(
        sales=sales, profit=profit,
        orders=d["Order ID"].nunique() if "Order ID" in d else len(d),
        customers=d["Customer ID"].nunique() if "Customer ID" in d else np.nan,
        units=d["Quantity"].sum() if "Quantity" in d else np.nan,
        margin=(profit / sales * 100) if pd.notna(profit) and pd.notna(sales) and sales != 0 else np.nan,
    )


def delta_html(cur, prev, pp=False):
    if prev is None or pd.isna(cur) or pd.isna(prev):
        return None
    if pp:
        diff, txt = cur - prev, f"{cur - prev:+.1f} pp"
    else:
        if prev == 0:
            return None
        diff = cur - prev
        txt = f"{diff / abs(prev) * 100:+.1f}%"
    cls, arrow = ("up", "▲") if diff >= 0 else ("down", "▼")
    return f'<span class="{cls}">{arrow} {txt}</span> vs previous period'


@st.cache_data(show_spinner="Fitting forecast model…")
def run_forecast(series, horizon):
    """Holt-Winters (additive trend + seasonality) with a simulated 80% range and a 12-month holdout check."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    def fit(s):
        return ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=12,
                                    initialization_method="estimated").fit()

    out = {}
    if len(series) >= 36:
        tr, te = series[:-12], series[-12:]
        hw, sn = fit(tr).forecast(12).values, tr[-12:].values
        wape = lambda p: float(np.abs(te.values - p).sum() / te.values.sum() * 100)
        out["backtest"] = dict(test=te, hw=hw, snaive=sn, wape_hw=wape(hw), wape_sn=wape(sn))
    model = fit(series)
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    sims = model.simulate(horizon, repetitions=1000, error="add", random_state=42)
    out.update(point=pd.Series(model.forecast(horizon).values, index=idx),
               lower=np.maximum(sims.quantile(0.10, axis=1).values, 0),
               upper=sims.quantile(0.90, axis=1).values)
    return out


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ✦ InsightLab")
    st.caption("AI-inspired • Learning analytics")
    st.markdown("---")
    st.markdown("### Data source")
    uploaded = st.file_uploader("Upload Superstore data", type=["csv", "xlsx", "xls"])
    st.caption("Expected fields include Sales, Profit, Order Date, Category, Region and Segment.")
    st.markdown("---")
    st.markdown("### About this dashboard")
    st.markdown("Explore performance, discover patterns, and turn retail data into evidence-based decisions.")
    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · Pandas")

raw = load_raw(uploaded)
if raw is None:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">DATA INTELLIGENCE / LEARNING STUDIO</div>
      <h1>InsightLab</h1>
      <p>A polished, interactive analytics workspace for exploring Superstore sales, profitability, customer segments and operational patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("ابدأ برفع ملف البيانات من الشريط الجانبي. يدعم التطبيق ملفات CSV وExcel، ويمكنه أيضًا قراءة `Superstore_Cleaned.csv` أو `Superstore.csv` إذا كان موجودًا بجانب `app.py`.")
    st.markdown("### ما الأسئلة التي سيجيب عنها التحليل؟")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**01 · Growth**")
        st.write("كيف تتغير المبيعات شهريًا؟ وهل توجد مواسم أو فترات ذروة؟")
    with c2:
        st.markdown("**02 · Profitability**")
        st.write("ما الفئات والمناطق التي تحقق الربح فعلًا، وليس المبيعات فقط؟")
    with c3:
        st.markdown("**03 · Decisions**")
        st.write("هل ترتبط الخصومات بانخفاض الربحية؟ وما المنتجات التي تستحق المراجعة؟")
    st.stop()

df = prepare_data(raw)
if df.empty:
    st.error("الملف لا يحتوي على صفوف بيانات.")
    st.stop()
if "Sales" not in df.columns:
    st.error("لم أجد عمود `Sales` في الملف. الأعمدة الموجودة: " + ", ".join(map(str, df.columns)))
    st.stop()

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ DATA INTELLIGENCE / LEARNING STUDIO</div>
  <h1>Superstore Analytics</h1>
  <p>From raw transactions to meaningful decisions. Explore sales momentum, profit drivers, customer behavior and operational efficiency through interactive visual analysis.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Filters ----------
f1, f2, f3, f4 = st.columns([1.25, 1, 1, 1])
has_dates = "Order Date" in df and df["Order Date"].notna().any()
with f2:
    region_opts = sorted(df["Region"].dropna().astype(str).unique()) if "Region" in df else []
    regions = st.multiselect("Region", region_opts, default=region_opts)
with f3:
    cat_opts = sorted(df["Category"].dropna().astype(str).unique()) if "Category" in df else []
    cats = st.multiselect("Category", cat_opts, default=cat_opts)
with f4:
    seg_opts = sorted(df["Segment"].dropna().astype(str).unique()) if "Segment" in df else []
    segs = st.multiselect("Customer segment", seg_opts, default=seg_opts)

base = df
if region_opts:
    base = base[base["Region"].astype(str).isin(regions)]
if cat_opts:
    base = base[base["Category"].astype(str).isin(cats)]
if seg_opts:
    base = base[base["Segment"].astype(str).isin(segs)]

filtered, prev_df = base, None
if has_dates:
    min_date, max_date = df["Order Date"].min().date(), df["Order Date"].max().date()
    with f1:
        date_range = st.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start = pd.Timestamp(date_range[0])
        end_excl = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        filtered = base[(base["Order Date"] >= start) & (base["Order Date"] < end_excl)]
        span = end_excl - start
        prev_df = base[(base["Order Date"] >= start - span) & (base["Order Date"] < start)]
        if prev_df.empty:
            prev_df = None
    else:
        f1.caption("Select both a start and an end date.")
else:
    f1.caption("Order dates unavailable")

if filtered.empty:
    st.warning("لا توجد سجلات تطابق الفلاتر المختارة. وسّع نطاق التاريخ أو أعد اختيار الفئات.")
    st.stop()

# ---------- KPI strip ----------
cur = kpi_block(filtered)
prv = kpi_block(prev_df) if prev_df is not None else None
avg_order = cur["sales"] / cur["orders"] if cur["orders"] else np.nan

st.markdown('<div class="section-title">Executive snapshot</div><div class="section-sub">Headline metrics for the selected data. Deltas compare with the immediately preceding period of equal length (when data exists).</div>', unsafe_allow_html=True)
kpis = [
    ("TOTAL SALES", money(cur["sales"]), delta_html(cur["sales"], prv and prv["sales"]) or "Revenue across selected records"),
    ("NET PROFIT", money(cur["profit"]),
     (delta_html(cur["profit"], prv and prv["profit"]) or "") + (f"<br>{cur['margin']:.1f}% profit margin" if pd.notna(cur["margin"]) else "")
     or "Profitability indicator"),
    ("ORDERS", fmt_int(cur["orders"]), delta_html(cur["orders"], prv and prv["orders"]) or (f"{money(avg_order)} average sales / order" if pd.notna(avg_order) else "Distinct order count")),
    ("CUSTOMERS", fmt_int(cur["customers"]), delta_html(cur["customers"], prv and prv["customers"]) or "Distinct customer IDs"),
    ("UNITS SOLD", fmt_int(cur["units"]), delta_html(cur["units"], prv and prv["units"]) or "Total quantity"),
]
for col, (label, value, note) in zip(st.columns(5), kpis):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

# ---------- Insights ----------
st.markdown('<div class="section-title">Signals worth investigating</div><div class="section-sub">Short, data-driven observations from the current selection — associations, not causal claims.</div>', unsafe_allow_html=True)
insights = []
if has(filtered, "Category", "Sales"):
    cs = filtered.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    if len(cs):
        insights.append(f"<strong>Sales leader:</strong> {cs.index[0]} contributes the most sales ({money(cs.iloc[0])}, {cs.iloc[0] / cs.sum() * 100:.0f}% of the total).")
if has(filtered, "Sub-Category", "Profit"):
    ps = filtered.groupby("Sub-Category")["Profit"].sum().sort_values()
    if len(ps):
        note = "investigate pricing, discounting and product mix." if ps.iloc[0] < 0 else "all sub-categories are profitable in this selection."
        insights.append(f"<strong>Profit watchlist:</strong> {ps.index[0]} has the lowest total profit ({money(ps.iloc[0])}); {note}")
if has(filtered, "Profit"):
    loss_share = (filtered["Profit"] < 0).mean() * 100
    loss_amt = filtered.loc[filtered["Profit"] < 0, "Profit"].sum()
    insights.append(f"<strong>Loss-making lines:</strong> {loss_share:.1f}% of order lines lose money, totalling {money(loss_amt)}.")
if has(filtered, "Discount", "Profit"):
    cd = filtered[["Discount", "Profit"]].dropna()
    if len(cd) > 2 and cd["Discount"].nunique() > 1 and cd["Profit"].nunique() > 1:
        corr = cd["Discount"].corr(cd["Profit"])
        direction = "negative" if corr < -0.15 else "positive" if corr > 0.15 else "weak"
        insights.append(f"<strong>Discount signal:</strong> discount and profit have a {direction} linear correlation ({corr:.2f}). This is an association, not proof of causation.")
if has(filtered, "Discount Band", "Profit"):
    bands = filtered.groupby("Discount Band", observed=True)["Profit"].agg(["sum", "size"])
    bands = bands[bands["size"] >= 30]
    neg = bands[bands["sum"] < 0]
    if len(neg):
        insights.append(f"<strong>Discount threshold:</strong> total profit turns negative from the <em>{neg.index[0]}</em> discount band upward in this selection.")
if has(filtered, "Region", "Profit"):
    rp = filtered.groupby("Region")["Profit"].sum().sort_values()
    if len(rp):
        insights.append(f"<strong>Regional opportunity:</strong> {rp.index[0]} has the lowest total profit ({money(rp.iloc[0])}).")
if has(filtered, "Order Date") and filtered["Order Date"].notna().any():
    m = filtered.groupby(filtered["Order Date"].dt.month)["Sales"].sum()
    if len(m) >= 6:
        mn = pd.Timestamp(2000, int(m.idxmax()), 1).strftime("%B")
        insights.append(f"<strong>Peak month:</strong> {mn} has the highest cumulative sales across the selected years.")
for item in insights or ["Upload a dataset with standard Superstore fields to unlock automated business signals."]:
    st.markdown(f'<div class="insight">{item}</div>', unsafe_allow_html=True)

# ---------- Charts ----------
st.markdown('<div class="section-title">Explore the business</div><div class="section-sub">Each chart answers a practical question. Hover for details; use the filters above to investigate a subset.</div>', unsafe_allow_html=True)
tab_overview, tab_products, tab_customers, tab_operations, tab_forecast, tab_data = st.tabs(
    ["✦ Overview", "▦ Products & Profit", "◎ Customers & Regions", "↗ Operations", "🔮 Forecast", "⌘ Data quality"]
)

# ===== Overview =====
with tab_overview:
    left, right = st.columns([1.45, 1])
    with left:
        if filtered["Order Date"].notna().any() if "Order Date" in filtered else False:
            monthly = filtered.dropna(subset=["Order Date"]).set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
            monthly["3-month avg"] = monthly["Sales"].rolling(3, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["Order Date"], y=monthly["Sales"], name="Monthly sales", mode="lines",
                                     line=dict(color=TEAL, width=2.5), fill="tozeroy", fillcolor="rgba(69,224,193,0.13)"))
            fig.add_trace(go.Scatter(x=monthly["Order Date"], y=monthly["3-month avg"], name="3-month average",
                                     mode="lines", line=dict(color=ORANGE, width=2, dash="dash")))
            fig.update_layout(title="Are sales growing or slowing over time?", legend=dict(orientation="h", y=1.12, x=0))
            fig.update_xaxes(dtick="M3", tickformat="%b\n%Y")
            plot(fig, 370)
            st.caption("The first and last months may be partial depending on the selected dates.")
        else:
            tmp = filtered.groupby("Category", as_index=False)["Sales"].sum() if "Category" in filtered else None
            if tmp is not None:
                plot(px.bar(tmp, x="Category", y="Sales", title="Sales by category", color="Category", color_discrete_sequence=PALETTE))
            else:
                st.info("Add Order Date and Sales columns to view sales trends.")
    with right:
        if has(filtered, "Category", "Sales"):
            tmp = filtered.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
            fig = px.pie(tmp, names="Category", values="Sales", hole=.68, title="Which categories drive revenue?", color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color=PANEL, width=3)))
            fig.update_layout(showlegend=False)
            plot(fig, 370)
        elif has(filtered, "Region", "Sales"):
            tmp = filtered.groupby("Region", as_index=False)["Sales"].sum()
            plot(px.bar(tmp, x="Region", y="Sales", title="Sales by region", color="Region", color_discrete_sequence=PALETTE))

    if "Order Date" in filtered and filtered["Order Date"].notna().any():
        left, right = st.columns([1.45, 1])
        with left:
            d = filtered.dropna(subset=["Order Date"])
            pv = d.pivot_table(index=d["Order Date"].dt.year, columns=d["Order Date"].dt.month, values="Sales", aggfunc="sum").reindex(columns=range(1, 13))
            pv.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            pv.index = pv.index.astype(int).astype(str)
            fig = px.imshow(pv, aspect="auto", color_continuous_scale=["#17213A", BLUE, TEAL],
                            title="Seasonality: which months are strongest each year?", labels=dict(color="Sales"))
            plot(fig, 300)
        with right:
            if has(filtered, "Profit"):
                yr = d.groupby(d["Order Date"].dt.year).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
                yr.columns = ["Year", "Sales", "Profit"]
                yr["Year"] = yr["Year"].astype(int).astype(str)
                fig = px.bar(yr, x="Year", y=["Sales", "Profit"], barmode="group", title="Year over year",
                             color_discrete_map={"Sales": BLUE, "Profit": TEAL})
                fig.update_layout(legend_title_text="")
                plot(fig, 300)

    left, right = st.columns(2)
    with left:
        if has(filtered, "Sales", "Profit"):
            n = min(len(filtered), 3000)
            samp = filtered.sample(n, random_state=0) if len(filtered) > n else filtered
            fig = px.scatter(samp, x="Sales", y="Profit", color="Category" if "Category" in samp else None,
                             hover_data=[c for c in ["Product Name", "Sub-Category", "Discount", "Region"] if c in samp],
                             title="Do high-sales transactions also deliver profit?",
                             color_discrete_sequence=PALETTE, opacity=.7, render_mode="webgl")
            fig.add_hline(y=0, line_dash="dash", line_color=RED, opacity=.8)
            plot(fig, 360)
            st.caption(f"Points below zero are loss-making lines. Showing {n:,} of {len(filtered):,} lines" + (" (random sample for speed)." if n < len(filtered) else "."))
    with right:
        if has(filtered, "Discount Band", "Profit"):
            dd = filtered.dropna(subset=["Discount Band"])
            fig = px.box(dd, x="Discount Band", y="Profit", points=False, title="How does profitability vary by discount level?",
                         color="Discount Band", color_discrete_sequence=PALETTE,
                         category_orders={"Discount Band": ["0%", "≤10%", "10–20%", "20–30%", "30–50%", ">50%"]})
            fig.update_layout(showlegend=False, xaxis_title="Discount band")
            plot(fig, 360)
            st.caption("Compare profit distributions across discount bands. Product and customer mix may also explain differences.")

# ===== Products =====
with tab_products:
    if has(filtered, "Sub-Category", "Sales", "Profit"):
        pp = filtered.groupby("Sub-Category", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        pp["Margin %"] = np.where(pp["Sales"] != 0, pp["Profit"] / pp["Sales"] * 100, np.nan)
        pp = pp.sort_values("Sales", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=pp["Sub-Category"], x=pp["Sales"], name="Sales", orientation="h", marker_color=BLUE))
        fig.add_trace(go.Bar(y=pp["Sub-Category"], x=pp["Profit"], name="Profit", orientation="h", marker_color=TEAL))
        fig.update_layout(title="Which sub-categories generate sales — and which generate profit?", barmode="group", xaxis_title="Amount")
        plot(fig, max(430, len(pp) * 27))

        left, right = st.columns(2)
        with left:
            m = pp.sort_values("Margin %")
            fig = px.bar(m, x="Margin %", y="Sub-Category", orientation="h", color="Margin %", color_continuous_scale=DIVERGING,
                         color_continuous_midpoint=0, title="Profit margin by sub-category")
            fig.update_layout(coloraxis_showscale=False)
            plot(fig, max(380, len(m) * 26))
        with right:
            fig = px.scatter(pp, x="Sales", y="Profit", text="Sub-Category", size=pp["Sales"].abs(), color="Margin %",
                             color_continuous_scale=DIVERGING, color_continuous_midpoint=0,
                             title="Portfolio map: scale vs. profitability",
                             hover_data={"Margin %": ":.1f", "Sales": ":,.0f", "Profit": ":,.0f"})
            fig.add_hline(y=0, line_dash="dash", line_color=RED)
            fig.update_traces(textposition="top center", textfont=dict(size=10))
            plot(fig, max(380, len(pp) * 26))

    if has(filtered, "Product Name", "Profit"):
        prod = filtered.groupby("Product Name", as_index=False).agg(Profit=("Profit", "sum"), Sales=("Sales", "sum"))
        prod["Label"] = prod["Product Name"].map(short)
        left, right = st.columns(2)
        with left:
            top = prod.nlargest(10, "Profit").sort_values("Profit")
            fig = px.bar(top, x="Profit", y="Label", orientation="h", title="Top 10 products by profit", color_discrete_sequence=[TEAL],
                         hover_data={"Product Name": True, "Label": False})
            plot(fig, 400)
        with right:
            bot = prod.nsmallest(10, "Profit").sort_values("Profit", ascending=False)
            fig = px.bar(bot, x="Profit", y="Label", orientation="h", title="Bottom 10 products by profit — review candidates",
                         color_discrete_sequence=[RED], hover_data={"Product Name": True, "Label": False})
            plot(fig, 400)

    if has(filtered, "Sub-Category", "Sales", "Profit"):
        st.markdown("#### Sub-category detail")
        table(pp.sort_values("Profit").style.format({"Sales": "${:,.2f}", "Profit": "${:,.2f}", "Margin %": "{:.1f}%"}))
    else:
        st.info("Products & Profit requires Sub-Category, Sales and Profit columns.")

# ===== Customers & regions =====
with tab_customers:
    left, right = st.columns(2)
    with left:
        if has(filtered, "Region", "Profit"):
            reg = filtered.groupby("Region", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
            fig = px.bar(reg, x="Region", y=["Sales", "Profit"], barmode="group", title="Which regions contribute most to sales and profit?",
                         color_discrete_map={"Sales": BLUE, "Profit": TEAL})
            fig.update_layout(legend_title_text="")
            plot(fig, 370)
        else:
            st.info("Requires Region, Sales and Profit.")
    with right:
        if has(filtered, "Segment", "Profit"):
            seg = filtered.groupby("Segment", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
            fig = px.bar(seg, x="Segment", y=["Sales", "Profit"], barmode="group", title="Which customer segment is most valuable?",
                         color_discrete_map={"Sales": PURPLE, "Profit": TEAL})
            fig.update_layout(legend_title_text="")
            plot(fig, 370)
        else:
            st.info("Requires Segment, Sales and Profit.")

    if has(filtered, "Region", "Category", "Sales", "Profit"):
        g = filtered.groupby(["Region", "Category"]).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum")).reset_index()
        g["Margin %"] = np.where(g["Sales"] != 0, g["Profit"] / g["Sales"] * 100, np.nan)
        pv = g.pivot(index="Region", columns="Category", values="Margin %")
        fig = px.imshow(pv, text_auto=".1f", aspect="auto", color_continuous_scale=DIVERGING, color_continuous_midpoint=0,
                        title="Profit margin (%) by region and category", labels=dict(color="Margin %"))
        plot(fig, 330)

    if has(filtered, "State", "Profit"):
        st_p = filtered.groupby("State", as_index=False)["Profit"].sum()
        left, right = st.columns(2)
        with left:
            plot(px.bar(st_p.nlargest(10, "Profit").sort_values("Profit"), x="Profit", y="State", orientation="h",
                        title="Top 10 states by profit", color_discrete_sequence=[TEAL]), 380)
        with right:
            plot(px.bar(st_p.nsmallest(10, "Profit").sort_values("Profit", ascending=False), x="Profit", y="State", orientation="h",
                        title="Bottom 10 states by profit", color_discrete_sequence=[RED]), 380)

    if has(filtered, "Customer ID", "Sales", "Profit"):
        cust = filtered.groupby("Customer ID", as_index=False).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        if "Customer Name" in filtered:
            names = filtered.drop_duplicates("Customer ID").set_index("Customer ID")["Customer Name"]
            cust["Customer"] = cust["Customer ID"].map(names).fillna(cust["Customer ID"])
        else:
            cust["Customer"] = cust["Customer ID"]
        cust = cust.nlargest(15, "Sales").sort_values("Sales")
        fig = px.bar(cust, x="Sales", y="Customer", orientation="h", color="Profit", color_continuous_scale=DIVERGING,
                     color_continuous_midpoint=0, title="Who are the top 15 customers by sales — and are they profitable?",
                     hover_data={"Profit": ":$,.2f", "Sales": ":$,.2f"})
        plot(fig, 480)
    if has(filtered, "Segment", "Order ID"):
        o = filtered.groupby("Segment").agg(Orders=("Order ID", "nunique")).reset_index()
        fig = px.pie(o, names="Segment", values="Orders", hole=.65, title="Share of distinct orders by customer segment", color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="outside", textinfo="label+percent")
        plot(fig, 330)

# ===== Operations =====
with tab_operations:
    left, right = st.columns(2)
    with left:
        if has(filtered, "Ship Mode", "Sales"):
            agg = {"Sales": ("Sales", "sum")}
            if "Shipping Days" in filtered:
                agg["Avg days"] = ("Shipping Days", "mean")
            ship = filtered.groupby("Ship Mode", as_index=False).agg(**agg)
            fig = px.bar(ship.sort_values("Sales"), x="Sales", y="Ship Mode", orientation="h", title="Which shipping modes carry the most sales?",
                         color="Avg days" if "Avg days" in ship else "Sales", color_continuous_scale=["#45E0C1", "#FFB86B", "#FF718B"]
                         if "Avg days" in ship else ["#6D9EFF", "#45E0C1"])
            plot(fig, 360)
            if "Avg days" in ship:
                st.caption("Bar color shows the average shipping time (days) for each mode.")
        else:
            st.info("Requires Ship Mode and Sales.")
    with right:
        if "Shipping Days" in filtered:
            sd = filtered["Shipping Days"].dropna()
            neg = int((sd < 0).sum())
            sd = sd[sd >= 0]
            if not sd.empty:
                hist = sd.value_counts().sort_index().rename_axis("Shipping Days").reset_index(name="Order lines")
                plot(px.bar(hist, x="Shipping Days", y="Order lines", title="How long does shipping take?", color_discrete_sequence=[PURPLE]), 360)
                st.caption(f"Median shipping time: {sd.median():.1f} days." + (f" {neg:,} rows with negative durations were excluded (check date format)." if neg else ""))
        else:
            st.info("Requires Order Date and Ship Date.")

    if has(filtered, "Discount Band", "Sales", "Profit"):
        db = filtered.groupby("Discount Band", observed=True).agg(
            Lines=("Profit", "size"), Sales=("Sales", "sum"), Profit=("Profit", "sum"),
            Avg_Profit=("Profit", "mean"), Loss_Share=("Profit", lambda s: (s < 0).mean() * 100)).reset_index()
        db["Margin %"] = np.where(db["Sales"] != 0, db["Profit"] / db["Sales"] * 100, np.nan)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=db["Discount Band"].astype(str), y=db["Profit"], name="Total profit",
                             marker_color=[TEAL if v >= 0 else RED for v in db["Profit"]]))
        fig.add_trace(go.Scatter(x=db["Discount Band"].astype(str), y=db["Loss_Share"], name="% of lines losing money",
                                 mode="lines+markers", line=dict(color=ORANGE, width=3), yaxis="y2"))
        fig.update_layout(title="Discount bands: total profit vs. share of loss-making lines",
                          yaxis=dict(title="Total profit"),
                          yaxis2=dict(title="% loss-making lines", overlaying="y", side="right", showgrid=False, range=[0, 105]),
                          legend=dict(orientation="h", y=1.12, x=0))
        plot(fig, 400)
        st.caption("Diagnostic, not a recommendation to remove discounts. Check product, segment and region mix before acting.")
        table(db.rename(columns={"Avg_Profit": "Avg profit / line", "Loss_Share": "Loss-making lines %"}).style.format(
            {"Lines": "{:,.0f}", "Sales": "${:,.0f}", "Profit": "${:,.0f}", "Avg profit / line": "${:,.2f}",
             "Loss-making lines %": "{:.1f}%", "Margin %": "{:.1f}%"}))

# ===== Forecast =====
with tab_forecast:
    st.markdown("#### Sales forecast")
    st.caption("Holt-Winters model on monthly sales. It respects the Region / Category / Segment filters but always uses the full date history.")
    try:
        import statsmodels  # noqa: F401
        has_sm = True
    except ImportError:
        has_sm = False
    hist = None
    if has_sm and "Order Date" in base and base["Order Date"].notna().any():
        hist = base.dropna(subset=["Order Date"]).set_index("Order Date")["Sales"].resample("MS").sum()
        hist.index.freq = "MS"
    if not has_sm:
        st.info("Install `statsmodels` to enable forecasting:  `pip install statsmodels`")
    elif hist is None or len(hist) < 30:
        st.info("Forecasting needs at least ~30 months of order history for the selected Region / Category / Segment.")
    else:
        horizon = st.slider("Months to forecast", 3, 12, 12)
        fc = run_forecast(hist, horizon)
        point, lo, hi = fc["point"], fc["lower"], fc["upper"]
        last_same = hist[-horizon:].sum()
        growth = (point.sum() / last_same - 1) * 100 if last_same else np.nan
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Forecast sales · next {horizon} mo", money(point.sum()))
        k2.metric("vs last same-length period", f"{growth:+.1f}%" if pd.notna(growth) else "—")
        k3.metric("Expected peak month", f"{point.idxmax():%b %Y}")
        k4.metric("80% range (total)", f"{money(lo.sum())} – {money(hi.sum())}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History", mode="lines", line=dict(color=TEAL, width=2.5)))
        fig.add_trace(go.Scatter(x=list(point.index) + list(point.index[::-1]), y=list(hi) + list(lo[::-1]), fill="toself",
                                 fillcolor="rgba(255,184,107,0.18)", line=dict(width=0), name="80% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[hist.index[-1]] + list(point.index), y=[hist.values[-1]] + list(point.values), name="Forecast",
                                 mode="lines+markers", line=dict(color=ORANGE, width=2.5, dash="dash")))
        fig.update_layout(title="Monthly sales: history and forecast", legend=dict(orientation="h", y=1.12, x=0))
        plot(fig, 400)

        out_tbl = pd.DataFrame({"Month": point.index.strftime("%b %Y"), "Forecast": point.values.round(0),
                                "Low (10%)": lo.round(0), "High (90%)": hi.round(0)})
        left, right = st.columns([1, 1.4])
        with left:
            table(out_tbl.style.format({"Forecast": "${:,.0f}", "Low (10%)": "${:,.0f}", "High (90%)": "${:,.0f}"}))
            st.download_button("Download forecast as CSV", out_tbl.to_csv(index=False).encode("utf-8-sig"),
                               file_name="sales_forecast.csv", mime="text/csv")
        with right:
            bt = fc.get("backtest")
            if bt:
                b1, b2 = st.columns(2)
                b1.metric("Holdout error · Holt-Winters", f"{bt['wape_hw']:.1f}%")
                b2.metric("Holdout error · seasonal naive", f"{bt['wape_sn']:.1f}%",
                          help="Baseline: same month last year. The model should beat it.")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bt["test"].index, y=bt["test"].values, name="Actual", line=dict(color=TEXT, width=2.5)))
                fig.add_trace(go.Scatter(x=bt["test"].index, y=bt["hw"], name="Holt-Winters", line=dict(color=ORANGE)))
                fig.add_trace(go.Scatter(x=bt["test"].index, y=bt["snaive"], name="Seasonal naive", line=dict(color=BLUE, dash="dot")))
                fig.update_layout(title="How accurate was it on the last 12 months?", legend=dict(orientation="h", y=1.15, x=0))
                plot(fig, 300)
        st.caption("Error = weighted absolute % error on the last 12 known months (model trained on earlier data only). "
                   "Expect roughly 15–25% on this dataset: use the forecast for direction and seasonality, not exact monthly targets. "
                   "The trend term extrapolates recent growth, so treat the upper range as a best case.")

# ===== Data quality =====
with tab_data:
    st.markdown("#### Dataset health")
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
                     title="Which columns need data-quality attention?", color="Missing %",
                     color_continuous_scale=["#45E0C1", "#FFB86B", "#FF718B"])
        plot(fig, max(320, min(600, 30 * len(miss.head(20)))))
    else:
        st.success("No missing values detected in the loaded dataset.")

    st.markdown("#### Preview filtered records")
    st.dataframe(filtered.head(500), hide_index=True, **STRETCH)
    st.download_button("Download filtered data as CSV", data=filtered.to_csv(index=False).encode("utf-8-sig"),
                       file_name="superstore_filtered.csv", mime="text/csv")

st.markdown("---")
st.markdown(
    f'<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:{MUTED};font-size:11px">'
    '<span>✦ InsightLab · Analytical learning workspace</span>'
    f'<span>{len(filtered):,} rows in current view · {len(df.columns)} fields detected</span></div>',
    unsafe_allow_html=True,
)
