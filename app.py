from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="InsightLab | Superstore Analytics",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual identity ----------
BG = "#0B1020"
PANEL = "#121A2E"
PANEL_2 = "#17213A"
TEXT = "#F2F5FF"
MUTED = "#98A6C7"
TEAL = "#45E0C1"
BLUE = "#6D9EFF"
PURPLE = "#B59CFF"
ORANGE = "#FFB86B"
RED = "#FF718B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {{ color-scheme: dark; }}
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
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
.kpi-note {{ color:{TEAL}; font-size:11px; margin-top:5px; }}
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
    margin=dict(l=18, r=18, t=45, b=18), title_font=dict(size=15, color=TEXT),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
    xaxis=dict(gridcolor="#26324D", zerolinecolor="#26324D", linecolor="#26324D"),
    yaxis=dict(gridcolor="#26324D", zerolinecolor="#26324D", linecolor="#26324D"),
)
PALETTE = [TEAL, BLUE, PURPLE, ORANGE, RED, "#7BD3FF", "#D6E685", "#F28BCB"]

def money(value):
    if pd.isna(value): return "—"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if v >= 1_000_000: return f"{sign}${v/1_000_000:,.2f}M"
    if v >= 1_000: return f"{sign}${v/1_000:,.1f}K"
    return f"{sign}${v:,.0f}"

def fmt_int(value):
    return "—" if pd.isna(value) else f"{value:,.0f}"

def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    return df

def prepare_data(raw):
    df = clean_columns(raw.copy())
    for col in ["Order Date", "Ship Date"]:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["Sales", "Profit", "Quantity", "Discount", "Postal Code"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Order Date" in df:
        df["Year"] = df["Order Date"].dt.year
        df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
        df["Month Name"] = df["Order Date"].dt.strftime("%b")
    if {"Order Date", "Ship Date"}.issubset(df.columns):
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df

def plotly_finish(fig, height=350):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    return fig

def has(df, *cols):
    return all(c in df.columns for c in cols)

def read_upload(uploaded):
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(uploaded)
            return pd.read_csv(uploaded, encoding_errors="replace")
        except Exception as e:
            st.error(f"تعذّر قراءة الملف: {e}")
            st.stop()

    candidates = [Path("Superstore_Cleaned.csv")]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path, encoding_errors="replace")
            except Exception:
                pass
    return None

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

raw = read_upload(uploaded)
if raw is None:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">DATA INTELLIGENCE / LEARNING STUDIO</div>
      <h1>InsightLab</h1>
      <p>A polished, interactive analytics workspace for exploring Superstore sales, profitability, customer segments and operational patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("ابدأ برفع ملف البيانات من الشريط الجانبي. يدعم التطبيق ملفات CSV وExcel، ويمكنه أيضًا قراءة `Superstore.csv` إذا كان موجودًا بجانب `app.py`.")
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

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ DATA INTELLIGENCE / LEARNING STUDIO</div>
  <h1>Superstore Analytics</h1>
  <p>From raw transactions to meaningful decisions. Explore sales momentum, profit drivers, customer behavior and operational efficiency through interactive visual analysis.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Filters ----------
with st.container():
    f1, f2, f3, f4 = st.columns([1.25, 1, 1, 1])
    filtered = df.copy()
    if "Order Date" in df and df["Order Date"].notna().any():
        min_date, max_date = df["Order Date"].min().date(), df["Order Date"].max().date()
        with f1:
            date_range = st.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered = filtered[filtered["Order Date"].between(start, end)]
    else:
        f1.caption("Order dates unavailable")
    with f2:
        region_opts = sorted(df["Region"].dropna().astype(str).unique()) if "Region" in df else []
        regions = st.multiselect("Region", region_opts, default=region_opts)
    with f3:
        cat_opts = sorted(df["Category"].dropna().astype(str).unique()) if "Category" in df else []
        cats = st.multiselect("Category", cat_opts, default=cat_opts)
    with f4:
        seg_opts = sorted(df["Segment"].dropna().astype(str).unique()) if "Segment" in df else []
        segs = st.multiselect("Customer segment", seg_opts, default=seg_opts)
if "Region" in filtered and region_opts: filtered = filtered[filtered["Region"].astype(str).isin(regions)]
if "Category" in filtered and cat_opts: filtered = filtered[filtered["Category"].astype(str).isin(cats)]
if "Segment" in filtered and seg_opts: filtered = filtered[filtered["Segment"].astype(str).isin(segs)]

if filtered.empty:
    st.warning("لا توجد سجلات تطابق الفلاتر المختارة. وسّع نطاق التاريخ أو أعد اختيار الفئات.")
    st.stop()

# ---------- KPI strip ----------
sales = filtered["Sales"].sum() if "Sales" in filtered else np.nan
profit = filtered["Profit"].sum() if "Profit" in filtered else np.nan
orders = filtered["Order ID"].nunique() if "Order ID" in filtered else len(filtered)
customers = filtered["Customer ID"].nunique() if "Customer ID" in filtered else np.nan
units = filtered["Quantity"].sum() if "Quantity" in filtered else np.nan
margin = (profit / sales * 100) if pd.notna(profit) and pd.notna(sales) and sales != 0 else np.nan
avg_order = sales / orders if pd.notna(sales) and orders else np.nan

st.markdown('<div class="section-title">Executive snapshot</div><div class="section-sub">The headline metrics for the selected data. All values update with your filters.</div>', unsafe_allow_html=True)
kpis = [
    ("TOTAL SALES", money(sales), "Revenue across selected records"),
    ("NET PROFIT", money(profit), f"{margin:.1f}% profit margin" if pd.notna(margin) else "Profitability indicator"),
    ("ORDERS", fmt_int(orders), f"{money(avg_order)} average sales / order" if pd.notna(avg_order) else "Distinct order count"),
    ("CUSTOMERS", fmt_int(customers), "Distinct customer IDs" if pd.notna(customers) else "Customer field unavailable"),
    ("UNITS SOLD", fmt_int(units), "Total quantity" if pd.notna(units) else "Quantity field unavailable"),
]
cols = st.columns(5)
for col, (label, value, note) in zip(cols, kpis):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

# ---------- Insights generated from actual selected data ----------
st.markdown('<div class="section-title">Signals worth investigating</div><div class="section-sub">Short, data-driven observations generated from the current selection — not causal claims.</div>', unsafe_allow_html=True)
insights = []
if has(filtered, "Category", "Sales"):
    cs = filtered.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    if len(cs): insights.append(f"<strong>Sales leader:</strong> {cs.index[0]} contributes the most sales ({money(cs.iloc[0])}).")
if has(filtered, "Sub-Category", "Profit"):
    ps = filtered.groupby("Sub-Category")["Profit"].sum().sort_values()
    if len(ps):
        insights.append(f"<strong>Profit watchlist:</strong> {ps.index[0]} has the lowest total profit ({money(ps.iloc[0])}); investigate pricing, discounting and product mix.")
if has(filtered, "Discount", "Profit"):
    corr_data = filtered[["Discount", "Profit"]].dropna()
    if len(corr_data) > 2 and corr_data["Discount"].nunique() > 1 and corr_data["Profit"].nunique() > 1:
        corr = corr_data["Discount"].corr(corr_data["Profit"])
        direction = "negative" if corr < -0.15 else "positive" if corr > 0.15 else "weak"
        insights.append(f"<strong>Discount signal:</strong> discount and profit have a {direction} linear correlation ({corr:.2f}). This is an association, not proof that discounts cause profit changes.")
if has(filtered, "Region", "Profit"):
    rp = filtered.groupby("Region")["Profit"].sum().sort_values()
    if len(rp): insights.append(f"<strong>Regional opportunity:</strong> {rp.index[0]} has the lowest total profit ({money(rp.iloc[0])}).")
if not insights:
    insights.append("Upload a dataset with standard Superstore fields to unlock automated business signals.")
for item in insights:
    st.markdown(f'<div class="insight">{item}</div>', unsafe_allow_html=True)

# ---------- Charts ----------
st.markdown('<div class="section-title">Explore the business</div><div class="section-sub">Each chart is designed to answer a practical question. Hover over marks for details; use the filters above to investigate a subset.</div>', unsafe_allow_html=True)
tab_overview, tab_products, tab_customers, tab_operations, tab_data = st.tabs([
    "✦ Overview", "▦ Products & Profit", "◎ Customers & Regions", "↗ Operations", "⌘ Data quality"
])

with tab_overview:
    left, right = st.columns([1.45, 1])
    with left:
        if has(filtered, "Order Date", "Sales") and filtered["Order Date"].notna().any():
            monthly = filtered.dropna(subset=["Order Date"]).set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
            fig = px.area(monthly, x="Order Date", y="Sales", title="Are sales growing or slowing over time?",
                          labels={"Order Date":"Month", "Sales":"Sales"}, color_discrete_sequence=[TEAL])
            fig.update_traces(line=dict(width=2.5), fillcolor="rgba(69,224,193,0.13)")
            fig.update_xaxes(dtick="M3", tickformat="%b\n%Y")
            st.plotly_chart(plotly_finish(fig, 370), use_container_width=True)
        elif has(filtered, "Category", "Sales"):
            tmp = filtered.groupby("Category", as_index=False)["Sales"].sum()
            fig = px.bar(tmp, x="Category", y="Sales", title="Sales by category", color="Category", color_discrete_sequence=PALETTE)
            st.plotly_chart(plotly_finish(fig), use_container_width=True)
        else:
            st.info("Add Order Date and Sales columns to view sales trends.")
    with right:
        if has(filtered, "Category", "Sales"):
            tmp = filtered.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
            fig = px.pie(tmp, names="Category", values="Sales", hole=.68, title="Which categories drive revenue?",
                         color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color=PANEL, width=3)))
            st.plotly_chart(plotly_finish(fig, 370), use_container_width=True)
        elif has(filtered, "Region", "Sales"):
            tmp = filtered.groupby("Region", as_index=False)["Sales"].sum()
            fig = px.bar(tmp, x="Region", y="Sales", title="Sales by region", color="Region", color_discrete_sequence=PALETTE)
            st.plotly_chart(plotly_finish(fig), use_container_width=True)
        else:
            st.info("Add Category or Region and Sales to view revenue mix.")
    left, right = st.columns(2)
    with left:
        if has(filtered, "Sales", "Profit"):
            fig = px.scatter(filtered, x="Sales", y="Profit", color="Category" if "Category" in filtered else None,
                             size="Quantity" if "Quantity" in filtered else None,
                             hover_data=[c for c in ["Product Name", "Sub-Category", "Discount", "Region"] if c in filtered],
                             title="Do high-sales transactions also deliver profit?",
                             color_discrete_sequence=PALETTE, opacity=.72)
            fig.add_hline(y=0, line_dash="dash", line_color=RED, opacity=.8)
            st.plotly_chart(plotly_finish(fig, 360), use_container_width=True)
            st.caption("Points below zero represent loss-making transaction lines. Large sales alone do not guarantee profit.")
    with right:
        if has(filtered, "Discount", "Profit"):
            fig = px.box(filtered, x="Discount", y="Profit", points=False,
                         title="How does profitability vary by discount level?",
                         labels={"Discount":"Discount rate", "Profit":"Profit"},
                         color="Discount", color_discrete_sequence=PALETTE)
            fig.update_layout(showlegend=False)
            st.plotly_chart(plotly_finish(fig, 360), use_container_width=True)
            st.caption("Compare the profit distribution at different discount levels. Product and customer mix may also explain differences.")

with tab_products:
    if has(filtered, "Sub-Category", "Sales", "Profit"):
        product_perf = filtered.groupby("Sub-Category", as_index=False).agg(Sales=("Sales","sum"), Profit=("Profit","sum"))
        product_perf["Margin %"] = np.where(product_perf["Sales"] != 0, product_perf["Profit"] / product_perf["Sales"] * 100, np.nan)
        product_perf = product_perf.sort_values("Sales", ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=product_perf["Sub-Category"], x=product_perf["Sales"], name="Sales", orientation="h", marker_color=BLUE))
        fig.add_trace(go.Bar(y=product_perf["Sub-Category"], x=product_perf["Profit"], name="Profit", orientation="h", marker_color=TEAL))
        fig.update_layout(title="Which sub-categories generate sales — and which generate profit?", barmode="group",
                          xaxis_title="Amount", yaxis_title="")
        st.plotly_chart(plotly_finish(fig, max(430, len(product_perf)*27)), use_container_width=True)
        st.caption("Sales and profit use the same axis for directional comparison; profit can be negative.")
        fig = px.scatter(product_perf, x="Sales", y="Profit", text="Sub-Category", size=product_perf["Sales"].abs(),
                         color="Margin %", color_continuous_scale=["#FF718B", "#B59CFF", "#45E0C1"],
                         title="Product portfolio map: scale vs. profitability",
                         hover_data={"Margin %":":.1f", "Sales":":,.0f", "Profit":":,.0f"})
        fig.add_hline(y=0, line_dash="dash", line_color=RED)
        fig.update_traces(textposition="top center", textfont=dict(size=10))
        st.plotly_chart(plotly_finish(fig, 470), use_container_width=True)
        st.dataframe(product_perf.sort_values("Profit", ascending=True).style.format({"Sales":"${:,.2f}", "Profit":"${:,.2f}", "Margin %":"{:.1f}%"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("The Products & Profit analysis requires Sub-Category, Sales and Profit columns.")

with tab_customers:
    left, right = st.columns(2)
    with left:
        if has(filtered, "Region", "Sales", "Profit"):
            reg = filtered.groupby("Region", as_index=False).agg(Sales=("Sales","sum"), Profit=("Profit","sum"))
            fig = px.bar(reg, x="Region", y=["Sales", "Profit"], barmode="group",
                         title="Which regions contribute most to sales and profit?",
                         color_discrete_map={"Sales":BLUE, "Profit":TEAL})
            st.plotly_chart(plotly_finish(fig, 370), use_container_width=True)
        else:
            st.info("Requires Region, Sales and Profit.")
    with right:
        if has(filtered, "Segment", "Sales", "Profit"):
            seg = filtered.groupby("Segment", as_index=False).agg(Sales=("Sales","sum"), Profit=("Profit","sum"))
            fig = px.bar(seg, x="Segment", y=["Sales", "Profit"], barmode="group",
                         title="Which customer segment is most valuable?",
                         color_discrete_map={"Sales":PURPLE, "Profit":TEAL})
            st.plotly_chart(plotly_finish(fig, 370), use_container_width=True)
        else:
            st.info("Requires Segment, Sales and Profit.")
    if has(filtered, "Customer ID", "Sales", "Profit"):
        customer = filtered.groupby("Customer ID", as_index=False).agg(Sales=("Sales","sum"), Profit=("Profit","sum"))
        customer = customer.sort_values("Sales", ascending=False).head(15)
        fig = px.bar(customer.sort_values("Sales"), x="Sales", y="Customer ID", orientation="h",
                     color="Profit", color_continuous_scale=["#FF718B", "#B59CFF", "#45E0C1"],
                     title="Who are the top 15 customers by sales — and are they profitable?",
                     hover_data={"Profit":":$,.2f", "Sales":":$,.2f"})
        st.plotly_chart(plotly_finish(fig, 480), use_container_width=True)
    if has(filtered, "Segment", "Order ID"):
        orders_seg = filtered.groupby("Segment").agg(Orders=("Order ID","nunique")).reset_index()
        fig = px.pie(orders_seg, names="Segment", values="Orders", hole=.65, title="Share of distinct orders by customer segment", color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="outside", textinfo="label+percent")
        st.plotly_chart(plotly_finish(fig, 330), use_container_width=True)

with tab_operations:
    left, right = st.columns(2)
    with left:
        if has(filtered, "Ship Mode", "Sales"):
            ship = filtered.groupby("Ship Mode", as_index=False).agg(Sales=("Sales","sum"))
            if "Order ID" in filtered:
                order_counts = filtered.groupby("Ship Mode")["Order ID"].nunique().rename("Orders").reset_index()
                ship = ship.merge(order_counts, on="Ship Mode", how="left")
            fig = px.bar(ship.sort_values("Sales"), x="Sales", y="Ship Mode", orientation="h",
                         title="Which shipping modes carry the most sales?", color="Sales", color_continuous_scale=["#6D9EFF", "#45E0C1"])
            st.plotly_chart(plotly_finish(fig, 360), use_container_width=True)
        else:
            st.info("Requires Ship Mode and Sales.")
    with right:
        if has(filtered, "Shipping Days"):
            ship_days = filtered["Shipping Days"].dropna()
            ship_days = ship_days[ship_days >= 0]
            if not ship_days.empty:
                hist = ship_days.value_counts().sort_index().rename_axis("Shipping Days").reset_index(name="Order lines")
                fig = px.bar(hist, x="Shipping Days", y="Order lines", title="How long does shipping take?",
                             color_discrete_sequence=[PURPLE])
                st.plotly_chart(plotly_finish(fig, 360), use_container_width=True)
                st.caption(f"Median shipping time: {ship_days.median():.1f} days. Negative date differences are excluded from this chart.")
        else:
            st.info("Requires Order Date and Ship Date.")
    if has(filtered, "Discount", "Sales", "Profit"):
        disc = filtered.groupby("Discount", as_index=False).agg(Sales=("Sales","sum"), Profit=("Profit","sum"), Avg_Profit=("Profit","mean"), Rows=("Profit","size"))
        disc["Discount label"] = (disc["Discount"] * 100).round(0).astype(str) + "%"
        fig = go.Figure()
        fig.add_trace(go.Bar(x=disc["Discount label"], y=disc["Sales"], name="Sales", marker_color=BLUE, yaxis="y"))
        fig.add_trace(go.Scatter(x=disc["Discount label"], y=disc["Avg_Profit"], name="Average profit per line", mode="lines+markers",
                                 line=dict(color=ORANGE, width=3), yaxis="y2"))
        fig.update_layout(title="Discount levels: revenue vs. average profit per transaction line",
                          yaxis=dict(title="Sales", gridcolor="#26324D"),
                          yaxis2=dict(title="Average profit", overlaying="y", side="right", showgrid=False))
        st.plotly_chart(plotly_finish(fig, 400), use_container_width=True)
        st.caption("This chart is diagnostic, not a recommendation to remove discounts. Check product, segment and region mix before acting.")

with tab_data:
    st.markdown("#### Dataset health")
    quality = pd.DataFrame({
        "Metric": ["Rows", "Columns", "Exact duplicate rows", "Missing cells", "Missing cells (%)"],
        "Value": [len(df), len(df.columns), int(df.duplicated().sum()), int(df.isna().sum().sum()),
                  round(df.isna().sum().sum() / max(df.size, 1) * 100, 2)]
    })
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows", f"{len(df):,}")
    q2.metric("Columns", f"{len(df.columns):,}")
    q3.metric("Exact duplicate rows", f"{df.duplicated().sum():,}")
    miss = (df.isna().sum().rename("Missing values").to_frame())
    miss["Missing %"] = (df.isna().mean() * 100).round(2)
    miss = miss[miss["Missing values"] > 0].sort_values("Missing values", ascending=False)
    if not miss.empty:
        fig = px.bar(miss.reset_index(names="Column").head(20), x="Missing values", y="Column", orientation="h",
                     title="Which columns need data-quality attention?", color="Missing %",
                     color_continuous_scale=["#45E0C1", "#FFB86B", "#FF718B"])
        st.plotly_chart(plotly_finish(fig, max(320, min(600, 30*len(miss.head(20))))), use_container_width=True)
    else:
        st.success("No missing values detected in the loaded dataset.")
    st.markdown("#### Preview filtered records")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download filtered data as CSV", data=filtered.to_csv(index=False).encode("utf-8-sig"),
                       file_name="superstore_filtered.csv", mime="text/csv")

st.markdown("---")
st.markdown(
    f'<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:{MUTED};font-size:11px">'
    '<span>✦ InsightLab · Analytical learning workspace</span>'
    f'<span>{len(filtered):,} rows in current view · {len(df.columns)} fields detected</span></div>',
    unsafe_allow_html=True
)
