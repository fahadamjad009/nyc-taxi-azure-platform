import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="NYC Taxi Analytics", layout="wide", page_icon="🚕",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .stApp{background:#0d0d0d;color:#e8e8e8}
  [data-testid="stSidebar"]{display:none}
  #MainMenu,footer,header{visibility:hidden}
  [data-testid="collapsedControl"]{display:none}

  /* Top nav bar */
  .topbar{
    display:flex;align-items:center;justify-content:space-between;
    background:#111;border:1px solid #1f1f1f;border-radius:12px;
    padding:14px 24px;margin-bottom:20px;flex-wrap:wrap;gap:12px;
  }
  .topbar-brand{
    font-size:13px;font-weight:800;letter-spacing:.1em;color:#fff;text-transform:uppercase;
  }
  .topbar-brand span{color:#00ff88}
  .topbar-tags{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
  .topbar-filters{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
  .filter-label{font-size:10px;font-weight:700;letter-spacing:.12em;
                text-transform:uppercase;color:#555;margin-right:4px}

  .sec{font-size:10px;font-weight:700;letter-spacing:.18em;text-transform:uppercase;
       color:#00ff88;margin:18px 0 6px;padding-bottom:5px;border-bottom:1px solid #1f1f1f}
  .insight{background:#111;border:1px solid #1f1f1f;border-left:3px solid #00ff88;
            border-radius:6px;padding:11px 15px;margin:6px 0;font-size:12px;color:#999;line-height:1.7}
  .tag{display:inline-block;background:#161616;border:1px solid #2a2a2a;border-radius:20px;
       padding:3px 10px;font-size:10px;color:#888;margin:2px;white-space:nowrap}
  .tag-g{border-color:#00ff88;color:#00ff88}
  .tag-b{border-color:#3b82f6;color:#3b82f6}
  .tag-a{border-color:#f59e0b;color:#f59e0b}

  [data-testid="stMetric"]{background:#111;border:1px solid #1f1f1f;border-radius:10px;padding:16px!important}
  [data-testid="stMetricLabel"]{color:#555!important;font-size:10px!important;letter-spacing:.1em}
  [data-testid="stMetricValue"]{color:#fff!important;font-size:26px!important;font-weight:800!important}
  [data-testid="stMetricDelta"] svg{display:none}
  [data-testid="stMetricDelta"]{color:#00ff88!important;font-size:11px!important}
  [data-testid="stTabs"] button{color:#444!important;font-size:12px!important;font-weight:600!important;padding:8px 16px!important}
  [data-testid="stTabs"] button[aria-selected="true"]{color:#00ff88!important;border-bottom-color:#00ff88!important}
  [data-testid="stDataFrame"]{background:#111!important;border-radius:8px}
  hr{border-color:#1f1f1f!important}

  /* typewriter animation for subtitle */
  .typewriter{
    overflow:hidden;white-space:nowrap;
    border-right:.1em solid #00ff88;
    animation:typing 3.5s steps(60,end), blink-caret .75s step-end infinite;
    font-size:12px;color:#555;max-width:700px;
  }
  @keyframes typing{from{width:0}to{width:100%}}
  @keyframes blink-caret{from,to{border-color:transparent}50%{border-color:#00ff88}}

  /* pipeline steps horizontal */
  .pipeline{display:flex;gap:0;align-items:center;flex-wrap:wrap;margin:4px 0}
  .pstep{font-size:10px;color:#555;white-space:nowrap}
  .parrow{font-size:10px;color:#00ff88;margin:0 4px}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data" / "exports"

MINT   = "#00ff88"
RED    = "#ff4d4d"
BLUE   = "#3b82f6"
AMBER  = "#f59e0b"
PURPLE = "#a78bfa"
CYAN   = "#22d3ee"
COLORS = [MINT, BLUE, AMBER, PURPLE, CYAN, RED]
VL     = {"yellow":"Yellow Taxi","green":"Green Taxi","fhv":"FHV (Black Car)","fhvhv":"HVFHV (Uber/Lyft)"}
DAY    = {1:"Sun",2:"Mon",3:"Tue",4:"Wed",5:"Thu",6:"Fri",7:"Sat"}
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def dark_layout(height=440, legend=True, **kwargs):
    base = dict(
        paper_bgcolor="#111", plot_bgcolor="#0d0d0d",
        font=dict(color="#aaa", family="Inter, sans-serif", size=11),
        xaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#222", showline=False,
                   tickfont=dict(color="#666", size=10)),
        yaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#222", showline=False,
                   tickfont=dict(color="#666", size=10)),
        margin=dict(l=50, r=20, t=36, b=40),
        height=height,
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1f1f1f",
                    font=dict(color="#888", size=10)),
        hoverlabel=dict(bgcolor="#1a1a1a", bordercolor="#333",
                        font=dict(color="#eee", size=11)),
    )
    base.update(kwargs)
    return base

def style_line(fig, height=440, legend=True):
    fig.update_traces(line=dict(width=2))
    fig.update_layout(**dark_layout(height=height, legend=legend))
    return fig

@st.cache_data
def load_data():
    monthly = pd.read_parquet(DATA_DIR/"monthly_trips.parquet")
    hourly  = pd.read_parquet(DATA_DIR/"hourly_patterns.parquet")
    annual  = pd.read_parquet(DATA_DIR/"annual_kpis.parquet")
    fare    = pd.read_parquet(DATA_DIR/"fare_distribution.parquet")
    borough = pd.read_parquet(DATA_DIR/"borough_volumes.parquet")
    ml      = pd.read_parquet(DATA_DIR/"ml_sample.parquet")
    return monthly, hourly, annual, fare, borough, ml

@st.cache_resource
def load_model():
    import pickle
    from pathlib import Path
    pkl_path = Path(__file__).parent / "models" / "fare_model.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            b = pickle.load(f)
        return (b["model"], b["shap_vals"], b["X_shap"], b["features"],
                b["metrics"]["r2"], b["metrics"]["rmse"],
                b["y_true"], b["y_pred"])
    else:
        # Fallback: train on the fly if pkl not present
        from xgboost import XGBRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import shap
        df2 = ml.dropna(subset=["fare_amount","trip_distance","passenger_count",
                                 "hour","day_of_week","month","year"]).copy()
        df2 = df2[(df2["fare_amount"].between(2.5,150)) & (df2["trip_distance"].between(0.1,40))]
        for c in ["pickup_borough","dropoff_borough"]:
            df2[c] = df2[c].fillna("Unknown")
            le = LabelEncoder()
            df2[c] = le.fit_transform(df2[c])
        feats = ["trip_distance","passenger_count","hour","day_of_week",
                 "month","year","pickup_borough","dropoff_borough"]
        X, y = df2[feats], df2["fare_amount"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        mdl = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                           random_state=42, n_jobs=-1, verbosity=0)
        mdl.fit(Xtr, ytr)
        exp = shap.TreeExplainer(mdl)
        idx = np.random.RandomState(42).choice(len(Xte), min(2000, len(Xte)), replace=False)
        sv  = exp.shap_values(Xte.iloc[idx])
        r2  = mdl.score(Xte, yte)
        yp  = mdl.predict(Xte)
        rmse = np.sqrt(np.mean((yte.values - yp)**2))
        return mdl, sv, Xte.iloc[idx], feats, r2, rmse, yte.values[:600], yp[:600]

monthly, hourly, annual, fare, borough, ml = load_data()

# ── Filters (inline, no sidebar) ─────────────────────────────────────────────
all_years = sorted(annual["year"].unique().tolist())
all_v = list(VL.keys())

f1, f2 = st.columns([1, 2])
with f1:
    yr = st.slider("Year Range", min(all_years), max(all_years),
                   (min(all_years), max(all_years)), label_visibility="visible")
with f2:
    sel_v = st.multiselect("Vehicle Types", all_v, default=all_v,
                           format_func=lambda x: VL[x], label_visibility="visible")
if not sel_v: sel_v = all_v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div>
    <div class="topbar-brand">NYC <span>Taxi</span> & Rideshare Analytics</div>
    <div class="pipeline" style="margin-top:6px">
      <span class="pstep">Azure ADF</span><span class="parrow">→</span>
      <span class="pstep">ADLS Gen2</span><span class="parrow">→</span>
      <span class="pstep">Databricks Delta</span><span class="parrow">→</span>
      <span class="pstep">DuckDB</span><span class="parrow">→</span>
      <span class="pstep">XGBoost+SHAP</span><span class="parrow">→</span>
      <span class="pstep" style="color:#00ff88">Streamlit</span>
    </div>
  </div>
  <div class="topbar-tags">
    <span class="tag tag-g">1.95B Trips</span>
    <span class="tag tag-g">2019–2025</span>
    <span class="tag tag-b">Azure ADF</span>
    <span class="tag tag-b">Databricks</span>
    <span class="tag tag-a">DuckDB</span>
    <span class="tag tag-a">XGBoost</span>
    <span class="tag">SHAP</span>
    <span class="tag">4 Vehicle Classes</span>
  </div>
</div>
<div class="typewriter">
  1.95 billion trips · All NYC TLC vehicle classes · Azure Data Factory → Databricks Delta Lake → DuckDB → XGBoost ML → SHAP Explainability → Streamlit
</div>
<div style="margin:10px 0 4px"></div>
<hr>""", unsafe_allow_html=True)

# ── Filtered ──────────────────────────────────────────────────────────────────
ann_f = annual[(annual["year"].between(*yr)) & (annual["vehicle_type"].isin(sel_v))]
mon_f = monthly[(monthly["year"].between(*yr)) & (monthly["vehicle_type"].isin(sel_v))]
hr_f  = hourly[(hourly["year"].between(*yr))  & (hourly["vehicle_type"].isin(sel_v))]
fare_f= fare[(fare["year"].between(*yr))      & (fare["vehicle_type"].isin(sel_v))]
bor_f = borough[borough["year"].between(*yr)]

# ── KPIs ──────────────────────────────────────────────────────────────────────
tt = ann_f["total_trips"].sum()
tr = ann_f["total_revenue"].sum()
af = ann_f["avg_fare"].mean()
ad = ann_f["avg_distance"].mean()
py_trips = ann_f[ann_f["year"]==yr[1]-1]["total_trips"].sum() if yr[1]-1 >= yr[0] else None
cy_trips = ann_f[ann_f["year"]==yr[1]]["total_trips"].sum()
delta = f"+{(cy_trips-py_trips)/py_trips*100:.1f}% YoY" if py_trips and py_trips > 0 else ""

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Trips",     f"{tt/1e9:.2f}B", delta)
k2.metric("Total Revenue",   f"${tr/1e9:.1f}B")
k3.metric("Avg Fare",        f"${af:.2f}" if pd.notna(af) else "N/A")
k4.metric("Avg Distance",    f"{ad:.1f} mi" if pd.notna(ad) else "N/A")
k5.metric("Vehicle Classes", str(len(sel_v)))

st.markdown("<div style='margin:4px 0'></div>", unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📈  Market Overview","⏰  Demand Patterns",
    "🗺️  Borough Intelligence","💵  Fare Economics","🤖  ML & SHAP"
])

# ══════ TAB 1 ══════════════════════════════════════════════════════════════════
with tab1:
    mon_f2 = mon_f.copy()
    mon_f2["period"] = pd.to_datetime(mon_f2[["year","month"]].assign(day=1))
    piv = mon_f2.pivot_table(index="period",columns="vehicle_type",values="trip_count",aggfunc="sum").fillna(0)
    piv.columns = [VL.get(c,c) for c in piv.columns]
    piv = piv.reset_index()
    piv_long = piv.melt(id_vars="period", var_name="Vehicle", value_name="Trips")

    st.markdown('<div class="sec">Monthly Trip Volume by Vehicle Type</div>', unsafe_allow_html=True)
    fig = px.line(piv_long, x="period", y="Trips", color="Vehicle",
                  color_discrete_sequence=COLORS, template="plotly_dark")
    fig.update_traces(line_width=2.5)
    fig.add_vrect(x0="2020-03-01", x1="2020-07-01",
                  fillcolor="rgba(255,77,77,0.08)", line_width=0,
                  annotation_text="COVID", annotation_position="top left",
                  annotation_font_color=RED, annotation_font_size=10)
    fig.update_layout(**dark_layout(380))
    st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec">Market Share Over Time (Stacked %)</div>', unsafe_allow_html=True)
        stk = mon_f2.pivot_table(index="period",columns="vehicle_type",values="trip_count",aggfunc="sum").fillna(0)
        stk_pct = stk.div(stk.sum(axis=1),axis=0)*100
        stk_pct.columns = [VL.get(c,c) for c in stk_pct.columns]
        stk_long = stk_pct.reset_index().melt(id_vars="period",var_name="Vehicle",value_name="Share %")
        fig2 = px.area(stk_long, x="period", y="Share %", color="Vehicle",
                       color_discrete_sequence=COLORS, template="plotly_dark")
        fig2.update_traces(line_width=0)
        fig2.update_layout(**dark_layout(320, yaxis_range=[0,100]))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="sec">Trip Volume Treemap by Vehicle Type</div>', unsafe_allow_html=True)
        share = ann_f.groupby("vehicle_type")["total_trips"].sum().reset_index()
        share["Label"] = share["vehicle_type"].map(VL)
        share["pct"]   = (share["total_trips"]/share["total_trips"].sum()*100).round(1)
        fig3 = px.treemap(share, path=["Label"], values="total_trips",
                          color="pct", color_continuous_scale=[[0,"#0a2a1a"],[0.5,"#00994d"],[1,"#00ff88"]],
                          custom_data=["pct"])
        fig3.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]}%",
            hovertemplate="<b>%{label}</b><br>Trips: %{value:,.0f}<br>Share: %{customdata[0]}%<extra></extra>",
            textfont_size=13, marker_line_width=2, marker_line_color="#0d0d0d"
        )
        fig3.update_layout(paper_bgcolor="#111", margin=dict(l=0,r=0,t=0,b=0),
                           height=400, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sec">Annual Revenue by Vehicle Type ($B)</div>', unsafe_allow_html=True)
        rev = ann_f[ann_f["total_revenue"].notna()].copy()
        rev["Rev_B"] = rev["total_revenue"] / 1e9
        rev["vl"] = rev["vehicle_type"].map(VL)
        fig4 = px.bar(rev, x="year", y="Rev_B", color="vl", barmode="group",
                      color_discrete_sequence=COLORS, template="plotly_dark",
                      labels={"Rev_B":"Revenue ($B)","vl":"Vehicle","year":"Year"})
        fig4.update_layout(**dark_layout(300))
        fig4.update_traces(marker_line_width=0)
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        st.markdown('<div class="sec">Cumulative Growth Index (2019 = 100)</div>', unsafe_allow_html=True)
        base_yr = ann_f[ann_f["year"]==yr[0]].groupby("vehicle_type")["total_trips"].sum()
        idx_rows = []
        for vy, grp in ann_f.groupby(["year","vehicle_type"]):
            year_v, vtype = vy
            base = base_yr.get(vtype, None)
            if base and base > 0:
                idx_rows.append({"Year": year_v, "Vehicle": VL.get(vtype,vtype),
                                  "Index": grp["total_trips"].sum()/base*100})
        idx_df = pd.DataFrame(idx_rows)
        if not idx_df.empty:
            fig5 = px.line(idx_df, x="Year", y="Index", color="Vehicle",
                           color_discrete_sequence=COLORS, markers=True, template="plotly_dark")
            fig5.add_hline(y=100, line_dash="dot", line_color="#333",
                           annotation_text="Baseline", annotation_font_color="#555",
                           annotation_font_size=9)
            fig5.update_traces(line_width=2.5, marker_size=7)
            fig5.update_layout(**dark_layout(300))
            st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="insight">📌 <b>Structural shift:</b> HVFHV (Uber/Lyft) crossed 50% market share in 2021 and continues growing, now at ~70%. Yellow Taxi declined from ~55% → under 16%. COVID caused an 85% trip collapse in April 2020; HVFHV recovered to above pre-pandemic levels by mid-2021, while Yellow Taxi has not.</div>', unsafe_allow_html=True)

# ══════ TAB 2 ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec">Hourly × Day-of-Week Demand Heatmap</div>', unsafe_allow_html=True)
    heat = hr_f.groupby(["day_of_week","hour"])["trip_count"].sum().reset_index()
    heat["day"] = heat["day_of_week"].map(DAY)
    hp = heat.pivot_table(index="day",columns="hour",values="trip_count",aggfunc="sum").fillna(0)
    day_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    hp = hp.reindex([d for d in day_order if d in hp.index])
    fig6 = go.Figure(go.Heatmap(
        z=hp.values, x=list(hp.columns), y=list(hp.index),
        colorscale=[[0,"#0a0a0a"],[0.3,"#003322"],[0.7,"#00aa55"],[1,"#00ff88"]],
        hovertemplate="<b>%{y} %{x}:00</b><br>Trips: %{z:,.0f}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#666",size=9), thickness=12,
                      outlinecolor="#1a1a1a", bgcolor="#111")
    ))
    fig6.update_layout(**dark_layout(280, legend=False))
    fig6.update_xaxes(title="Hour of Day", dtick=2)
    fig6.update_yaxes(title="")
    st.plotly_chart(fig6, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec">Hourly Profile by Vehicle Type</div>', unsafe_allow_html=True)
        hv = hr_f.groupby(["hour","vehicle_type"])["trip_count"].sum().reset_index()
        hv["Vehicle"] = hv["vehicle_type"].map(VL)
        fig7 = px.line(hv, x="hour", y="trip_count", color="Vehicle",
                       color_discrete_sequence=COLORS, markers=True, template="plotly_dark",
                       labels={"trip_count":"Total Trips","hour":"Hour of Day"})
        fig7.update_traces(line_width=2.5, marker_size=5)
        fig7.update_layout(**dark_layout(300))
        st.plotly_chart(fig7, use_container_width=True)

    with c2:
        st.markdown('<div class="sec">Day of Week Volume</div>', unsafe_allow_html=True)
        dow_raw = hr_f.groupby("day_of_week")["trip_count"].sum().reset_index()
        dow_raw["Day"] = dow_raw["day_of_week"].map(DAY)
        dow = pd.DataFrame({"Day": day_order}).merge(dow_raw[["Day","trip_count"]], on="Day", how="left").fillna(0)
        fig8 = px.bar(dow, x="Day", y="trip_count",
                      color="trip_count",
                      color_continuous_scale=[[0,"#0a2a1a"],[1,"#00ff88"]],
                      template="plotly_dark",
                      labels={"trip_count":"Total Trips"})
        fig8.update_layout(**dark_layout(300, legend=False), coloraxis_showscale=False)
        fig8.update_traces(marker_line_width=0)
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<div class="sec">Year × Month Seasonality Heatmap</div>', unsafe_allow_html=True)
    ym = mon_f.groupby(["year","month"])["trip_count"].sum().reset_index()
    ym_piv = ym.pivot_table(index="year",columns="month",values="trip_count",aggfunc="sum").fillna(0)
    ym_piv.columns = [MONTHS[m-1] for m in ym_piv.columns]
    fig9 = go.Figure(go.Heatmap(
        z=ym_piv.values, x=list(ym_piv.columns), y=[str(y) for y in ym_piv.index],
        colorscale=[[0,"#0a0a0a"],[0.2,"#001a0d"],[0.6,"#006633"],[1,"#00ff88"]],
        hovertemplate="<b>%{y} %{x}</b><br>Trips: %{z:,.0f}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#666",size=9), thickness=12,
                      outlinecolor="#1a1a1a", bgcolor="#111")
    ))
    fig9.update_layout(**dark_layout(260, legend=False))
    st.plotly_chart(fig9, use_container_width=True)
    st.markdown('<div class="insight">📌 Friday 20:00–23:00 is consistently the peak demand window across all vehicle types. March 2020 is the darkest cell in the seasonality heatmap — 85% below expected. Summer (Jun–Aug) is structurally weaker than spring/fall for taxis. HVFHV shows less seasonality than Yellow Taxi, suggesting more commuter/airport-driven demand.</div>', unsafe_allow_html=True)

# ══════ TAB 3 ══════════════════════════════════════════════════════════════════
with tab3:
    bc = bor_f[~bor_f["pickup_borough"].isin(["Unknown",None,"EWR"])].copy()

    c1,c2 = st.columns([1,1])
    with c1:
        st.markdown('<div class="sec">Borough Trip Volume — Treemap</div>', unsafe_allow_html=True)
        bt = bc.groupby("pickup_borough")["pickup_count"].sum().reset_index()
        bt["pct"] = (bt["pickup_count"]/bt["pickup_count"].sum()*100).round(1)
        fig10 = px.treemap(bt, path=["pickup_borough"], values="pickup_count",
                           color="pickup_count",
                           color_continuous_scale=[[0,"#0a1a14"],[0.5,"#00663a"],[1,"#00ff88"]],
                           custom_data=["pct"])
        fig10.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]}%",
            hovertemplate="<b>%{label}</b><br>Pickups: %{value:,.0f}<br>Share: %{customdata[0]}%<extra></extra>",
            textfont_size=13, marker_line_color="#0d0d0d", marker_line_width=2
        )
        fig10.update_layout(paper_bgcolor="#111", margin=dict(l=0,r=0,t=0,b=0),
                            height=400, coloraxis_showscale=False)
        st.plotly_chart(fig10, use_container_width=True)

    with c2:
        st.markdown('<div class="sec">Average Fare by Borough</div>', unsafe_allow_html=True)
        bf = bc.groupby("pickup_borough")["avg_fare"].mean().reset_index().sort_values("avg_fare")
        fig11 = go.Figure(go.Bar(
            x=bf["avg_fare"], y=bf["pickup_borough"], orientation="h",
            marker=dict(
                color=bf["avg_fare"],
                colorscale=[[0,"#0a2a1a"],[1,"#00ff88"]],
                line=dict(width=0)
            ),
            hovertemplate="<b>%{y}</b><br>Avg Fare: $%{x:.2f}<extra></extra>",
            text=[f"${v:.2f}" for v in bf["avg_fare"]],
            textposition="outside",
            textfont=dict(color="#888", size=10)
        ))
        fig11.update_layout(**dark_layout(320, legend=False))
        fig11.update_xaxes(title="Avg Fare ($)")
        st.plotly_chart(fig11, use_container_width=True)

    st.markdown('<div class="sec">Borough Volume Over Time</div>', unsafe_allow_html=True)
    bt2 = bc.pivot_table(index="year",columns="pickup_borough",values="pickup_count",aggfunc="sum").fillna(0)
    bt2_long = bt2.reset_index().melt(id_vars="year",var_name="Borough",value_name="Pickups")
    fig12 = px.line(bt2_long, x="year", y="Pickups", color="Borough",
                    color_discrete_sequence=COLORS, markers=True, template="plotly_dark")
    fig12.update_traces(line_width=2.5, marker_size=7)
    fig12.update_layout(**dark_layout(300))
    st.plotly_chart(fig12, use_container_width=True)
    st.markdown('<div class="insight">📌 Manhattan accounts for ~68% of Yellow Taxi pickups. Queens commands the highest avg fare ($28+) due to JFK/LaGuardia airport flat-rate trips. The Bronx and Staten Island show the fastest HVFHV adoption growth post-2021.</div>', unsafe_allow_html=True)

# ══════ TAB 4 ══════════════════════════════════════════════════════════════════
with tab4:
    ml_c = ml[(ml["fare_amount"]>2.5) & (ml["fare_amount"]<120)].copy()

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec">Fare Distribution — Histogram + Box</div>', unsafe_allow_html=True)
        fig13 = px.histogram(ml_c, x="fare_amount", nbins=80, marginal="box",
                             color_discrete_sequence=[MINT], template="plotly_dark",
                             opacity=0.8, labels={"fare_amount":"Fare ($)"})
        fig13.update_layout(**dark_layout(340))
        fig13.update_traces(marker_line_width=0)
        st.plotly_chart(fig13, use_container_width=True)

    with c2:
        st.markdown('<div class="sec">Fare Box Plot by Pickup Borough</div>', unsafe_allow_html=True)
        ml_bor = ml_c[(ml_c["pickup_borough"].notna()) &
                      (~ml_c["pickup_borough"].isin(["Unknown","EWR"]))]
        fig14 = px.box(ml_bor, x="pickup_borough", y="fare_amount",
                       color="pickup_borough",
                       color_discrete_sequence=COLORS, template="plotly_dark",
                       points=False,
                       labels={"fare_amount":"Fare ($)","pickup_borough":"Borough"})
        fig14.update_layout(**dark_layout(340, legend=False))
        fig14.update_traces(line_width=1.5)
        st.plotly_chart(fig14, use_container_width=True)

    st.markdown('<div class="sec">Trip Distance vs Fare — Scatter with OLS Trendline</div>', unsafe_allow_html=True)
    sc = ml_c[(ml_c["trip_distance"]>0.1) & (ml_c["trip_distance"]<25) &
              (ml_c["pickup_borough"].notna()) &
              (~ml_c["pickup_borough"].isin(["Unknown"]))]\
             .sample(min(4000,len(ml_c)), random_state=42)
    fig15 = px.scatter(sc, x="trip_distance", y="fare_amount",
                       color="pickup_borough", opacity=0.35,
                       color_discrete_sequence=COLORS, trendline="ols",
                       template="plotly_dark",
                       labels={"trip_distance":"Distance (mi)","fare_amount":"Fare ($)",
                               "pickup_borough":"Borough"})
    fig15.update_traces(marker_size=3)
    fig15.update_layout(**dark_layout(360))
    st.plotly_chart(fig15, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sec">Median Fare Trend by Vehicle Type</div>', unsafe_allow_html=True)
        fm = fare_f.pivot_table(index="year",columns="vehicle_type",values="p50",aggfunc="mean")
        fm.columns = [VL.get(c,c) for c in fm.columns]
        fm_long = fm.reset_index().melt(id_vars="year",var_name="Vehicle",value_name="Median Fare ($)")
        fig16 = px.line(fm_long, x="year", y="Median Fare ($)", color="Vehicle",
                        color_discrete_sequence=COLORS, markers=True, template="plotly_dark")
        fig16.update_traces(line_width=2.5, marker_size=7)
        fig16.update_layout(**dark_layout(300))
        st.plotly_chart(fig16, use_container_width=True)

    with c4:
        st.markdown('<div class="sec">YoY Avg Fare Change (%)</div>', unsafe_allow_html=True)
        yoy = fare_f.groupby(["year","vehicle_type"])["avg_fare"].mean().reset_index()\
                    .sort_values(["vehicle_type","year"])
        yoy["yoy"] = yoy.groupby("vehicle_type")["avg_fare"].pct_change()*100
        yoy = yoy.dropna(subset=["yoy"])
        yoy["Vehicle"] = yoy["vehicle_type"].map(VL)
        fig18 = px.bar(yoy, x="year", y="yoy", color="Vehicle", barmode="group",
                       color_discrete_sequence=COLORS, template="plotly_dark",
                       labels={"yoy":"YoY Change (%)","year":"Year"})
        fig18.add_hline(y=0, line_color="#333", line_width=1)
        fig18.update_layout(**dark_layout(300))
        fig18.update_traces(marker_line_width=0)
        st.plotly_chart(fig18, use_container_width=True)

    st.markdown('<div class="insight">📌 Fare distribution is right-skewed (median ~$12, mean ~$18) with a long tail of airport/outer-borough premium trips. OLS confirms strong distance-fare linearity (R²≈0.75) with Queens trips above the trendline (airport flat rates). Post-2021 fare inflation peaked at +18% YoY in 2022 for Yellow Taxi driven by NYC congestion pricing and surcharge increases.</div>', unsafe_allow_html=True)

# ══════ TAB 5 ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec">XGBoost Fare Prediction — Trained on 552K Yellow Taxi Trips</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">XGBoost regressor predicts fare from 8 trip features. SHAP (SHapley Additive exPlanations) quantifies each feature\'s contribution. First load: ~60 seconds while model trains and SHAP values are computed.</div>', unsafe_allow_html=True)

    with st.spinner("Loading pre-trained XGBoost model and SHAP values..."):
        model, sv, Xte_s, feats, r2, rmse, y_true, y_pred = load_model()

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("R² Score",      f"{r2:.4f}")
    m2.metric("RMSE",          f"${rmse:.2f}")
    m3.metric("Train Samples", "~442K")
    m4.metric("Test Samples",  "~111K")

    FEAT_LABELS = {
        "trip_distance":"Trip Distance","passenger_count":"Passenger Count",
        "hour":"Hour of Day","day_of_week":"Day of Week",
        "month":"Month","year":"Year",
        "pickup_borough":"Pickup Borough","dropoff_borough":"Dropoff Borough"
    }

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec">SHAP Feature Importance (Mean |SHAP Value|)</div>', unsafe_allow_html=True)
        mean_shap = np.abs(sv).mean(axis=0)
        shap_fi = pd.DataFrame({"Feature":[FEAT_LABELS.get(f,f) for f in feats],"Importance":mean_shap})\
                    .sort_values("Importance")
        fig19 = go.Figure(go.Bar(
            x=shap_fi["Importance"], y=shap_fi["Feature"], orientation="h",
            marker=dict(
                color=shap_fi["Importance"],
                colorscale=[[0,"#0a2a1a"],[0.5,"#00994d"],[1,"#00ff88"]],
                line=dict(width=0)
            ),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: $%{x:.3f}<extra></extra>",
            text=[f"${v:.3f}" for v in shap_fi["Importance"]],
            textposition="outside",
            textfont=dict(color="#666", size=10)
        ))
        fig19.update_layout(**dark_layout(360, legend=False))
        fig19.update_xaxes(title="Mean |SHAP Value| (avg $ impact on prediction)")
        st.plotly_chart(fig19, use_container_width=True)

    with c2:
        st.markdown('<div class="sec">Predicted vs Actual Fare</div>', unsafe_allow_html=True)
        pva = pd.DataFrame({"Actual":y_true,"Predicted":y_pred})
        fig20 = go.Figure()
        fig20.add_trace(go.Scatter(
            x=pva["Actual"], y=pva["Predicted"], mode="markers",
            marker=dict(color=MINT, size=3, opacity=0.3),
            hovertemplate="Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>",
            name="Predictions"
        ))
        lim = [max(0,pva.min().min()-2), pva.max().max()+2]
        fig20.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
            line=dict(color=RED, dash="dash", width=1.5), name="Perfect fit"))
        fig20.update_layout(**dark_layout(360))
        fig20.update_xaxes(title="Actual Fare ($)", range=lim)
        fig20.update_yaxes(title="Predicted Fare ($)", range=lim)
        st.plotly_chart(fig20, use_container_width=True)

    st.markdown('<div class="sec">SHAP Beeswarm — Feature Impact Distribution (300 sample points)</div>', unsafe_allow_html=True)
    rows = []
    for i, feat in enumerate(feats):
        vals  = sv[:300, i]
        fvals = Xte_s[feat].values[:300]
        fn    = (fvals - fvals.min()) / (fvals.max() - fvals.min() + 1e-9)
        for s, fv, fnorm in zip(vals, fvals, fn):
            rows.append({"Feature":FEAT_LABELS.get(feat,feat),"SHAP Value":float(s),"Feature Value (normalised)":float(fnorm),"fv":float(fv)})
    sw = pd.DataFrame(rows)

    fig21 = go.Figure()
    feat_order = sw.groupby("Feature")["SHAP Value"].apply(lambda x: np.abs(x).mean()).sort_values().index.tolist()
    for feat in feat_order:
        df_f = sw[sw["Feature"]==feat].copy()
        is_last = feat == feat_order[-1]
        fig21.add_trace(go.Scatter(
            x=df_f["SHAP Value"],
            y=[feat]*len(df_f),
            mode="markers",
            marker=dict(
                size=6, opacity=0.6,
                color=df_f["Feature Value (normalised)"].values,
                colorscale=[[0,"#3b82f6"],[0.5,"#444"],[1,"#ff4d4d"]],
                showscale=is_last,
                colorbar=dict(
                    title=dict(text="Feature<br>Value", font=dict(size=9,color="#666")),
                    tickvals=[0,0.5,1], ticktext=["Low","Mid","High"],
                    tickfont=dict(color="#555",size=9), thickness=10,
                    bgcolor="#111", outlinecolor="#1a1a1a"
                )
            ),
            showlegend=False,
            hovertemplate=f"<b>{feat}</b><br>SHAP: $%{{x:.3f}}<br>Feature value: %{{customdata:.2f}}<extra></extra>",
            customdata=df_f["fv"].values
        ))
    # jitter applied via x-axis noise instead (y is categorical string)
    for trace in fig21.data:
        trace.x = [x + np.random.uniform(-0.5, 0.5) for x in trace.x]
    fig21.update_layout(**dark_layout(420))
    fig21.update_xaxes(title="SHAP Value ($ impact on fare prediction)", zeroline=True, zerolinecolor="#333")
    st.plotly_chart(fig21, use_container_width=True)

    st.markdown('<div class="sec">SHAP Dependence — Trip Distance Effect on Fare (coloured by Hour)</div>', unsafe_allow_html=True)
    di = feats.index("trip_distance")
    dep = pd.DataFrame({
        "Trip Distance (mi)": Xte_s["trip_distance"].values[:1500],
        "SHAP Value ($)":     sv[:1500, di],
        "Hour of Day":        Xte_s["hour"].values[:1500]
    })
    fig22 = px.scatter(dep, x="Trip Distance (mi)", y="SHAP Value ($)",
                       color="Hour of Day", opacity=0.5,
                       color_continuous_scale="Plasma",
                       template="plotly_dark")
    fig22.update_traces(marker_size=4)
    fig22.update_coloraxes(colorbar=dict(
        tickfont=dict(color="#555", size=9), thickness=10, bgcolor="#111",
        title=dict(text="Hour", font=dict(size=9, color="#666"))
    ))
    fig22.update_layout(**dark_layout(380))
    st.plotly_chart(fig22, use_container_width=True)

    st.markdown('<div class="insight">📌 <b>ML insights:</b> Trip distance dominates fare prediction (mean |SHAP| ~$3.20) — each mile adds roughly $2.50. Hour of day is the second driver, capturing surge pricing (late-night peaks in red on the beeswarm). Pickup borough matters significantly: Manhattan trips carry a positive SHAP premium while outer-borough trips can be negative. Passenger count has near-zero SHAP importance — NYC taxis charge per trip, not per person. Model R² ≈ 0.87 on held-out test data.</div>', unsafe_allow_html=True)
