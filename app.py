# ╔══════════════════════════════════════════════════════════╗
# ║   Black Swan Event Prediction Model                      ║
# ╚══════════════════════════════════════════════════════════╝
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Swan Event Prediction Model",
    page_icon="🦢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Anuphan:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #F8FAFC !important;
    font-family: 'Anuphan', sans-serif;
}

/* ── Section header ── */
.sec-header {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 22px; border-radius: 12px;
    margin: 32px 0 20px 0;
    border-left: 5px solid;
}
.sec-header.blue  { background:#EFF6FF; border-color:#3B82F6; }
.sec-header.green { background:#F0FDF4; border-color:#10B981; }
.sec-header.violet{ background:#F5F3FF; border-color:#8B5CF6; }
.sec-tag   { font-size:11px; font-weight:600; letter-spacing:.1em;
             text-transform:uppercase; opacity:.6; }
.sec-title { font-size:1.35rem; font-weight:600; margin:0; }

/* ── Cards ── */
.card {
    background:#ffffff; border-radius:14px; padding:20px 22px;
    border:1px solid #E2E8F0; box-shadow:0 2px 6px rgba(0,0,0,.05);
    height:100%;
}
.card-dark {
    background:#0F172A; border-radius:14px; padding:20px 22px;
    color:#F1F5F9;
}

/* ── Metric tile ── */
.stMetric {
    background:#ffffff; padding:18px; border-radius:12px;
    border:1px solid #E2E8F0; box-shadow:0 2px 4px rgba(0,0,0,.04);
}

/* ── Status box ── */
.status-box {
    text-align:center; padding:22px 18px; border-radius:12px;
    background:white; border:1px solid #E2E8F0;
    box-shadow:0 2px 6px rgba(0,0,0,.05);
}

/* ── Divider ── */
.swan-hr { border:none; border-top:1px solid #E2E8F0; margin:30px 0; }

/* ── Probability pill ── */
.prob-pill {
    display:inline-block; padding:6px 16px; border-radius:20px;
    font-weight:600; font-size:.85rem; margin:4px;
}

/* ── Shake animation for panic duck ── */
.shake { animation: shake 0.5s infinite; display:inline-block; }
@keyframes shake {
    0%   { transform: translate(1px,1px)  rotate(0deg); }
    10%  { transform: translate(-1px,-2px) rotate(-1deg); }
    20%  { transform: translate(-3px,0px)  rotate(1deg); }
    50%  { transform: translate(0px,2px)   rotate(0deg); }
    100% { transform: translate(1px,-2px)  rotate(-1deg); }
}

/* ── App title bar ── */
.app-title-bar {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
    border-radius: 16px; padding: 28px 36px; margin-bottom: 28px;
    display: flex; align-items: center; justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# BACKEND — shared functions (เหมือนเดิมทุก logic)
# ════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data():
    """Section 1 — long-horizon data since 1975."""
    tickers = {
        'NSE_India': '^NSEI', 'NYSE': '^NYA', 'SSE': '000001.SS',
        'JPX': '^N225', 'Euronext': '^N100', 'LSE': '^FTSE',
        'VIX': '^VIX', 'Gold': 'GC=F', 'Crude_Oil': 'BZ=F',
        'Copper': 'HG=F', 'USD_Index': 'DX-Y.NYB',
        '10Y_Bond': '^TNX', '2Y_Bond': '^IRX',
    }
    df = yf.download(list(tickers.values()), start="1975-01-01", progress=False)['Close']
    if df.empty:
        return None, None, None, None
    df = df.ffill().bfill()
    df = df.rename(columns={v: k for k, v in tickers.items()})
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    price_cols  = ['NSE_India','NYSE','SSE','JPX','Euronext','LSE','Gold','Crude_Oil','USD_Index']
    valid_cols  = [c for c in price_cols if c in df.columns]
    df_norm     = df[valid_cols].copy()
    for col in df_norm.columns:
        first = df_norm[col].dropna().iloc[0]
        df_norm[col] = (df_norm[col] / first) * 100
    return df, df_norm, latest, prev


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data():
    """Section 2 — recent data for metric calculation."""
    tickers = {
        "NSE_India": "^NSEI", "NYSE": "^NYA",  "SSE": "000001.SS",
        "JPX": "^N225",       "Euronext": "^N100", "LSE": "^FTSE",
        "Gold": "GC=F",       "Copper": "HG=F",  "SP500": "^GSPC",
        "10Y_Yield": "^TNX",  "2Y_Yield": "^IRX",
    }
    df = yf.download(list(tickers.values()), start="2022-01-01", progress=False)['Close']
    df = df.ffill().bfill()
    df = df.rename(columns={v: k for k, v in tickers.items()})
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_realtime_data():
    """Section 3 — sandbox baseline."""
    tickers = {
        "VIX": "^VIX", "10Y": "^TNX", "2Y": "^IRX",
        "Gold": "GC=F", "Copper": "HG=F", "SP500": "^GSPC",
    }
    df = yf.download(list(tickers.values()), period="2y", progress=False)['Close']
    df = df.ffill().bfill()
    vol         = df['^VIX'].iloc[-1] / 100
    yield_spread= (df['^TNX'] - df['^IRX']).iloc[-1]
    returns     = df['^GSPC'].pct_change().dropna()
    kurt        = returns.rolling(252).kurt().iloc[-1]
    coupling    = 0.45
    gold_copper = (df['GC=F'] / df['HG=F']).iloc[-1]
    return vol, yield_spread, coupling, kurt, gold_copper


def calculate_metrics(df):
    markets = ['NYSE','SP500','Euronext','LSE','JPX','SSE','NSE_India']
    returns = df[markets].pct_change().dropna()
    kurt    = returns.rolling(252).kurt().mean(axis=1).iloc[-1]
    vol     = (returns.rolling(252).std().mean(axis=1) * np.sqrt(252)).iloc[-1]
    corr_m  = returns.tail(60).corr()
    coupling= corr_m.where(
        np.triu(np.ones(corr_m.shape), k=1).astype(bool)
    ).stack().mean()
    yield_spread = (df['10Y_Yield'] - df['2Y_Yield']).iloc[-1]
    stress  = (vol * 0.3389 + abs(yield_spread/100) * 0.2450
               + coupling * 0.1463 + (kurt/15) * 0.1411)
    index   = float(np.clip(stress / 0.5 * 100, 2.5, 98.5))
    return index, stress


def estimate_black_swan_mc(stress, horizon_days=30, simulations=50000):
    baseline_daily_prob = 1 / 5000
    threshold = 0.0549
    if stress > threshold:
        risk_factor = np.power(stress / threshold, 1.15)
    else:
        risk_factor = stress / threshold
    draws = np.random.random((simulations, horizon_days))
    return float((np.any(draws < (baseline_daily_prob * risk_factor), axis=1).sum()
                  / simulations) * 100)


def get_stress_score(v, y, c, k):
    return (v * 0.3389 + abs(y/100) * 0.2450 + c * 0.1463 + (k/15) * 0.1411)


def risk_color(p):
    if p < 5:   return "#10B981"
    if p < 15:  return "#F59E0B"
    return "#EF4444"

def risk_bg(p):
    if p < 5:   return "#ECFDF5"
    if p < 15:  return "#FFF7ED"
    return "#FEF2F2"

def gauge_status(idx):
    if idx >= 70: return "CRITICAL", "#EF4444", "ระบบมีความเปราะบางสูงมาก เสี่ยงต่อการเกิดภาวะ Black Swan"
    if idx >= 35: return "ELEVATED", "#F59E0B", "ระบบมีความเครียดสะสมเหนือระดับปกติ ควรเพิ่มความระมัดระวัง"
    return "NORMAL", "#10B981", "สภาวะตลาดโลกมีความยืดหยุ่นสูง ความเสี่ยงเชิงระบบอยู่ในเกณฑ์ต่ำ"


# ════════════════════════════════════════════════════════════
# ── APP TITLE BAR ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════
import datetime
now_str = datetime.datetime.now().strftime("%d %b %Y · %H:%M UTC")

st.markdown(f"""
<div class="app-title-bar">
  <div>
    <div style="font-size:1.9rem;font-weight:700;color:#F1F5F9;letter-spacing:.01em">
      🦢 Black Swan Event Prediction Model
    </div>
    <div style="color:#94A3B8;font-size:.95rem;margin-top:6px">
      Systemic Risk Intelligence · Long-horizon Data (Since 1975) · Monte Carlo Engine
    </div>
  </div>
  <div style="text-align:right">
    <div style="color:#94A3B8;font-size:.82rem;font-family:monospace">{now_str}</div>
    <div style="color:#38BDF8;font-size:.82rem;margin-top:4px;font-family:monospace">
      ⟳ Data refreshes every 1 hr
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# FETCH ALL DATA ONCE
# ════════════════════════════════════════════════════════════
with st.spinner("กำลังดึงข้อมูลตลาดโลก…"):
    df_raw, df_norm, latest, prev = fetch_historical_data()
    hist_ok = df_raw is not None

with st.spinner("กำลังคำนวณ Systemic Stress…"):
    try:
        data2       = get_market_data()
        risk_index, stress_today = calculate_metrics(data2)
        monthly_trend = 0.015
        p_today = estimate_black_swan_mc(stress_today)
        p_3m    = estimate_black_swan_mc(stress_today + (monthly_trend * 3 * 0.85))
        p_6m    = estimate_black_swan_mc(stress_today + (monthly_trend * 6 * 0.70))
        pred_ok = True
    except Exception as e:
        st.error(f"⚠️ Section 2 data error: {e}")
        pred_ok = False

with st.spinner("กำลังโหลด Sandbox baseline…"):
    try:
        v_real, y_real, c_real, k_real, g_real = get_realtime_data()
        stress_real   = get_stress_score(v_real, y_real, c_real, k_real)
        p_real_today  = estimate_black_swan_mc(stress_real)
        p_real_3m     = estimate_black_swan_mc(stress_real + 0.012)
        p_real_6m     = estimate_black_swan_mc(stress_real + 0.025)
        sandbox_ok    = True
    except Exception as e:
        st.warning(f"⚠️ Sandbox baseline error: {e}")
        sandbox_ok = False


# ════════════════════════════════════════════════════════════
# SECTION 1 — LIVE WATCHTOWER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-header blue">
  <div>
    <div class="sec-tag">Section 1</div>
    <p class="sec-title">🔭 Live Data Watchtower</p>
  </div>
</div>
""", unsafe_allow_html=True)

if hist_ok:

    # ── Row A: Global Equity (6 metrics) ─────────────────────────────
    st.markdown("**🌎 Global Equity Indices**")
    market_units = {
        'NSE_India':'INR — Nifty 50',   'NYSE':'USD — Composite',
        'SSE':'CNY — Composite',         'JPX':'JPY — Nikkei 225',
        'Euronext':'EUR — Enext 100',    'LSE':'GBP — FTSE 100',
    }
    eq_cols = st.columns(6)
    for col, (key, label) in zip(eq_cols, market_units.items()):
        val  = latest.get(key, 0)
        pval = prev.get(key, 1)
        chg  = ((val - pval) / pval * 100) if pval else 0
        col.metric(label, f"{val:,.0f}", f"{chg:+.2f}%")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Row B: Commodities + VIX (4 metrics) ─────────────────────────
    st.markdown("**📊 Commodities & Macro Indicators**")
    cm1, cm2, cm3, cm4, cm5 = st.columns(5)
    cm1.metric("Gold (XAU/USD)",     f"${latest['Gold']:,.2f}",      "USD / oz")
    cm2.metric("Brent Crude",        f"${latest['Crude_Oil']:.2f}",  "USD / bbl")
    cm3.metric("USD Index",          f"{latest['USD_Index']:.2f}",   "Points")
    cm4.metric("VIX Index",          f"{latest['VIX']:.2f}",
               f"{latest['VIX']-prev['VIX']:+.2f}", delta_color="inverse")
    if '10Y_Bond' in latest.index and '2Y_Bond' in latest.index:
        spread = latest['10Y_Bond'] - latest['2Y_Bond']
        cm5.metric("Yield Spread (10Y−2Y)", f"{spread:.2f}%",
                   "Inverted ⚠️" if spread < 0 else "Normal ✓")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Row C: Long-term chart ────────────────────────────────────────
    with st.expander("📈 Long-term Growth Comparison (Since 1975 — Normalized, Log Scale)", expanded=True):
        st.caption("ดัชนีบางตัวอาจเริ่มแสดงผลช้ากว่าปี 1975 ตามวันที่มีข้อมูลครั้งแรกในระบบ")
        fig_lt = go.Figure()
        colors = ["#3B82F6","#10B981","#F59E0B","#EF4444",
                  "#8B5CF6","#06B6D4","#F97316","#64748B","#EC4899"]
        for i, col in enumerate(df_norm.columns):
            fig_lt.add_trace(go.Scatter(
                x=df_norm.index, y=df_norm[col], name=col,
                line=dict(width=1.4, color=colors[i % len(colors)])
            ))
        fig_lt.update_layout(
            yaxis_type="log", template="plotly_white",
            hovermode="x unified", height=520,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
            legend=dict(orientation="h", y=1.08, font=dict(size=11)),
            yaxis=dict(title="Growth Index (Base 100)", gridcolor="#F1F5F9"),
            xaxis=dict(title="Year", gridcolor="#F1F5F9"),
            margin=dict(t=10, b=40, l=10, r=10),
        )
        st.plotly_chart(fig_lt, use_container_width=True)

else:
    st.error("ไม่สามารถดึงข้อมูลตลาดได้ กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")


# ════════════════════════════════════════════════════════════
# SECTION 2 — PREDICTION
# ════════════════════════════════════════════════════════════
st.markdown("<hr class='swan-hr'>", unsafe_allow_html=True)
st.markdown("""
<div class="sec-header green">
  <div>
    <div class="sec-tag">Section 2</div>
    <p class="sec-title">🔮 Prediction of Black Swan Event</p>
  </div>
</div>
""", unsafe_allow_html=True)

if pred_ok:
    status, s_color, s_desc = gauge_status(risk_index)

    # ── Left: Gauge · Right: Forecast ────────────────────────────────
    col_gauge, col_forecast = st.columns([1, 1], gap="large")

    with col_gauge:
        st.markdown("**📊 Global Systemic Risk Index**")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_index,
            number={'font': {'size': 72, 'color': '#1E293B'}, 'valueformat': '.1f'},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1,
                         'tickcolor': '#94A3B8'},
                'bar':  {'color': s_color, 'thickness': 0.25},
                'bgcolor': 'white',
                'bordercolor': '#E2E8F0',
                'steps': [
                    {'range': [0,  35], 'color': '#D1FAE5'},
                    {'range': [35, 70], 'color': '#FEF9C3'},
                    {'range': [70, 100],'color': '#FEE2E2'},
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 4},
                    'thickness': 0.8, 'value': 90,
                },
            },
        ))
        fig_gauge.update_layout(
            height=360, margin=dict(t=20, b=0, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div class="status-box" style="border-top:5px solid {s_color}">
          <h2 style="color:{s_color};margin:0;font-weight:700">{status}</h2>
          <p style="color:#64748B;margin-top:8px;font-size:1rem">{s_desc}</p>
          <p style="color:#94A3B8;font-size:.85rem;margin-top:6px">
            Systemic Stress Index: <strong>{stress_today:.4f}</strong>
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col_forecast:
        st.markdown("**🔮 Black Swan Probability Forecast**")

        # Probability cards
        fc1, fc2, fc3 = st.columns(3)
        for col_f, (lbl, val, base) in zip(
            [fc1, fc2, fc3],
            [("Today", p_today, None),
             ("3M Forward", p_3m, p_today),
             ("6M Forward", p_6m, p_today)],
        ):
            clr  = risk_color(val)
            bg   = risk_bg(val)
            delta_str = f"{val-base:+.2f}%" if base is not None else ""
            col_f.markdown(f"""
            <div style="background:{bg};border:1.5px solid {clr};border-radius:12px;
                        padding:18px;text-align:center">
              <div style="font-size:.8rem;color:#64748B;font-weight:600;
                          text-transform:uppercase;letter-spacing:.06em">{lbl}</div>
              <div style="font-size:2.2rem;font-weight:700;color:{clr};margin:6px 0">
                {val:.2f}%
              </div>
              <div style="font-size:.82rem;color:{clr}">{delta_str}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Probability path chart
        df_path = pd.DataFrame({
            "Timeline": ["Today", "3M Forward", "6M Forward"],
            "Probability": [p_today, p_3m, p_6m],
        })
        fig_path = go.Figure()
        fig_path.add_trace(go.Scatter(
            x=df_path["Timeline"], y=df_path["Probability"],
            mode="lines+markers+text",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(size=10, color="#3B82F6",
                        line=dict(color="white", width=2)),
            text=[f"{v:.2f}%" for v in df_path["Probability"]],
            textposition="top center",
            fill="tozeroy", fillcolor="rgba(59,130,246,.08)",
        ))
        fig_path.update_layout(
            height=220, template="plotly_white",
            yaxis=dict(title="Probability (%)", range=[0, max(p_6m*1.4, 5)],
                       gridcolor="#F1F5F9"),
            xaxis=dict(gridcolor="#F1F5F9"),
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_path, use_container_width=True)

        with st.expander("ℹ️ Methodology Insight"):
            st.caption("""
- **Daily Baseline:** อิงจากสถิติเหตุการณ์หายากระดับโลก (1 ใน 5,000 วันทำการ)
- **Monte Carlo Engine:** จำลองเหตุการณ์ตลาดอนาคต 50,000 รูปแบบในแต่ละจุดเวลา
- **Power Law Scaling:** ปรับระดับความเสี่ยงตามความเครียดเชิงระบบแบบ Non-linear
- **Mean Reversion:** การพยากรณ์ระยะยาวรวมสมมติฐานการปรับตัวของกลไกตลาด
            """)

else:
    st.warning("ไม่สามารถคำนวณ Prediction ได้ กรุณาตรวจสอบข้อมูล")


# ════════════════════════════════════════════════════════════
# SECTION 3 — SIMULATION PROBABILITY SANDBOX
# ════════════════════════════════════════════════════════════
st.markdown("<hr class='swan-hr'>", unsafe_allow_html=True)
st.markdown("""
<div class="sec-header violet">
  <div>
    <div class="sec-tag">Section 3</div>
    <p class="sec-title">🎮 Simulation Probability Sandbox</p>
  </div>
</div>
""", unsafe_allow_html=True)

if sandbox_ok:

    # ── Controls (2 rows × 3 cols) ────────────────────────────────────
    st.markdown("**⚙️ ปรับตัวแปรเพื่อจำลองสถานการณ์**")

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        s_vol  = st.slider("📊 Volatility (VIX)",       0.05, 0.90, float(round(v_real, 3)))
        s_kurt = st.slider("📐 Kurtosis (Fat-Tail)",    0.0,  20.0, float(round(k_real, 2)))
    with ctrl2:
        s_yield = st.slider("📈 Yield Spread",         -1.50,  1.50, float(round(y_real, 3)))
        s_gold  = st.slider("🥇 Gold/Copper Ratio",   200.0, 1000.0, float(round(g_real, 1)))
    with ctrl3:
        s_coupling = st.slider("🌐 Global Coupling",    0.0,   1.0, float(c_real))
        butterfly  = st.checkbox("🦋 Activate Butterfly Effect (Surprise Shock)")
        if 'chaos_val' not in st.session_state:
            st.session_state.chaos_val = float(np.random.uniform(1.4, 2.5))
        chaos_mult = st.session_state.chaos_val if butterfly else 1.0

    # ── Calculate simulation ──────────────────────────────────────────
    stress_sim  = get_stress_score(s_vol, s_yield, s_coupling, s_kurt) * chaos_mult
    p_sim_today = estimate_black_swan_mc(stress_sim)
    p_sim_3m    = estimate_black_swan_mc(stress_sim + 0.015)
    p_sim_6m    = estimate_black_swan_mc(stress_sim + 0.030)

    bg_sim  = risk_bg(p_sim_today)
    clr_sim = risk_color(p_sim_today)

    if butterfly:
        st.warning(f"🦋 Butterfly Effect Active — Stress multiplied ×{chaos_mult:.2f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Result panel ──────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:{bg_sim};border:2px solid {clr_sim};"
        f"border-radius:18px;padding:24px 28px'>",
        unsafe_allow_html=True,
    )

    duck_col, result_col = st.columns([1, 2], gap="large")

    with duck_col:
        # Duck GIF by risk level
        if p_sim_today < 5:
            tenor_id, label = "15568846810302620355", "🦢 Happy Duck"
            text_color = clr_sim
            shake_cls  = ""
        elif p_sim_today < 15:
            tenor_id, label = "13982082229451252813", "😰 Anxious Duck"
            text_color = clr_sim
            shake_cls  = ""
        else:
            tenor_id, label = "25805348", "🚨 PANIC DUCK!"
            text_color = "#EF4444"
            shake_cls  = "shake"

        tenor_html = f"""
        <style>
          body{{margin:0;padding:0;display:flex;justify-content:center;
               align-items:center;background:transparent}}
          .tenor-gif-embed{{max-width:100% !important}}
        </style>
        <div class="{shake_cls}">
          <div class="tenor-gif-embed"
               data-postid="{tenor_id}"
               data-share-method="host"
               data-aspect-ratio="1.0"
               data-width="100%">
          </div>
        </div>
        <script type="text/javascript" async
          src="https://tenor.com/embed.js"></script>
        """
        components.html(tenor_html, height=260)
        st.markdown(
            f"<div style='text-align:center;font-size:1.05rem;font-weight:600;"
            f"color:{text_color};margin-top:-8px'>{label}</div>",
            unsafe_allow_html=True,
        )

    with result_col:
        st.markdown(
            f"<div style='font-size:1.6rem;font-weight:700;color:{clr_sim};"
            f"margin-bottom:16px'>Simulated Risk: {p_sim_today:.2f}%</div>",
            unsafe_allow_html=True,
        )

        # 3 metric tiles comparing sim vs real
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Sim — Today",    f"{p_sim_today:.2f}%",
                   delta=f"{p_sim_today-p_real_today:+.2f}%", delta_color="inverse")
        sm2.metric("Sim — 3M",       f"{p_sim_3m:.2f}%",
                   delta=f"{p_sim_3m-p_real_3m:+.2f}%",   delta_color="inverse")
        sm3.metric("Sim — 6M",       f"{p_sim_6m:.2f}%",
                   delta=f"{p_sim_6m-p_real_6m:+.2f}%",   delta_color="inverse")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Heatmap ───────────────────────────────────────────────────
        vol_range   = np.linspace(max(0.05, s_vol - 0.2),  min(0.9, s_vol + 0.2),  10)
        yield_range = np.linspace(s_yield - 1.0, s_yield + 1.0, 10)
        z_prob = []
        for y in yield_range:
            row = []
            for v in vol_range:
                s = get_stress_score(v, y, s_coupling, s_kurt) * chaos_mult
                row.append(estimate_black_swan_mc(s))
            z_prob.append(row)

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_prob,
            x=np.round(vol_range, 2),
            y=np.round(yield_range, 2),
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Risk %", thickness=14),
        ))
        fig_heat.add_trace(go.Scatter(
            x=[s_vol], y=[s_yield],
            mode='markers+text',
            marker=dict(color='white', size=14, symbol='star',
                        line=dict(color='black', width=2)),
            text=["YOU ARE HERE"], textposition="top center",
            name="Current",
        ))
        fig_heat.update_layout(
            title=dict(text="Risk Sensitivity Landscape (Volatility vs Yield Spread)",
                       font=dict(size=13)),
            xaxis_title="Volatility (Panic Level)",
            yaxis_title="Yield Spread",
            height=310,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)   # close result panel

else:
    st.warning("ไม่สามารถโหลด Sandbox baseline ได้")


# ── Footer ────────────────────────────────────────────────────────────
st.markdown("<hr class='swan-hr'>", unsafe_allow_html=True)
st.caption(
    "🦢 Black Swan Event Prediction Model  ·  "
    "Data: Yahoo Finance  ·  "
    "Framework: Antifragile Quantitative Risk (Taleb)  ·  "
    "⚠️ For educational purposes only — not financial advice"
)
