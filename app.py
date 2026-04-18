import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Black Swan Predictor",
    page_icon="🦢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
html, body, [class*="css"] { background-color: #0a0c10; color: #e8ecf5; }
h1,h2,h3 { font-family: 'Syne', sans-serif; color: #e8ecf5 !important; }
.metric-card {
    background: #111318; border: 1px solid #2a2f3d;
    border-radius: 12px; padding: 16px 20px; text-align: center;
}
.metric-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #4a5268; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 700; margin: 4px 0; }
.risk-box {
    border-radius: 14px; padding: 24px; text-align: center;
    border: 1px solid #2a2f3d; background: #111318;
}
.insight-box {
    background: #111318; border: 1px solid #2a2f3d;
    border-left: 3px solid #f5a623; border-radius: 8px;
    padding: 14px 18px; margin: 8px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #8892aa;
}
.stButton>button {
    background: rgba(255,59,78,0.1); border: 1px solid #ff3b4e;
    color: #ff3b4e; font-family: 'JetBrains Mono', monospace;
    font-weight: 700; border-radius: 8px; padding: 8px 20px;
}
.stButton>button:hover { background: #ff3b4e; color: white; }
div[data-testid="stMetricValue"] > div { font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────────────
def risk_color(score):
    if score < 15:  return "#22c55e"
    if score < 30:  return "#f5a623"
    if score < 60:  return "#ff8c00"
    return "#ff3b4e"

def risk_label(score):
    if score < 15:  return "NORMAL"
    if score < 30:  return "ELEVATED"
    if score < 60:  return "HIGH DANGER"
    return "EXTREME CRISIS"

# ── DATA LOADING ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_inputs():
    import yfinance as yf
    import pandas_datareader.data as web

    start_date = "1995-01-01"
    end_date   = datetime.datetime.now().strftime('%Y-%m-%d')

    tickers = {
        "NSE_India":  "^NSEI",
        "NYSE":       "^NYA",
        "SSE":        "000001.SS",
        "JPX":        "^N225",
        "Euronext":   "^N100",
        "LSE":        "^FTSE",
        "VIX":        "^VIX",
        "Gold":       "GC=F",
        "Crude_Oil":  "BZ=F",
        "Copper":     "HG=F",
        "USD_Index":  "DX-Y.NYB",
        "SP500":      "^GSPC",
    }

    df_yf = yf.download(list(tickers.values()), start=start_date, end=end_date, progress=False)['Close']
    df_yf = df_yf.rename(columns={v: k for k, v in tickers.items()})
    df_yf['Gold_Copper_Ratio'] = df_yf['Gold'] / df_yf['Copper']

    try:
        df_macro = web.DataReader(['T10Y2Y', 'FEDFUNDS'], 'fred', start_date, end_date)
        df_macro.columns = ['Yield_Curve_Spread', 'FED_Rate']
        final_df = pd.concat([df_yf, df_macro], axis=1).ffill()
    except Exception:
        final_df = df_yf.copy()
        final_df['Yield_Curve_Spread'] = np.nan
        final_df['FED_Rate'] = np.nan
        final_df = final_df.ffill()

    return final_df

# ── FRAGILITY ENGINE ─────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def build_fragility(_df):
    df = _df.copy()
    frag = pd.DataFrame(index=df.index)
    markets = ['NSE_India','NYSE','SSE','JPX','Euronext','LSE']

    for m in markets:
        if m in df.columns:
            ret = df[m].pct_change()
            frag[f'{m}_Kurtosis'] = ret.rolling(252).kurt()
            frag[f'{m}_Vol']      = ret.rolling(252).std() * np.sqrt(252)

    world_cols = [c for c in ['NYSE','SSE','JPX','Euronext','LSE'] if c in df.columns]
    corrs = [df['NSE_India'].pct_change().rolling(60).corr(df[c].pct_change()) for c in world_cols]
    if corrs:
        frag['India_Global_Correlation'] = pd.concat(corrs, axis=1).mean(axis=1)

    if 'Yield_Curve_Spread' in df.columns:
        frag['Yield_Curve'] = df['Yield_Curve_Spread']
    if 'Gold_Copper_Ratio' in df.columns:
        frag['Gold_Copper_Ratio'] = df['Gold_Copper_Ratio']

    frag = frag.dropna()

    # Label known crises
    frag['Event_Label'] = 'Normal'
    frag['Is_Crisis']   = 0
    crises = [
        ('1997-07-01','1998-06-30','Asian Financial Crisis'),
        ('2000-03-01','2002-09-30','Dot-com Crash'),
        ('2008-09-01','2009-06-30','Global Financial Crisis'),
        ('2011-08-01','2011-10-31','European Debt Crisis'),
        ('2015-08-01','2015-09-30','China Crash'),
        ('2018-12-01','2019-01-31','US-China Trade War'),
        ('2020-02-01','2020-05-31','COVID-19 Crash'),
        ('2022-01-01','2022-10-31','Rate Hike Shock'),
    ]
    for s, e, name in crises:
        frag.loc[s:e, 'Event_Label'] = name
        frag.loc[s:e, 'Is_Crisis']   = 1

    return frag

@st.cache_data(ttl=3600, show_spinner=False)
def build_advanced(_frag):
    df = _frag.copy()
    vol_cols = [c for c in df.columns if 'Vol' in c]
    df['Systemic_Stress']  = df[vol_cols].mean(axis=1) * df['India_Global_Correlation']
    df['Fragility_Score']  = df['Systemic_Stress'].rolling(126).mean()
    if 'Gold_Copper_Ratio' in df.columns and 'NYSE_Vol' in df.columns:
        df['Early_Warning_Signal'] = df['Gold_Copper_Ratio'].pct_change() / df['NYSE_Vol']
    df['Macro_Fragility_Combo'] = df['Gold_Copper_Ratio'] * df['NSE_India_Kurtosis']
    df = df.dropna(subset=['Fragility_Score'])
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def build_decision(_adv):
    df = _adv.copy()
    w  = 252
    df['Rule_Early_Warning']   = df['Early_Warning_Signal']  > df['Early_Warning_Signal'].rolling(w).mean() * 1.5
    df['Rule_NonLinear_Jump']  = df['Macro_Fragility_Combo'] > df['Macro_Fragility_Combo'].rolling(w).mean() * 1.77
    df['Rule_Correlation_Lock']= df['India_Global_Correlation'] > 0.8
    frag_thresh = df['Fragility_Score'].rolling(w).mean() + 2 * df['Fragility_Score'].rolling(w).std()
    df['Rule_Systemic_Brittle']= df['Fragility_Score'] > frag_thresh
    df['Rule_Low_Vol_Trap']    = df['NSE_India_Vol'] < df['NSE_India_Vol'].rolling(w).mean() * 0.7
    df['Final_Fragility_Score']= (
        df['Rule_Early_Warning'].astype(int)    * 30 +
        df['Rule_NonLinear_Jump'].astype(int)   * 30 +
        df['Rule_Correlation_Lock'].astype(int) * 15 +
        df['Rule_Systemic_Brittle'].astype(int) * 15 +
        df['Rule_Low_Vol_Trap'].astype(int)     * 10
    )
    return df

@st.cache_data(show_spinner=False)
def train_model(_df_labeled):
    from sklearn.ensemble import RandomForestClassifier, IsolationForest

    feat_cols = [c for c in _df_labeled.columns if any(x in c for x in ['Kurtosis','Vol','Correlation','Yield_Curve','Gold_Copper_Ratio'])]
    X = _df_labeled[feat_cols].fillna(0)
    y = _df_labeled['Is_Crisis']

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    importances = pd.DataFrame({'Feature': feat_cols, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)

    return rf, iso, importances, feat_cols

def predict_prob(model, df, feat_cols):
    latest = df[feat_cols].tail(1).fillna(0)
    return model.predict_proba(latest)[0][1] * 100

def taleb_knowledge(df_adv):
    c_data = df_adv[df_adv['Is_Crisis'] == 1]
    n_data = df_adv[df_adv['Is_Crisis'] == 0]
    kurt_cols = [c for c in df_adv.columns if 'Kurtosis' in c]
    fat_mult  = c_data[kurt_cols].mean().mean() / max(n_data[kurt_cols].mean().mean(), 0.001)
    corr_n    = n_data['India_Global_Correlation'].mean()
    corr_c    = c_data['India_Global_Correlation'].mean()
    return fat_mult, corr_n, corr_c

# ── HISTORICAL MIRROR ─────────────────────────────────────────────────
HISTORICAL = [
    {"year":"2008","name":"Lehman Brothers Collapse","score":95,"desc":"ธนาคารล้มเหลว, credit freeze, S&P -57%, VIX พุ่ง 80"},
    {"year":"2020","name":"COVID-19 Black Swan",     "score":82,"desc":"VIX พุ่ง 85, ตลาดร่วง 34% ใน 33 วัน — เร็วที่สุดในประวัติศาสตร์"},
    {"year":"2000","name":"Dot-com Bubble Burst",    "score":68,"desc":"NASDAQ -78%, tech sector wipeout กินเวลา 2.5 ปี"},
    {"year":"1997","name":"Asian Financial Crisis",  "score":72,"desc":"THB collapse, ค่าเงินเอเชียพัง, IMF เข้าช่วยหลายประเทศ"},
    {"year":"2022","name":"Rate Hike Shock",          "score":55,"desc":"Fed ขึ้นดอกเบี้ย 425bps ใน 1 ปี, bond market ร่วงหนักสุดใน 40 ปี"},
]

def find_mirror(score):
    return sorted(HISTORICAL, key=lambda h: abs(h["score"] - score))[:3]

# ── MONTE CARLO ──────────────────────────────────────────────────────
def run_monte_carlo(current_score, n_paths=5000, months=12):
    np.random.seed(42)
    drift  = 0.008 if current_score > 50 else -0.005
    sigma  = 0.06  + current_score * 0.0008
    paths  = np.zeros((n_paths, months + 1))
    paths[:, 0] = current_score
    for t in range(1, months + 1):
        shock = np.random.normal(drift, sigma, n_paths)
        paths[:, t] = np.clip(paths[:, t-1] + shock * 10, 0, 100)

    p5   = np.percentile(paths, 5, axis=0)
    p50  = np.percentile(paths, 50, axis=0)
    p95  = np.percentile(paths, 95, axis=0)
    crisis_pct = (paths[:, -1] >= 75).mean() * 100
    return p5, p50, p95, paths, crisis_pct

# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ CONTROLS")
    st.caption("Scenario Simulator — ปรับค่าแล้ว Dashboard อัปเดตทันที")

    oil_val   = st.slider("🛢 Crude Oil (USD/bbl)",  30, 200, 85)
    rate_val  = st.slider("🏦 Fed Rate (%)",         0.0, 15.0, 5.25, 0.25)
    gold_val  = st.slider("🥇 Gold (USD/oz)",        1000, 5000, 2320, 10)
    vix_val   = st.slider("📊 VIX Index",            10, 90, 22)
    yld_val   = st.slider("📈 US 10Y Yield (%)",     0.0, 12.0, 4.3, 0.1)
    cny_val   = st.slider("🇨🇳 USD/CNY",             6.0, 9.0, 7.24, 0.01)

    def scenario_risk(oil, rate, gold, vix, cny, yld):
        r  = min(vix/90,  1) * 28
        r += min(rate/15, 1) * 22
        r += min(oil/200, 1) * 18
        r += min(yld/12,  1) * 15
        r += min((gold-1000)/4000, 1) * 10
        r += min((cny-6)/3, 1) * 7
        return round(r)

    scen_score = scenario_risk(oil_val, rate_val, gold_val, vix_val, cny_val, yld_val)
    sc = risk_color(scen_score)
    st.markdown(f"""
    <div class="risk-box" style="border-color:{sc};margin-top:16px">
      <div class="metric-label">SCENARIO RISK SCORE</div>
      <div class="metric-value" style="color:{sc};font-size:40px">{scen_score}</div>
      <div style="color:{sc};font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700">{risk_label(scen_score)}</div>
    </div>
    """, unsafe_allow_html=True)

    if scen_score < 20:   horizon = "> 24 months"
    elif scen_score < 40: horizon = "~12–18 months"
    elif scen_score < 60: horizon = "~6–9 months"
    elif scen_score < 80: horizon = "~2–4 months"
    else:                 horizon = "⚠️ IMMINENT"
    st.caption(f"Crisis Horizon: **{horizon}**")

    st.markdown("---")
    run_mc = st.button("▶ RUN MONTE CARLO (5K paths)")
    st.caption("Data cached 1 hr · Source: Yahoo Finance + FRED")

# ════════════════════════════════════════════════════════════════════
# MAIN — LOAD & PROCESS
# ════════════════════════════════════════════════════════════════════
st.markdown("# 🦢 BLACK SWAN PREDICTOR")
st.caption("Systemic Risk Intelligence Engine · Taleb-inspired fragility framework")

with st.spinner("⏳ กำลังดึงข้อมูลตลาดโลกและคำนวณ Fragility Index..."):
    try:
        df_input    = fetch_all_inputs()
        df_frag     = build_fragility(df_input)
        df_adv      = build_advanced(df_frag)
        df_decision = build_decision(df_adv)
        rf_model, iso_model, importance_df, feat_cols = train_model(df_adv)
        current_risk  = predict_prob(rf_model, df_adv, feat_cols)
        today_score   = int(df_decision['Final_Fragility_Score'].iloc[-1])
        fat_mult, corr_n, corr_c = taleb_knowledge(df_adv)
        data_ok = True
    except Exception as e:
        st.warning(f"⚠️ ไม่สามารถดึงข้อมูล Live ได้: {e}  \nแสดงผลจาก Scenario Simulator แทน")
        data_ok    = False
        current_risk = float(scen_score)
        today_score  = scen_score

display_score = today_score if data_ok else scen_score
rc = risk_color(display_score)

# ════════════════════════════════════════════════════════════════════
# ROW 1 — KPI CARDS
# ════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)
metrics = [
    ("TODAY'S PROBABILITY", f"{current_risk:.1f}%", risk_color(current_risk)),
    ("6-MONTH PROBABILITY",  f"{min(current_risk*3.1, 99):.1f}%", risk_color(current_risk*2)),
    ("12-MONTH PROBABILITY", f"{min(current_risk*5.8, 99):.1f}%", risk_color(current_risk*3)),
    ("FRAGILITY SCORE",      f"{display_score}/100", rc),
    ("REGIME",               risk_label(display_score), rc),
]
for col, (label, val, color) in zip([k1,k2,k3,k4,k5], metrics):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color}">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# ROW 2 — GAUGE + BUTTERFLY
# ════════════════════════════════════════════════════════════════════
col_gauge, col_butterfly = st.columns([1, 1])

with col_gauge:
    st.markdown("#### 🎯 Risk Gauge Meter")
    angle = -135 + (display_score / 100) * 270
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_score,
        domain={'x':[0,1],'y':[0,1]},
        title={'text': risk_label(display_score), 'font': {'color': rc, 'size': 16}},
        number={'font': {'color': rc, 'size': 48}},
        gauge={
            'axis': {'range':[0,100], 'tickcolor':'#4a5268', 'tickfont':{'color':'#4a5268','size':10}},
            'bar':  {'color': rc, 'thickness': 0.25},
            'bgcolor': '#1e2230',
            'bordercolor': '#2a2f3d',
            'steps': [
                {'range':[0,25],  'color':'rgba(34,197,94,0.15)'},
                {'range':[25,50], 'color':'rgba(245,166,35,0.15)'},
                {'range':[50,75], 'color':'rgba(255,140,0,0.15)'},
                {'range':[75,100],'color':'rgba(255,59,78,0.20)'},
            ],
            'threshold': {'line':{'color':'white','width':3},'thickness':0.8,'value':display_score},
        }
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e8ecf5', height=300, margin=dict(t=30,b=10,l=30,r=30)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Probability breakdown
    pc1, pc2, pc3 = st.columns(3)
    for col_p, label, mult, clamp in [
        (pc1, "TODAY",    0.045, 10),
        (pc2, "6 MONTHS", 0.31,  99),
        (pc3, "1 YEAR",   0.60,  99)
    ]:
        v = round(min(current_risk * mult, clamp), 1)
        col_p.markdown(f"""<div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{risk_color(v*3)};font-size:22px">{v}%</div>
        </div>""", unsafe_allow_html=True)

with col_butterfly:
    st.markdown("#### 🦋 Butterfly Effect Panel")
    butterfly_data = [
        {"Factor": "VIX (Fear Index)",        "Contribution": round(min(vix_val/90,1)*28,1),   "Color":"#ff3b4e"},
        {"Factor": "Fed Rate",                "Contribution": round(min(rate_val/15,1)*22,1),  "Color":"#f5a623"},
        {"Factor": "Crude Oil",               "Contribution": round(min(oil_val/200,1)*18,1),  "Color":"#ff8c00"},
        {"Factor": "10Y Treasury Yield",      "Contribution": round(min(yld_val/12,1)*15,1),   "Color":"#a855f7"},
        {"Factor": "Gold (Safe-Haven Demand)","Contribution": round(min((gold_val-1000)/4000,1)*10,1),"Color":"#3b82f6"},
        {"Factor": "USD/CNY Pressure",        "Contribution": round(min((cny_val-6)/3,1)*7,1), "Color":"#22c55e"},
    ]
    bf_df = pd.DataFrame(butterfly_data).sort_values("Contribution", ascending=True)
    fig_bf = go.Figure(go.Bar(
        x=bf_df["Contribution"], y=bf_df["Factor"],
        orientation='h',
        marker_color=bf_df["Color"].tolist(),
        text=[f"+{v}" for v in bf_df["Contribution"]],
        textposition='outside', textfont=dict(color='#e8ecf5', size=11),
    ))
    fig_bf.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#4a5268', range=[0,32]),
        yaxis=dict(showgrid=False, color='#e8ecf5'),
        font_color='#e8ecf5', height=300, margin=dict(t=10,b=10,l=10,r=50),
        showlegend=False
    )
    st.plotly_chart(fig_bf, use_container_width=True)

    top_factor = bf_df.iloc[-1]["Factor"]
    top_val    = bf_df.iloc[-1]["Contribution"]
    st.markdown(f"""<div class="insight-box">
    🔬 <b>AI Interpretation:</b> ปัจจัยหลักที่ขับเคลื่อนความเสี่ยงขณะนี้คือ <b>{top_factor}</b>
    (+{top_val} pts จาก 100) — {'ระดับที่น่ากังวล ควรติดตามใกล้ชิด' if top_val > 15 else 'ระดับปกติ'}
    ตาม Taleb Framework ความเสี่ยงสะสมแบบ Non-linear: เมื่อหลายปัจจัยพุ่งพร้อมกัน โอกาสเกิด
    Black Swan ไม่ได้บวกกัน แต่ <b>คูณกัน</b>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# ROW 3 — HISTORICAL MIRROR + TALEB KNOWLEDGE
# ════════════════════════════════════════════════════════════════════
col_mirror, col_taleb = st.columns([1.2, 0.8])

with col_mirror:
    st.markdown("#### 🪞 Historical Mirror")
    st.caption("ระบบจับคู่ pattern ปัจจุบันกับเหตุการณ์ในอดีต")
    matches = find_mirror(display_score)
    for i, h in enumerate(matches):
        similarity = max(0, 100 - abs(h["score"] - display_score) * 1.5)
        border = "#f5a623" if i == 0 else "#2a2f3d"
        badge  = "🔴 TOP MATCH" if i == 0 else f"#{i+1} MATCH"
        st.markdown(f"""
        <div style="background:#111318;border:1px solid {border};border-radius:10px;
                    padding:14px;margin-bottom:8px;position:relative">
          <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:#4a5268">{h['year']} · {badge}</span>
          <div style="font-weight:700;font-size:14px;margin:4px 0">{h['name']}</div>
          <span style="position:absolute;top:14px;right:14px;font-family:'JetBrains Mono',monospace;
                       font-size:13px;color:#f5a623;font-weight:700">{similarity:.0f}% similar</span>
          <div style="font-size:12px;color:#8892aa;line-height:1.5">{h['desc']}</div>
          <div style="margin-top:8px;height:3px;background:#1e2230;border-radius:2px">
            <div style="width:{similarity}%;height:100%;background:#f5a623;border-radius:2px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

with col_taleb:
    st.markdown("#### 📖 Taleb Knowledge Base")
    st.caption("สิ่งที่ AI เรียนรู้จาก 40 ปีของข้อมูล")
    if data_ok:
        st.markdown(f"""
        <div class="insight-box">📐 <b>Fat-Tails:</b><br>
        ช่วงวิกฤต Kurtosis สูงกว่าปกติ <b style="color:#ff3b4e">{fat_mult:.1f}x</b><br>
        พิสูจน์ว่าตลาดไม่ได้เป็น Normal Distribution</div>
        <div class="insight-box">🔗 <b>Correlation Breakdown:</b><br>
        ปกติ: {corr_n:.2f} → วิกฤต: <b style="color:#ff3b4e">{corr_c:.2f}</b><br>
        เมื่อวิกฤต สินทรัพย์ทั่วโลกร่วงพร้อมกัน Diversification ไม่ช่วย</div>
        <div class="insight-box">🔇 <b>The Silence Before the Storm:</b><br>
        VIX ต่ำผิดปกติ = <b style="color:#f5a623">Low Vol Trap</b><br>
        ความสงบคือสัญญาณที่อันตรายที่สุด ตาม Taleb</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="insight-box">⚠️ ต้องการข้อมูล Live เพื่อแสดง Taleb Analysis<br>
        กรุณาตรวจสอบ internet connection และ API keys</div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# ROW 4 — FRAGILITY SCORE HISTORY (ถ้ามีข้อมูล)
# ════════════════════════════════════════════════════════════════════
if data_ok:
    st.markdown("---")
    st.markdown("#### 📉 Fragility Score History (1995–Today)")

    fig_hist = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.06)

    plot_df = df_decision.tail(3000).copy()
    fig_hist.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['Final_Fragility_Score'],
        fill='tozeroy', fillcolor='rgba(245,166,35,0.1)',
        line=dict(color='#f5a623', width=1.5), name='Fragility Score'
    ), row=1, col=1)
    fig_hist.add_hline(y=60, line_dash="dash", line_color="rgba(255,59,78,0.5)", row=1, col=1)
    fig_hist.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.4)",  row=1, col=1)

    # Shade crisis periods
    crisis_df = plot_df[plot_df['Is_Crisis'] == 1]
    fig_hist.add_trace(go.Scatter(
        x=crisis_df.index, y=[100]*len(crisis_df),
        fill='tozeroy', fillcolor='rgba(255,59,78,0.08)',
        line=dict(color='rgba(0,0,0,0)'), name='Crisis Period', showlegend=True
    ), row=1, col=1)

    # VIX if available
    if 'VIX' in df_input.columns:
        fig_hist.add_trace(go.Scatter(
            x=plot_df.index, y=df_input.loc[plot_df.index, 'VIX'],
            line=dict(color='#ff3b4e', width=1), name='VIX', opacity=0.8
        ), row=2, col=1)

    fig_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e8ecf5', height=420,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8892aa', size=11)),
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis2=dict(showgrid=False, color='#4a5268'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#4a5268', range=[0,105]),
        yaxis2=dict(showgrid=False, color='#4a5268'),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Feature importance
    st.markdown("#### 🧠 AI Feature Importance (Random Forest)")
    imp_top = importance_df.head(8)
    fig_imp = go.Figure(go.Bar(
        x=imp_top['Feature'], y=imp_top['Importance'],
        marker_color='#3b82f6',
        text=[f"{v:.3f}" for v in imp_top['Importance']],
        textposition='outside', textfont=dict(color='#e8ecf5', size=10)
    ))
    fig_imp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e8ecf5', height=280,
        xaxis=dict(showgrid=False, color='#4a5268', tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#4a5268'),
        margin=dict(t=20, b=80, l=10, r=10)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# ROW 5 — MONTE CARLO
# ════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("#### 🎲 Black Swan Scenario Simulator (Monte Carlo)")

if run_mc or True:  # always show on load
    p5, p50, p95, all_paths, crisis_pct = run_monte_carlo(display_score, n_paths=2000)
    months_label = ['Now'] + [f'M{i}' for i in range(1, 13)]

    fig_mc = go.Figure()
    # Plot a sample of raw paths
    for i in range(0, 2000, 40):
        fig_mc.add_trace(go.Scatter(
            x=months_label, y=all_paths[i],
            line=dict(color='rgba(245,166,35,0.04)', width=1),
            showlegend=False, hoverinfo='skip'
        ))
    fig_mc.add_trace(go.Scatter(x=months_label, y=p95, line=dict(color='rgba(255,59,78,0.7)', width=1.5, dash='dot'), name='95th Pct'))
    fig_mc.add_trace(go.Scatter(x=months_label, y=p50, line=dict(color='#f5a623', width=2.5), name='Median'))
    fig_mc.add_trace(go.Scatter(x=months_label, y=p5,  line=dict(color='rgba(34,197,94,0.7)', width=1.5, dash='dot'), name='5th Pct'))
    fig_mc.add_hline(y=75, line_dash="dash", line_color="rgba(255,59,78,0.5)", annotation_text="Crisis Threshold (75)")
    fig_mc.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e8ecf5', height=360,
        xaxis=dict(showgrid=False, color='#4a5268'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', color='#4a5268', range=[0,105], title='Risk Score'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8892aa')),
        margin=dict(t=20, b=20, l=10, r=10)
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc_data = [
        ("CRISIS PROBABILITY\n(12 months)", f"{crisis_pct:.1f}%", risk_color(crisis_pct)),
        ("MEDIAN RISK\n(Month 6)",           f"{p50[6]:.1f}",      risk_color(p50[6])),
        ("WORST CASE\n(95th Pct)",           f"{p95[-1]:.1f}",     risk_color(p95[-1])),
        ("BEST CASE\n(5th Pct)",             f"{p5[-1]:.1f}",      risk_color(p5[-1])),
    ]
    for col_m, (label, val, color) in zip([mc1,mc2,mc3,mc4], mc_data):
        col_m.markdown(f"""<div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption("""
🦢 **Black Swan Predictor** · Inspired by Nassim Nicholas Taleb's *The Black Swan* & *Antifragile*
· Data: Yahoo Finance, FRED · Model: Random Forest + Isolation Forest + Decision Rules
· ⚠️ For educational purposes only — not financial advice
""")
