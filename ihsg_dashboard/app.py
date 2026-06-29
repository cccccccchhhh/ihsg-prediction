"""
app.py - IHSG Next-Day Direction Dashboard (Random Forest only)
Streamlit Cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import os

from pipeline import (
    fetch_prices, fetch_bei_stocks, compute_whi,
    build_features, build_lag_features, select_features,
    temporal_split, tune_and_train, retrain_on_trainval,
    save_models, load_models, predict_tomorrow,
    MODEL_PATH
)

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="IHSG Direction Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Custom CSS -----------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e2b96f; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a0aec0; margin: 0.3rem 0 0; font-size: 1rem; }

    .pred-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid;
        margin-bottom: 1rem;
    }
    .pred-hijau  { background: #f0fff4; border-color: #38a169; }
    .pred-merah  { background: #fff5f5; border-color: #e53e3e; }
    .pred-netral { background: #f7fafc; border-color: #718096; }

    .info-box {
        background: #eef2ff;
        border-left: 4px solid #5a67d8;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #434190;
    }

    .stAlert { border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ===============================================================================
# SIDEBAR
# ===============================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Bursa_Efek_Indonesia.svg/320px-Bursa_Efek_Indonesia.svg.png",
              width=160)
    st.markdown("### ⚙️ Konfigurasi")

    retrain_mode = st.selectbox(
        "Mode Model",
        ["Gunakan model tersimpan (cepat)", "Retrain ulang (lambat ~5 menit)"],
        index=0
    )
    force_retrain = retrain_mode.startswith("Retrain")

    st.markdown("---")
    st.markdown("### 📅 Info Pipeline")
    st.markdown("""
    <div class='info-box'>
    <b>Train:</b> Jan 2023 – Des 2024<br>
    <b>Val:</b> Jan – Mar 2025<br>
    <b>Test:</b> Apr – Jun 2025<br>
    <b>Predict:</b> Hari berikutnya dari data terbaru
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Metodologi")
    st.markdown("""
    - **Features:** 4 layer (Macro, Technical, Confluence, Regime)
    - **Selection:** Spearman + Mutual Information + VIF
    - **Lag:** Brute-force 1–10 hari
    - **Model:** Random Forest (terbaik dari evaluasi LR/RF/ANN)
    - **Tuning:** GridSearchCV + Threshold F1
    """)

    run_btn = st.button("🚀 Jalankan Pipeline", type="primary", use_container_width=True)

# -- Header -----------------------------------------------------------------------
st.markdown("""
<div class='main-header'>
    <h1>📈 IHSG Next-Day Direction Predictor</h1>
    <p>Macro + Technical + Cross-Market Confluence | Random Forest Model</p>
</div>
""", unsafe_allow_html=True)

# ===============================================================================
# QUICK MARKET SNAPSHOT (selalu tampil, tanpa run)
# ===============================================================================

st.markdown("### 🌍 Market Snapshot Hari Ini")


@st.cache_data(ttl=900)
def get_snapshot():
    snap_tickers = {
        'IHSG': '^JKSE', 'S&P500': '^GSPC', 'NIKKEI': '^N225',
        'FTSE': '^FTSE', 'VIX': '^VIX', 'USD/IDR': 'IDR=X',
        'GOLD': 'GC=F', 'WTI': 'CL=F', 'DXY': 'DX-Y.NYB',
    }
    rows = {}
    for name, ticker in snap_tickers.items():
        try:
            df = yf.download(ticker, period='5d', progress=False, auto_adjust=True)
            if len(df) >= 2:
                last = float(df['Close'].iloc[-1])
                prev = float(df['Close'].iloc[-2])
                chg = (last - prev) / prev * 100
                rows[name] = {'price': last, 'chg': chg}
        except Exception:
            pass
    return rows


snap = get_snapshot()
cols = st.columns(len(snap))
for col, (name, data) in zip(cols, snap.items()):
    col.metric(
        label=name,
        value=f"{data['price']:,.2f}",
        delta=f"{data['chg']:+.2f}%",
    )

st.markdown("---")

# ===============================================================================
# MAIN PIPELINE - hanya jalan saat tombol diklik
# ===============================================================================

if run_btn:

    with st.status("⏳ Menjalankan pipeline prediksi...", expanded=True) as status:

        st.write("📡 Fetching macro & global data dari Yahoo Finance...")
        prices, failed = fetch_prices(start='2023-01-01')
        if failed:
            st.warning(f"Gagal download: {failed}")
        st.write(f"  ✅ {len(prices.columns)} instrumen · {len(prices)} hari trading")

        st.write("📊 Fetching 100 saham BEI untuk WHI...")
        bei_prices = fetch_bei_stocks(start='2023-01-01')
        whi_series, ath_cnt, atl_cnt = compute_whi(bei_prices)
        st.write(f"  ✅ WHI computed · {whi_series.notna().sum()} hari valid")

        st.write("🔧 Feature engineering (4 layer)...")
        feat = build_features(prices, whi_series, ath_cnt, atl_cnt)
        st.write(f"  ✅ {feat.shape[1]} base features")

        st.write("⏱ Building lag features (lag 1–10)...")
        analysis_df, base_features = build_lag_features(feat)
        st.write(f"  ✅ {len(base_features) * 10} kombinasi · {len(analysis_df)} obs setelah dropna")

        st.write("🔍 Feature selection (Spearman + MI + VIF)...")
        final_features, model_df, combined_df = select_features(analysis_df)
        st.write(f"  ✅ {len(final_features)} fitur terpilih")

        models_exist = os.path.exists(f'{MODEL_PATH}models.pkl')

        if models_exist and not force_retrain:
            st.write("📦 Loading model tersimpan...")
            final_models, scaler, saved_features, best_thresh = load_models()
            final_features = saved_features
            st.write(f"  ✅ Model loaded · {len(final_features)} features")
        else:
            st.write("🏋️ Training Random Forest (GridSearchCV + threshold tuning)...")
            (X_tr, y_tr, X_va, y_va, X_te, y_te,
             scaler, dates_val, dates_test) = temporal_split(model_df, final_features)

            st.write(f"  Train: {len(y_tr)} obs | Val: {len(y_va)} obs | Test: {len(y_te)} obs")

            tuned_models, best_params, best_thresh = tune_and_train(X_tr, y_tr, X_va, y_va)
            final_models = retrain_on_trainval(tuned_models, best_params, X_tr, y_tr, X_va, y_va)
            save_models(final_models, scaler, final_features, best_thresh)
            st.write("  ✅ Model trained & saved")
            st.write(f"  Best threshold: {best_thresh}")

        st.write("🔮 Predicting tomorrow...")
        pred_results, x_row = predict_tomorrow(final_models, scaler, final_features, best_thresh, feat)
        status.update(label="✅ Pipeline selesai!", state="complete", expanded=False)

    # ===========================================================================
    # HASIL PREDIKSI (Random Forest only)
    # ===========================================================================

    tomorrow = (datetime.today() + timedelta(days=1)).strftime('%A, %d %B %Y')
    st.markdown(f"## 🔮 Prediksi IHSG — {tomorrow}")

    r = pred_results['Random Forest']
    pct = r['probability'] * 100
    is_hijau = r['prediction'] == 1
    card_class = 'pred-hijau' if is_hijau else 'pred-merah'
    emoji = '🟢' if is_hijau else '🔴'

    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        st.markdown(f"""
        <div class='pred-card {card_class}'>
            <h3 style='margin:0; color:#2d3748; font-size:1rem'>Random Forest</h3>
            <div style='font-size:4rem; margin:0.5rem 0'>{emoji}</div>
            <div style='font-size:1.8rem; font-weight:bold; color:{"#276749" if is_hijau else "#9b2335"}'>
                {r['label']}
            </div>
            <div style='color:#718096; margin-top:0.5rem'>
                P(hijau) = <b>{pct:.1f}%</b><br>
                Threshold = {r['threshold']:.2f}<br>
                Confidence = {r['confidence']:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -- Probability gauge --------------------------------------------------------
    st.markdown("### 📊 Probability Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': '%', 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#27ae60' if is_hijau else '#e74c3c'},
            'steps': [
                {'range': [0, 40], 'color': '#fde8e8'},
                {'range': [40, 60], 'color': '#faf0e6'},
                {'range': [60, 100], 'color': '#e8f5e9'},
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 3},
                'thickness': 0.8,
                'value': r['threshold'] * 100
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # -- IHSG Historical Chart -----------------------------------------------------
    st.markdown("### 📈 IHSG — 6 Bulan Terakhir")

    @st.cache_data(ttl=3600)
    def get_ihsg_history():
        df = yf.download('^JKSE', period='6mo', progress=False, auto_adjust=True)
        return df

    ihsg_hist = get_ihsg_history()

    fig_ihsg = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                              shared_xaxes=True, vertical_spacing=0.05)

    fig_ihsg.add_trace(go.Scatter(
        x=ihsg_hist.index, y=ihsg_hist['Close'].squeeze(),
        mode='lines', name='IHSG Close', line=dict(color='#2c3e50', width=2)
    ), row=1, col=1)

    ema20 = ihsg_hist['Close'].squeeze().ewm(span=20).mean()
    fig_ihsg.add_trace(go.Scatter(
        x=ihsg_hist.index, y=ema20, mode='lines', name='EMA 20',
        line=dict(color='#e67e22', width=1.5, dash='dash')
    ), row=1, col=1)

    daily_ret = ihsg_hist['Close'].squeeze().pct_change() * 100
    fig_ihsg.add_trace(go.Bar(
        x=ihsg_hist.index, y=daily_ret,
        marker_color=['#27ae60' if v >= 0 else '#e74c3c' for v in daily_ret],
        name='Daily Return %', opacity=0.7
    ), row=2, col=1)

    fig_ihsg.update_layout(
        height=450, showlegend=True,
        margin=dict(t=20, b=20, l=10, r=10),
        paper_bgcolor='white', plot_bgcolor='white',
        xaxis2_title='Tanggal', yaxis_title='IHSG Level', yaxis2_title='Return (%)',
    )
    fig_ihsg.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
    fig_ihsg.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig_ihsg, use_container_width=True)

    # -- Feature Values Hari Ini ---------------------------------------------------
    st.markdown("### 🔍 Feature Values (Input ke Model Hari Ini)")

    x_display = x_row.T.reset_index()
    x_display.columns = ['Feature', 'Value']
    x_display['Value'] = x_display['Value'].round(4)

    def color_val(v):
        if isinstance(v, float):
            if v > 0:
                return 'color: #276749'
            elif v < 0:
                return 'color: #9b2335'
        return ''

    st.dataframe(
        x_display.style.applymap(color_val, subset=['Value']),
        use_container_width=True, height=300
    )

    # -- Feature Importance (RF) ----------------------------------------------------
    if 'Random Forest' in final_models:
        st.markdown("### 🏆 Feature Importance (Random Forest)")
        rf = final_models['Random Forest']
        imp = pd.Series(rf.feature_importances_, index=final_features).nlargest(20).sort_values()

        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation='h',
            marker_color='#3498db', opacity=0.85
        ))
        fig_imp.update_layout(
            height=max(350, len(imp) * 22),
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title='Importance', paper_bgcolor='white', plot_bgcolor='white',
        )
        fig_imp.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        st.plotly_chart(fig_imp, use_container_width=True)

    # -- WHI Chart -------------------------------------------------------------------
    st.markdown("### 🌀 Whitehole Index (WHI) — Retail Flush Indicator")
    whi_recent = whi_series.dropna().tail(120)

    fig_whi = make_subplots(rows=2, cols=1, row_heights=[0.4, 0.6],
                             shared_xaxes=True, vertical_spacing=0.05)

    ihsg_recent = prices['IHSG'].reindex(whi_recent.index).dropna()
    fig_whi.add_trace(go.Scatter(
        x=ihsg_recent.index, y=ihsg_recent.values,
        mode='lines', name='IHSG', line=dict(color='#2c3e50', width=2)
    ), row=1, col=1)

    whi_colors = ['#e74c3c' if v > 4000 else '#7f8c8d' for v in whi_recent]
    fig_whi.add_trace(go.Bar(
        x=whi_recent.index, y=whi_recent.values,
        marker_color=whi_colors, name='WHI', opacity=0.75
    ), row=2, col=1)
    fig_whi.add_hline(y=4000, line_dash='dash', line_color='red',
                       annotation_text='Threshold 4000 (Retail Flush)', row=2, col=1)

    fig_whi.update_layout(
        height=400, showlegend=True,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='white', plot_bgcolor='white',
    )
    st.plotly_chart(fig_whi, use_container_width=True)

    # -- Footer -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align:center; color:#a0aec0; font-size:0.85rem'>
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M WIB')} &nbsp;|&nbsp;
        Model: Random Forest &nbsp;|&nbsp;
        Data: Yahoo Finance &nbsp;|&nbsp;
        WHI: Satrio (2025)
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Klik **Jalankan Pipeline** di sidebar untuk memulai prediksi harian.")

    st.markdown("### 📐 Arsitektur Pipeline")
    st.markdown("""
    ```
    Yahoo Finance (23 ticker)          Yahoo Finance (100 saham BEI)
            │                                       │
            ▼                                       ▼
    Layer 1: Macro Returns               Whitehole Index (WHI)
    Layer 2: Technical Indicators  ──────────────┘
    Layer 3: Cross-Market Confluence
    Layer 4: Macro Regime Indicators
            │
            ▼
    Brute-Force Lag (1–10 hari)
    Spearman + Mutual Information Selection
    VIF Filter (threshold = 10)
            │
            ▼
    ┌──────────────────────────┬──────────────┬─────────────┐
    │ TRAIN (Jan23–Des24)      │ VAL (Q1'25)  │ TEST (Q2'25)│
    └──────────────────────────┴──────────────┴─────────────┘
            │
    GridSearchCV (inner TimeSeries 3-fold)
    Threshold Sweep (F1 optimization, val set)
    Refit pada Train+Val
            │
            ▼
    Random Forest
            │
            ▼
    🔮 Prediksi: IHSG besok HIJAU atau MERAH?
    ```
    """)
