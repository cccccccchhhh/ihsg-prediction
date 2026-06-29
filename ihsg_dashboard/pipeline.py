"""
pipeline.py
Core ML pipeline: data fetching -> feature engineering -> train/predict
Faithful port dari ihsg_v6_with_save.ipynb. RF-only.

Catatan penting (fixed dari versi sebelumnya):
- TIDAK pakai pandas-ta lagi (package itu yang bikin install gagal di Streamlit
  Cloud karena bug numpy.NaN). Semua indikator teknikal (RSI, MACD, BBANDS, ATR,
  EMA) di-reimplement manual pakai pandas/numpy, formula standar yang sama
  persis dengan default pandas_ta (Wilder smoothing utk RSI/ATR).
- EM_stress_score, RI_commodity_basket, IHSG_NIKKEI_corr20, ticker UST_2Y
  dibenarkan agar identik dengan notebook (sebelumnya beda -> prediksi salah).
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

# == Config (identik dengan notebook) =========================================
TICKERS = {
    'IHSG'    : '^JKSE',
    'NIKKEI'  : '^N225',
    'HSI'     : '^HSI',
    'STI'     : '^STI',
    'KOSPI'   : '^KS11',
    'SHANGHAI': '000001.SS',
    'SP500'   : '^GSPC',
    'DJI'     : '^DJI',
    'NASDAQ'  : '^IXIC',
    'FTSE'    : '^FTSE',
    'DAX'     : '^GDAXI',
    'MSCI_EM' : 'EEM',
    'USDIDR'  : 'IDR=X',
    'DXY'     : 'DX-Y.NYB',
    'EURUSD'  : 'EURUSD=X',
    'USDJPY'  : 'JPY=X',
    'GOLD'    : 'GC=F',
    'OIL_WTI' : 'CL=F',
    'COPPER'  : 'HG=F',
    'CPO'     : '2445.KL',
    'VIX'     : '^VIX',
    'UST_2Y'  : 'SHY',      # FIXED: notebook pakai SHY, bukan ^IRX
    'UST_10Y' : '^TNX',
}

BEI_TOP100 = [
    'BBCA.JK','BBRI.JK','BMRI.JK','TLKM.JK','ASII.JK',
    'BBNI.JK','UNVR.JK','ICBP.JK','INDF.JK','KLBF.JK',
    'HMSP.JK','GGRM.JK','BRIS.JK','PTBA.JK','ADRO.JK',
    'ITMG.JK','HRUM.JK','CPIN.JK','JPFA.JK','INCO.JK',
    'ANTM.JK','TINS.JK','MDKA.JK','AMMN.JK','MEDC.JK',
    'PGAS.JK','JSMR.JK','WIKA.JK','PTPP.JK','SMGR.JK',
    'INTP.JK','BSDE.JK','CTRA.JK','PWON.JK','SMRA.JK',
    'EXCL.JK','ISAT.JK','BUKA.JK','GOTO.JK','EMTK.JK',
    'MNCN.JK','ACES.JK','MAPI.JK','LPPF.JK','AALI.JK',
    'LSIP.JK','SIMP.JK','BBTN.JK','BNGA.JK','BJBR.JK',
    'BRIS.JK','BTPS.JK','ARTO.JK','KAEF.JK','MIKA.JK',
    'HEAL.JK','TPIA.JK','BRPT.JK','AKRA.JK','INKP.JK',
    'TKIM.JK','AUTO.JK','UNTR.JK','SMSM.JK','MAIN.JK',
    'MYOR.JK','ULTJ.JK','ROTI.JK','SIDO.JK','DLTA.JK',
    'GGRM.JK','WIIM.JK','PNBN.JK','BDMN.JK','NISP.JK',
    'BJTM.JK','MAYA.JK','AMRT.JK','MIDI.JK','TBIG.JK',
    'TOWR.JK','RAJA.JK','ELSA.JK','PGEO.JK','INDY.JK',
    'DEWA.JK','MBMA.JK','NCKL.JK','CMRY.JK','MCAS.JK',
    'BFIN.JK','ADMF.JK','WSKT.JK','ADHI.JK','WTON.JK',
    'FASW.JK','INKP.JK','SRIL.JK','PBRX.JK','CARE.JK'
]

LAG_MAX       = 10
PVAL_CUTOFF   = 0.05
MI_PERCENTILE = 50
TOP_N         = 30
RANDOM_STATE  = 42
TRAIN_END     = '2024-12-31'
VAL_END       = '2025-03-31'
MODEL_PATH    = 'models/'


# ===============================================================================
# 0. MANUAL TECHNICAL INDICATORS (replacement utk pandas_ta - lihat docstring)
# ===============================================================================

def _rsi(close, length=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bbands(close, length=20, std=2):
    mid   = close.rolling(length).mean()
    sd    = close.rolling(length).std()
    upper = mid + std * sd
    lower = mid - std * sd
    return lower, mid, upper


def _atr(high, low, close, length=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _ema(close, length):
    return close.ewm(span=length, adjust=False).mean()


# ===============================================================================
# 1. DATA FETCHING
# ===============================================================================

def fetch_prices(start='2023-01-01', progress=False):
    """Download semua harga dari yfinance + imputasi (identik notebook)."""
    raw = {}
    failed = []
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=start, progress=progress, auto_adjust=True)
            if len(df) > 50:
                raw[name] = df['Close'].squeeze()
        except Exception:
            failed.append(name)

    prices = pd.DataFrame(raw)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Imputasi 3-step identik notebook: ffill(limit1) -> bfill(limit1) -> interpolate
    for col in prices.columns:
        if prices[col].isnull().any():
            s = prices[col]
            s = s.ffill(limit=1)
            s = s.bfill(limit=1)
            s = s.interpolate(method='linear', limit_direction='both')
            prices[col] = s

    return prices, failed


def compute_whi(price_df, ma_period=3, lookback=5, norm_factor=1000):
    """Hitung WHI harian dari saham BEI (Satrio 2025) - identik notebook."""
    p   = price_df.dropna(axis=1, thresh=int(len(price_df) * 0.7)).ffill().bfill()
    ma3 = p.rolling(ma_period).mean()
    mx5 = ma3.shift(1).rolling(lookback).max()
    mn5 = ma3.shift(1).rolling(lookback).min()
    up  = (ma3 > mx5).sum(axis=1)
    dn  = (ma3 < mn5).sum(axis=1)
    whi = (dn / (up + 1)) * norm_factor
    whi.name = 'WHI'
    return whi, up, dn


def fetch_bei_stocks(start='2023-01-01', progress=False):
    df = yf.download(BEI_TOP100, start=start, auto_adjust=True, progress=progress)['Close']
    df = df.ffill()
    return df


# ===============================================================================
# 2. FEATURE ENGINEERING (identik notebook: Layer 1-4)
# ===============================================================================

def build_features(prices, whi_series, ath_cnt, atl_cnt):
    all_dates = prices.index.union(whi_series.index)
    feat = pd.DataFrame(index=all_dates)

    # -- Layer 1: Macro Returns & Levels --
    RETURN_VARS = ['IHSG', 'NIKKEI', 'HSI', 'STI', 'KOSPI', 'SHANGHAI',
                   'SP500', 'DJI', 'NASDAQ', 'FTSE', 'DAX', 'MSCI_EM',
                   'USDIDR', 'DXY', 'EURUSD', 'USDJPY',
                   'GOLD', 'OIL_WTI', 'COPPER', 'CPO']
    LEVEL_VARS = ['VIX', 'UST_2Y', 'UST_10Y']

    for v in RETURN_VARS:
        if v in prices.columns:
            feat[f'{v}_ret'] = prices[v].pct_change() * 100
    for v in LEVEL_VARS:
        if v in prices.columns:
            feat[f'{v}_lvl'] = prices[v]

    feat['WHI_lvl']      = whi_series
    feat['WHI_extreme']  = (whi_series > 4000).astype(int)
    feat['WHI_momentum'] = whi_series - whi_series.shift(3)
    feat['ATH_count']    = ath_cnt
    feat['ATL_count']    = atl_cnt

    # Imputasi WHI-related (identik notebook cell 16)
    feat['WHI_lvl'] = feat['WHI_lvl'].interpolate(method='linear', limit_direction='both')
    feat['WHI_extreme']  = (feat['WHI_lvl'] > 4000).astype(int)
    feat['WHI_momentum'] = feat['WHI_lvl'] - feat['WHI_lvl'].shift(3)
    feat['ATH_count'] = feat['ATH_count'].interpolate(method='linear', limit_direction='both')
    feat['ATL_count'] = feat['ATL_count'].interpolate(method='linear', limit_direction='both')

    # -- Layer 2: Technical Indicators dari IHSG --
    if 'IHSG' in prices.columns:
        ihsg_close = prices['IHSG'].dropna()
        ihsg_high  = ihsg_close * 1.005
        ihsg_low   = ihsg_close * 0.995

        feat['RSI_14'] = _rsi(ihsg_close, length=14)
        feat['RSI_7']  = _rsi(ihsg_close, length=7)

        macd_line, macd_signal, macd_hist = _macd(ihsg_close, fast=12, slow=26, signal=9)
        feat['MACD_line']   = macd_line
        feat['MACD_signal'] = macd_signal
        feat['MACD_hist']   = macd_hist
        feat['MACD_cross']  = (macd_hist > 0).astype(int)

        bb_lower, bb_mid, bb_upper = _bbands(ihsg_close, length=20, std=2)
        feat['BB_pct_b']    = (ihsg_close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        feat['BB_width']    = (bb_upper - bb_lower) / (bb_mid + 1e-8) * 100
        feat['BB_oversold'] = (feat['BB_pct_b'] < 0.1).astype(int)

        atr = _atr(ihsg_high, ihsg_low, ihsg_close, length=14)
        feat['ATR_14']  = atr
        feat['ATR_pct'] = atr / ihsg_close * 100

        ema20 = _ema(ihsg_close, length=20)
        ema50 = _ema(ihsg_close, length=50)
        feat['IHSG_above_EMA20'] = (ihsg_close > ema20).astype(int)
        feat['EMA20_slope']      = ema20.pct_change(5) * 100
        feat['IHSG_above_EMA50'] = (ihsg_close > ema50).astype(int)
        feat['EMA20_above_EMA50'] = (ema20 > ema50).astype(int)

        feat['IHSG_mom_3d']  = ihsg_close.pct_change(3) * 100
        feat['IHSG_mom_5d']  = ihsg_close.pct_change(5) * 100
        feat['IHSG_mom_10d'] = ihsg_close.pct_change(10) * 100

    # -- Layer 3: Cross-Market Confluence (identik notebook, termasuk "bug" nama
    #    IHSG_NIKKEI_corr20 yang sebenarnya rolling 7 hari, sama dgn notebook asli) --
    if 'IHSG' in prices.columns and 'SP500' in prices.columns:
        ihsg_r     = prices['IHSG'].pct_change()
        sp500_r    = prices['SP500'].pct_change()
        hsi_r      = prices['HSI'].pct_change()
        sti_r      = prices['STI'].pct_change()
        kospi_r    = prices['KOSPI'].pct_change()
        shanghai_r = prices['SHANGHAI'].pct_change()
        nasdaq_r   = prices['NASDAQ'].pct_change()
        ftse_r     = prices['FTSE'].pct_change()
        nikkei_r   = prices['NIKKEI'].pct_change()

        feat['IHSG_SP500_corr7']    = ihsg_r.rolling(7).corr(sp500_r)
        feat['IHSG_HSI_corr7']      = ihsg_r.rolling(7).corr(hsi_r)
        feat['IHSG_STI_corr7']      = ihsg_r.rolling(7).corr(sti_r)
        feat['IHSG_KOSPI_corr7']    = ihsg_r.rolling(7).corr(kospi_r)
        feat['IHSG_SHANGHAI_corr7'] = ihsg_r.rolling(7).corr(shanghai_r)
        feat['IHSG_NASDAQ_corr7']   = ihsg_r.rolling(7).corr(nasdaq_r)
        feat['IHSG_FTSE_corr7']     = ihsg_r.rolling(7).corr(ftse_r)
        feat['IHSG_NIKKEI_corr20']  = ihsg_r.rolling(7).corr(nikkei_r)  # nama sesuai notebook

        def rolling_beta(y, x, window=60):
            cov = y.rolling(window).cov(x)
            var = x.rolling(window).var()
            return cov / (var + 1e-10)

        feat['IHSG_SP500_beta60']    = rolling_beta(ihsg_r, sp500_r, 60)
        feat['IHSG_HSI_beta60']      = rolling_beta(ihsg_r, hsi_r, 60)
        feat['IHSG_STI_beta60']      = rolling_beta(ihsg_r, sti_r, 60)
        feat['IHSG_KOSPI_beta60']    = rolling_beta(ihsg_r, kospi_r, 60)
        feat['IHSG_SHANGHAI_beta60'] = rolling_beta(ihsg_r, shanghai_r, 60)
        feat['IHSG_NASDAQ_beta60']   = rolling_beta(ihsg_r, nasdaq_r, 60)
        feat['IHSG_FTSE_beta60']     = rolling_beta(ihsg_r, ftse_r, 60)

    # -- Layer 4: Macro Regime Indicators --
    if 'UST_2Y' in prices.columns and 'UST_10Y' in prices.columns:
        feat['yield_spread']   = prices['UST_10Y'] - prices['UST_2Y']
        feat['yield_inverted'] = (feat['yield_spread'] < 0).astype(int)

    if 'DXY' in prices.columns:
        dxy_sma50 = prices['DXY'].rolling(50).mean()
        feat['DXY_above_SMA50'] = (prices['DXY'] > dxy_sma50).astype(int)
        feat['DXY_5d_mom']      = prices['DXY'].pct_change(5) * 100

    if 'VIX' in prices.columns:
        vix_sma20 = prices['VIX'].rolling(20).mean()
        vix_std20 = prices['VIX'].rolling(20).std()
        feat['VIX_spike']   = (prices['VIX'] > vix_sma20 + vix_std20).astype(int)
        feat['VIX_change']  = prices['VIX'].diff()
        feat['VIX_above25'] = (prices['VIX'] > 25).astype(int)

    # EM Stress Score: VIX z + DXY z + yield_spread z (negated), rolling 60 hari
    # FIXED: sebelumnya window 252 dan tanpa komponen yield_spread -> beda dgn notebook
    stress_components = []
    if 'VIX_lvl' in feat.columns:
        vix_z = (feat['VIX_lvl'] - feat['VIX_lvl'].rolling(60).mean()) / \
                (feat['VIX_lvl'].rolling(60).std() + 1e-8)
        stress_components.append(vix_z)
    if 'DXY_5d_mom' in feat.columns:
        dxy_z = (feat['DXY_5d_mom'] - feat['DXY_5d_mom'].rolling(60).mean()) / \
                (feat['DXY_5d_mom'].rolling(60).std() + 1e-8)
        stress_components.append(dxy_z)
    if 'yield_spread' in feat.columns:
        yld_z = -(feat['yield_spread'] - feat['yield_spread'].rolling(60).mean()) / \
                 (feat['yield_spread'].rolling(60).std() + 1e-8)
        stress_components.append(yld_z)
    if stress_components:
        feat['EM_stress_score'] = pd.concat(stress_components, axis=1).mean(axis=1)

    # Commodity basket RI: CPO + OIL_WTI (FIXED nama & komponen, sebelumnya
    # 'commodity_basket' pakai GOLD+OIL+COPPER -> tidak match dgn fitur model)
    commodity_rets = []
    for comm in ['CPO', 'OIL_WTI']:
        col = f'{comm}_ret'
        if col in feat.columns:
            commodity_rets.append(feat[col])
    if commodity_rets:
        feat['RI_commodity_basket'] = pd.concat(commodity_rets, axis=1).mean(axis=1)

    return feat


# ===============================================================================
# 3. LAG + FEATURE SELECTION
# ===============================================================================

def build_lag_features(feat, lag_max=LAG_MAX):
    feat = feat.copy()
    feat['Y'] = (feat['IHSG_ret'].shift(-1) > 0).astype(int)
    base_features = [c for c in feat.columns if c not in ('Y', 'IHSG_ret')]

    lag_dict = {}
    for f in base_features:
        for lag in range(1, lag_max + 1):
            lag_dict[f'{f}_lag{lag}'] = feat[f].shift(lag)

    lag_df = pd.DataFrame(lag_dict)
    analysis_df = pd.concat([lag_df, feat['Y']], axis=1).dropna(subset=['Y'])
    analysis_df = analysis_df.dropna(thresh=int(len(lag_dict) * 0.5))
    return analysis_df, base_features


def select_features(analysis_df, pval_cutoff=PVAL_CUTOFF,
                     mi_percentile=MI_PERCENTILE, top_n=TOP_N):
    lag_cols = [c for c in analysis_df.columns if c != 'Y']
    Y_vec = analysis_df['Y']

    spearman_res = []
    for col in lag_cols:
        x = analysis_df[col]
        valid = x.notna() & Y_vec.notna()
        if valid.sum() < 50:
            continue
        corr, pval = spearmanr(x[valid], Y_vec[valid])
        spearman_res.append({
            'feature': col, 'spearman_r': corr,
            'abs_r': abs(corr), 'pval': pval,
            'sig_spearman': pval < pval_cutoff
        })
    spearman_df = pd.DataFrame(spearman_res)

    X_mi = analysis_df[lag_cols].fillna(analysis_df[lag_cols].median())
    mi_scores = mutual_info_classif(X_mi, Y_vec, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({'feature': lag_cols, 'mi_score': mi_scores})
    mi_thresh = np.percentile(mi_scores, mi_percentile)
    mi_df['sig_mi'] = mi_df['mi_score'] > mi_thresh

    combined = spearman_df.merge(mi_df, on='feature')
    combined['both_sig'] = combined['sig_spearman'] & combined['sig_mi']

    top_both = combined[combined['both_sig']].sort_values('abs_r', ascending=False).head(top_n)
    if len(top_both) == 0:
        top_both = combined.sort_values('abs_r', ascending=False).head(top_n)

    selected = top_both['feature'].tolist()

    model_df = analysis_df[selected + ['Y']].dropna()
    X_vif = model_df[selected]
    final_features = _iterative_vif(X_vif)

    return final_features, model_df, combined


def _iterative_vif(X, threshold=10.0):
    Xc = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    while True:
        vif = pd.DataFrame({
            'feature': Xc.columns,
            'VIF': [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
        }).sort_values('VIF', ascending=False)
        if vif['VIF'].max() <= threshold or len(Xc.columns) <= 3:
            break
        Xc = Xc.drop(columns=[vif.iloc[0]['feature']])
    return Xc.columns.tolist()


# ===============================================================================
# 4. TRAINING (Random Forest only)
# ===============================================================================

def temporal_split(model_df, final_features, train_end=TRAIN_END, val_end=VAL_END):
    X_df = model_df[final_features].fillna(model_df[final_features].median())
    y_s = model_df['Y']

    tm = X_df.index <= train_end
    vm = (X_df.index > train_end) & (X_df.index <= val_end)
    te = X_df.index > val_end

    X_tr, y_tr = X_df[tm].values, y_s[tm].values
    X_va, y_va = X_df[vm].values, y_s[vm].values
    X_te, y_te = X_df[te].values, y_s[te].values

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_te)

    dates_val = X_df.index[vm]
    dates_test = X_df.index[te]

    return (X_tr_sc, y_tr, X_va_sc, y_va, X_te_sc, y_te,
            scaler, dates_val, dates_test)


def tune_and_train(X_tr, y_tr, X_va, y_va):
    """GridSearchCV Random Forest pada train, threshold sweep pada val."""
    inner_cv = TimeSeriesSplit(n_splits=3, gap=3)
    THRESHOLD_GRID = np.arange(0.30, 0.71, 0.01)

    estimator = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    param_grid = {'max_depth': [3, 5, 7], 'min_samples_leaf': [5, 10, 20]}

    gs = GridSearchCV(estimator, param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)

    best_params = {'Random Forest': gs.best_params_}
    tuned_models = {'Random Forest': gs.best_estimator_}

    prob = gs.best_estimator_.predict_proba(X_va)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        p = (prob >= t).astype(int)
        f1 = f1_score(y_va, p, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_thresh = {'Random Forest': round(float(best_t), 2)}

    return tuned_models, best_params, best_thresh


def retrain_on_trainval(tuned_models, best_params, X_tr, y_tr, X_va, y_va):
    X_tv = np.vstack([X_tr, X_va])
    y_tv = np.concatenate([y_tr, y_va])

    p = best_params['Random Forest']
    m = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, **p)
    m.fit(X_tv, y_tv)
    return {'Random Forest': m}


def save_models(final_models, scaler, final_features, best_thresh, path=MODEL_PATH):
    os.makedirs(path, exist_ok=True)
    joblib.dump(final_models, f'{path}models.pkl')
    joblib.dump(scaler, f'{path}scaler.pkl')
    joblib.dump(final_features, f'{path}features.pkl')
    joblib.dump(best_thresh, f'{path}thresholds.pkl')


def load_models(path=MODEL_PATH):
    """Load model. Kompatibel dgn models.pkl lama yang masih berisi 3 model
    (LR/RF/ANN) -> otomatis ambil 'Random Forest' saja."""
    models = joblib.load(f'{path}models.pkl')
    scaler = joblib.load(f'{path}scaler.pkl')
    features = joblib.load(f'{path}features.pkl')
    thresh = joblib.load(f'{path}thresholds.pkl')

    if 'Random Forest' in models:
        models = {'Random Forest': models['Random Forest']}
    if isinstance(thresh, dict) and 'Random Forest' in thresh:
        thresh = {'Random Forest': thresh['Random Forest']}

    return models, scaler, features, thresh


# ===============================================================================
# 5. DAILY PREDICTION
# ===============================================================================

def predict_tomorrow(models, scaler, final_features, best_thresh, feat_df):
    """Ambil baris terakhir dari feat_df, predict probabilitas IHSG besok hijau."""
    results = {}
    lag_df_last = {}

    for f in final_features:
        parts = f.rsplit('_lag', 1)
        base = parts[0]
        lag = int(parts[1])
        if base in feat_df.columns:
            series = feat_df[base].shift(lag)
            lag_df_last[f] = series.iloc[-1] if len(series) > 0 else np.nan
        else:
            lag_df_last[f] = np.nan

    x_row = pd.DataFrame([lag_df_last])[final_features]
    x_row = x_row.fillna(x_row.median())
    x_row = x_row.fillna(0)
    x_scaled = scaler.transform(x_row.values)

    for name, model in models.items():
        prob = model.predict_proba(x_scaled)[0, 1]
        thresh = best_thresh.get(name, 0.5)
        pred = int(prob >= thresh)
        results[name] = {
            'probability': round(float(prob), 4),
            'prediction': pred,
            'label': '\U0001F7E2 HIJAU' if pred == 1 else '\U0001F534 MERAH',
            'threshold': thresh,
            'confidence': abs(prob - 0.5) * 2
        }

    return results, x_row
