# IHSG Next-Day Direction Predictor (Random Forest)
## Streamlit Cloud Deployment Guide

### Struktur File
```
ihsg_dashboard/
├── app.py              ← Streamlit app utama
├── pipeline.py         ← ML pipeline (fetch → features → train → predict)
├── requirements.txt    ← Python dependencies
├── models/              ← models.pkl, scaler.pkl, features.pkl, thresholds.pkl
└── README.md
```

### Changelog (fix terbaru)
- **Model disederhanakan jadi Random Forest only** (model terbaik). LR & ANN dihapus dari tampilan; `load_models()` tetap kompatibel kalau `models.pkl` lama masih berisi 3 model — otomatis ambil RF saja.
- **`pandas-ta` dihapus total** dari dependency — package ini unmaintained dan gagal install di Streamlit Cloud (bug `numpy.NaN` di numpy versi baru). Semua indikator teknikal (RSI, MACD, Bollinger Bands, ATR, EMA) sekarang di-reimplement manual di `pipeline.py`, hasilnya identik (Wilder smoothing utk RSI/ATR, sama seperti default pandas_ta).
- **`scikit-learn` dipin ke `1.6.1`** — match versi yang dipakai saat training di notebook, menghindari warning/risk inconsistency dari versi yang lebih baru.
- **Bug feature-engineering dibenarkan** agar identik dengan notebook (sebelumnya 7 dari 22 fitur final model salah dihitung → prediksi bisa salah):
  - `EM_stress_score`: window rolling dibenarkan 60 hari (bukan 252) + ditambah komponen `yield_spread`
  - `RI_commodity_basket`: nama kolom & komponen dibenarkan (CPO+OIL_WTI, bukan GOLD+OIL+COPPER dengan nama `commodity_basket`)
  - `IHSG_NIKKEI_corr20`: nama kolom dibenarkan agar match fitur model
  - Ticker `UST_2Y`: dibenarkan ke `SHY` (sebelumnya salah pakai `^IRX`)

---

### Deploy ke Streamlit Cloud (Gratis)

**Step 1 — Push ke GitHub**
```bash
# Buat repo baru di GitHub (misal: ihsg-predictor)
git init
git add .
git commit -m "initial: IHSG direction predictor"
git remote add origin https://github.com/USERNAME/ihsg-predictor.git
git push -u origin main
```

**Step 2 — Deploy di Streamlit Cloud**
1. Buka https://share.streamlit.io
2. Login dengan GitHub
3. Klik **New app**
4. Pilih repo: `ihsg-predictor`
5. Branch: `main`
6. Main file: `app.py`
7. Klik **Deploy!**

Selesai. App live dalam ~3 menit.

---

### Cara Penggunaan App

1. Buka app → market snapshot langsung tampil (auto-refresh 15 menit)
2. Klik **🚀 Jalankan Pipeline** di sidebar
3. Pipeline akan:
   - Download data terbaru dari Yahoo Finance
   - Hitung WHI dari 100 saham BEI
   - Build & select features
   - Load model tersimpan ATAU retrain (pilih di sidebar)
   - Predict arah IHSG besok
4. Hasil tampil: prediksi Random Forest + gauge + charts

### Mode Model
- **Gunakan model tersimpan** → cepat (~30 detik), pakai model dari run sebelumnya
- **Retrain ulang** → lambat (~3-5 menit, RF only), train ulang dari scratch dengan data terbaru

### Auto-refresh Harian
Streamlit Cloud tidak punya native scheduler.
Untuk daily auto-prediction, gunakan salah satu:

**Opsi A — GitHub Actions (recommended, gratis):**
```yaml
# .github/workflows/daily_predict.yml
name: Daily IHSG Prediction
on:
  schedule:
    - cron: '30 1 * * 1-5'  # 08:30 WIB setiap hari kerja
  workflow_dispatch:
jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python -c "
          from pipeline import *
          prices, _ = fetch_prices()
          bei = fetch_bei_stocks()
          whi, ath, atl = compute_whi(bei)
          feat = build_features(prices, whi, ath, atl)
          analysis_df, _ = build_lag_features(feat)
          final_features, model_df, _ = select_features(analysis_df)
          X_tr,y_tr,X_va,y_va,X_te,y_te,scaler,_,_ = temporal_split(model_df, final_features)
          tm,bp,bt = tune_and_train(X_tr,y_tr,X_va,y_va)
          fm = retrain_on_trainval(tm,bp,X_tr,y_tr,X_va,y_va)
          save_models(fm,scaler,final_features,bt)
          res,_ = predict_tomorrow(fm,scaler,final_features,bt,feat)
          print(res)
          "
      - uses: actions/upload-artifact@v3
        with:
          name: models
          path: models/
```

**Opsi B — Buka app tiap pagi:**
Cukup klik Jalankan Pipeline setiap pagi sebelum BEI buka (sebelum 09:00 WIB).
App akan fetch data H-1 dan predict hari ini.

---

### Catatan Penting
- Data VIX, S&P500, FTSE, dll tutup lebih malam dari BEI → prediksi H+1 valid
- IHSG buka 09:00 WIB → jalankan pipeline sebelum jam 9 pagi
- Model perlu retrain berkala (tiap 3 bulan disarankan) agar tidak concept drift
