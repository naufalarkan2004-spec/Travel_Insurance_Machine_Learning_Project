# Travel Insurance Claim Prediction

Pembuatan model yang memprediksi apakah pemegang polis asuransi perjalanan akan mengajukan klaim (`Claim` = Yes/No), lalu mensimulasikan dampak finansialnya dengan *cost-benefit simulation*.

> Seluruh analisis ada di satu notebook: [`notebook/TravelInsuranceClaim_Capstone3_AlphaTeam_Final.ipynb`](notebook/TravelInsuranceClaim_Capstone3_AlphaTeam_Final.ipynb) (ditulis dalam Bahasa Indonesia).

## Ringkasan

| | |
|---|---|
| **Masalah** | Klasifikasi biner dengan kelas sangat timpang (klaim hanya ±1,5% dari polis) |
| **Data** | 44.328 polis, 11 kolom (setelah cleaning: 43.639 polis, 7 fitur) |
| **Model akhir** | Logistic Regression (`class_weight='balanced'`) + `SelectFpr`, di-tuning dengan GridSearchCV |
| **Metrik utama** | PR-AUC (step-wise), Recall, F2, dan penghematan finansial (cost-benefit) |
| **Hasil (test set, threshold 0,5)** | Recall ≈ 0,78, precision ≈ 0,06, PR-AUC ≈ 0,078 (±5× baseline acak 0,015) |

Model ini lebih cocok dibaca sebagai **alat prioritisasi/penyaringan polis**, bukan pengambil keputusan otomatis. Lihat bagian [Keterbatasan](#keterbatasan).

## Dataset

File: `raw/data_travel_insurance.csv` (data mentah), `data/interim/data_clean.csv` (setelah cleaning)

| Kolom | Keterangan |
|---|---|
| `Agency`, `Agency Type` | Agen penjual polis dan tipenya (Airlines / Travel Agency) |
| `Distribution Channel` | Online / Offline |
| `Product Name` | Nama produk asuransi (26 produk) |
| `Gender` | Jenis kelamin (71,4% kosong) |
| `Duration` | Lama perjalanan (hari) |
| `Destination` | Negara tujuan (138 kategori) |
| `Net Sales` | Nilai penjualan polis |
| `Commission (in value)` | Komisi agen |
| `Age` | Usia pemegang polis |
| `Claim` | **Target**: Yes / No |

Sumber dan lisensi data:https://www.kaggle.com/datasets/mhdzahier/travel-insurance/code?datasetId=111060&sortBy=voteCount

## Alur Analisis

1. **Data understanding.** Target hanya 1,53% positif (677 dari 44.328), sehingga accuracy tidak informatif.
2. **Data cleaning**
   - `Gender` di-drop: 71,4% kosong dan polanya struktural per agensi (EPX, SSI, CWT hampir 100% kosong), bukan acak.
   - Duplikat (4.667 baris, 10,5%) **hanya diidentifikasi, tidak dihapus**. Dataset tidak punya ID polis, sehingga duplikat sah tidak bisa dibedakan dari duplikat error.
   - Baris tidak masuk akal dibuang: `Duration` < 0 atau > 761 hari, dan `Age` > 100 (689 baris, 7 di antaranya klaim).
3. **EDA dan seleksi fitur** (hanya memakai data train, untuk mencegah *data leakage*)
   - Uji asosiasi fitur-target: Mann-Whitney U dan Spearman (numerik); chi-square dengan p-value permutasi dan Cramér's V (kategorikal).
   - `Distribution Channel` di-drop (tidak signifikan, p = 0,73).
   - `Agency Type` di-drop (redundan dengan `Agency`, Cramér's V = 1,0).
   - Fitur final (7): `Agency`, `Product Name`, `Destination`, `Duration`, `Net Sales`, `Commission`, `Age`.
4. **Feature engineering** (`ColumnTransformer`)
   - `Agency`, `Product Name`: One-Hot Encoding.
   - `Destination` (138 kategori): `TargetEncoder`.
   - Numerik: median imputer, `Winsorizer` (IQR, fold = 3), `StandardScaler`.
5. **Modeling**
   - Split 80/20 stratified (train 34.911, test 8.728).
   - Benchmark 14 model dengan 5-fold stratified CV, dengan dan tanpa `SelectFpr`: Dummy, Logistic Regression, Decision Tree, KNN, Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM (beberapa dengan class weight).
   - Model kompleks (Random Forest, XGBoost, LightGBM) overfit berat, sedangkan model linear stabil dengan PR-AUC setara. Dipilih **Logistic Regression + class weight + SelectFpr**.
   - Tuning: `C = 0.01`, `SelectFpr alpha = 0.05`. Peningkatan PR-AUC CV dari tuning lebih kecil daripada std CV, jadi tidak bermakna secara statistik.
   - Resampling (Random Over/Under Sampler, SMOTE, NearMiss) tidak memberi nilai tambah dibanding class weight. NearMiss memperburuk hasil.
6. **Interpretasi.** Koefisien Logistic Regression dan SHAP: `Agency_C2B`, `Agency_LWC`, dan `Net Sales` menaikkan risiko klaim; `Agency_EPX` dan produk *Cancellation Plan* menurunkannya. Ini menjelaskan *model*, bukan sebab-akibat.
7. **Cost-benefit simulation.** Membandingkan kerugian tanpa model vs dengan model pada test set.

## Hasil

Evaluasi pada test set (8.728 polis, 134 klaim), threshold 0,5:

| Metrik | Nilai |
|---|---|
| Recall | ≈ 0,78 (104 klaim terdeteksi, 30 terlewat) |
| Precision (kelas klaim) | ≈ 0,06 |
| F2 | ≈ 0,22 |
| PR-AUC | ≈ 0,078 (baseline acak = prevalensi 0,015) |
| False positive | 1.758 polis |

Simulasi cost-benefit (asumsi di bawah):

| Skenario | Kerugian |
|---|---|
| Tanpa model | USD 349.606 |
| Dengan model | USD 132.517 |
| **Penghematan** | **USD 217.089 (62,1%)** |

**Asumsi biaya**
- `COST_FN` = USD 2.609 per klaim yang lolos, diambil dari rata-rata nilai klaim asuransi perjalanan menurut sumber eksternal (Squaremouth, data pasar AS). Nilai klaim **tidak ada di dataset**, jadi angka ini perlu diverifikasi dan sangat menentukan hasil.
- `COST_FP` = rata-rata premi dikurangi rata-rata komisi (USD 30,86 per polis), yaitu keuntungan yang hilang jika polis non-klaim ditolak/ditahan.
- `Net Sales` dan `Commission` diperlakukan sebagai USD, padahal dataset tidak menyebut mata uangnya.

## Keterbatasan

- **Sinyal prediktif lemah.** PR-AUC ≈ 0,078 dan precision ≈ 6%: sekitar 94% polis yang ditandai adalah false positive. Hasil finansial positif karena `COST_FN` diasumsikan jauh lebih besar dari `COST_FP`; jika asumsi itu meleset, kesimpulannya bisa berbalik.
- **Test set kecil.** Hanya 134 klaim, sehingga estimasi metrik punya varians besar.
- **Agensi JWT hilang dari data.** Seluruh 522 baris JWT punya `Age` = 118 (tampak seperti placeholder) dan ikut terbuang oleh filter `Age` > 100. Model tidak pernah melihat JWT.
- **Polis tahunan berdurasi panjang** (`Duration` > 761 hari) dikeluarkan, sehingga hasil tidak mewakili segmen tersebut.
- **Duplikat dipertahankan**, sehingga baris identik bisa jatuh di train dan test dan skor test sedikit lebih optimistis (besarnya perlu verifikasi).
- **Threshold tetap 0,5.** Threshold berbasis biaya belum dieksplorasi dan bisa mengubah hasil cost-benefit.
- Asosiasi `Agency` dan `Product Name` tinggi (Cramér's V = 0,83), sehingga koefisien keduanya tidak boleh dibaca sebagai efek independen.

## Struktur Repository

```
.
├── assets/                       # logo
├── data/interim/data_clean.csv   # fitur final + target (hasil cleaning)
├── model/model_logreg_v1.sav     # model pickle (dilatih pada seluruh data bersih)
├── notebook/
│   └── TravelInsuranceClaim_Capstone3_AlphaTeam_Final.ipynb   # notebook utama
├── raw/data_travel_insurance.csv # data mentah
├── reports/slide/Business Presentation.pdf
├── requirements.txt
└── README.md
```

Notebook menulis `data_clean.csv` dan `model_logreg_v1.sav` ke folder tempat notebook dijalankan. Salinan hasilnya sudah disimpan di `data/interim/` dan `model/`.

## Cara Menjalankan

```bash
git clone https://github.com/naufalarkan2004-spec/Travel_Insurance_Machine_Learning_Project.git
cd Travel_Insurance_Machine_Learning_Project

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# notebook membaca data dari folder kerja, jadi salin dulu data mentahnya
cp raw/data_travel_insurance.csv notebook/
cd notebook
jupyter notebook TravelInsuranceClaim_Capstone3_AlphaTeam_Final.ipynb
```

Jalankan semua sel secara berurutan dari atas. Notebook memakai API scikit-learn yang relatif baru (mis. `TargetEncoder`); pada saat penulisan dijalankan dengan scikit-learn 1.9.1 dan Python 3.14. Jika terjadi error versi, samakan versi paket dengan itu.

## Memakai Model yang Sudah Disimpan

```python
import pickle
import pandas as pd

with open("model/model_logreg_v1.sav", "rb") as f:
    model = pickle.load(f)

# kolom yang dibutuhkan: Agency, Product Name, Duration, Destination,
#                        Net Sales, Commission, Age
X_new = pd.read_csv("data/interim/data_clean.csv").drop(columns="target")
proba = model.predict_proba(X_new)[:, 1]
pred = model.predict(X_new)
```

Catatan: `model_logreg_v1.sav` dilatih ulang pada **seluruh** data bersih (bukan hanya train), sehingga memprediksi `data_clean.csv` dengannya adalah evaluasi in-sample dan tidak boleh dipakai untuk mengukur performa. Berkas pickle hanya boleh dimuat dari sumber tepercaya dan dengan versi scikit-learn yang sama.

## Tech Stack

Python, pandas, NumPy, SciPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, feature-engine, SHAP, Matplotlib, Seaborn.

## Penulis

Capstone 3, Purwadhika Data Science & Machine Learning (AlphaTeam).

_Tambahkan nama anggota tim dan tautan LinkedIn/GitHub di sini._
