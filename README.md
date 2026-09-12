 # ☕ Coffee Bean Classification System (SVM + Genetic Algorithm)

Aplikasi berbasis web untuk analisis dan klasifikasi mutu serta jenis biji kopi menggunakan metode **Support Vector Machine (SVM)** yang dioptimasi dengan **Algoritma Genetika (GA)**. Proyek ini disusun dengan mengimplementasikan standar metodologi **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*).

🌐 **Demo Aplikasi Live:** [https://klasifikasibijikopi.streamlit.app](https://klasifikasibijikopi.streamlit.app)

## 📌 Fitur Utama

- 🔍 **Ekstraksi Fitur Citra:** Menggabungkan ekstraksi fitur warna (HSV & Histogram RGB) serta fitur tekstur *grayscale*.
- 🧬 **Optimasi Hyperparameter GA:** Menggunakan framework `DEAP` untuk mencari kombinasi nilai $C$ dan $\gamma$ (gamma) optimal pada kernel RBF SVM.
- ⚖️ **Penanganan Imbalanced Data:** Memanfaatkan pengujian berbasis *F1-Score Macro* dan *class weighting balanced*.
- 🖼️ **Visualisasi Tepi (Canny Edges):** Menampilkan pemrosesan deteksi kontur citra biji kopi secara *real-time*.
- 📊 **Laporan Analisis & Rekomendasi:** Memberikan gambaran kelas, tingkat mutu (grade), deskripsi fisik, serta saran penanganan pascapanen/roasting.

## 🏗️ Metodologi (CRISP-DM)

1. **Business Understanding:** Mengidentifikasi kebutuhan pemisahan mutu biji kopi secara otomatis dan akurat.
2. **Data Understanding:** Analisis dan verifikasi struktur dataset citra biji kopi.
3. **Data Preparation:** Pembentukan fitur numerik dari citra (Warna HSV, Histogram RGB, Tekstur) dan *standard scaling*.
4. **Modeling:** Pengujian model SVM RBF yang dioptimasi menggunakan Algoritma Genetika.
5. **Evaluation:** Evaluasi performa model menggunakan *Accuracy*, *F1-Score Macro*, dan *Confusion Matrix*.
6. **Deployment:** Pembangunan antarmuka interaktif menggunakan Streamlit dan penyebarannya via Streamlit Community Cloud.

## 📁 Struktur Repositori

```text
├── save_models/
│   ├── svm_ga_model.pkl      # Model SVM hasil pelatihan
│   ├── scaler.pkl            # Standard Scaler artefak
│   └── label_encoder.pkl     # Label Encoder artefak
├── app.py                    # Script antarmuka aplikasi Streamlit
├── train.py                  # Script pelatihan model & optimasi GA
├── requirements.txt          # Daftar dependensi library Python
└── README.md                 # Dokumentasi proyek
