import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Deteksi Jenis & Kualitas Biji Kopi",
    page_icon="☕",
    layout="wide"
)

# Load Model, Scaler, dan Label Encoder dari Folder save_models
@st.cache_resource
def load_artifacts():
    model = joblib.load("./save_models/svm_ga_model.pkl")
    scaler = joblib.load("./save_models/scaler.pkl")
    label_encoder = joblib.load("./save_models/label_encoder.pkl")
    return model, scaler, label_encoder

try:
    model, scaler, le = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False

# Ekstraksi Fitur Gambar Tunggal
def extract_features_single(img_array, target_size=(128, 128)):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, target_size)

    # Fitur Warna HSV
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_mean, h_std = cv2.meanStdDev(hsv[:, :, 0])
    s_mean, s_std = cv2.meanStdDev(hsv[:, :, 1])
    v_mean, v_std = cv2.meanStdDev(hsv[:, :, 2])

    # Fitur Histogram RGB
    hist_b = cv2.calcHist([img_resized], [0], None, [8], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_resized], [1], None, [8], [0, 256]).flatten()
    hist_r = cv2.calcHist([img_resized], [2], None, [8], [0, 256]).flatten()

    # Fitur Tekstur Grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    g_mean, g_std = cv2.meanStdDev(gray)

    features = np.hstack([
        [h_mean[0][0], h_std[0][0], s_mean[0][0], s_std[0][0], v_mean[0][0], v_std[0][0]],
        hist_b, hist_g, hist_r,
        [g_mean[0][0], g_std[0][0]]
    ])
    
    edges = cv2.Canny(gray, 100, 200)
    return features, edges, v_mean[0][0], s_mean[0][0], g_std[0][0]

# Pemetaan Keterangan Kualitas & Deskripsi Berdasarkan Label
def get_quality_info(label_name):
    label_lower = label_name.lower()
    
    # 1. Kategori Cacat / Defect
    if "defect" in label_lower or "cacat" in label_lower or "busuk" in label_lower:
        return {
            "status": "Cacat / Defektif (Non-Standard)",
            "grade": "Mutu Rendah / Off-Grade",
            "deskripsi": "Biji kopi memiliki cacat fisik (pecah, berlubang karena hama, atau mengalami pembusukan/jamur). Hal ini dapat merusak cita rasa utama (off-flavor) saat diseduh.",
            "rekomendasi": "Dipisahkan dari kelompok biji utama. Tidak disarankan untuk *specialty coffee*, dapat dialokasikan untuk produk olahan turunan komersial non-premium.",
            "color": "error"
        }
    # 2. Kategori Arabica
    elif "arabica" in label_lower:
        return {
            "status": "Biji Kopi Arabika (Normal)",
            "grade": "Grade 1 / Specialty Potential",
            "deskripsi": "Biji kopi dalam kondisi fisik utuh dengan bentuk agak lonjong dan alur tengah yang khas. Memiliki karakteristik keasaman (*acidity*) yang kompleks dan aroma manis-floral.",
            "rekomendasi": "Sangat cocok untuk *medium roast* atau *light roast* untuk mempertahankan profil rasa uniknya (specialty coffee).",
            "color": "success"
        }
    # 3. Kategori Robusta
    elif "robusta" in label_lower:
        return {
            "status": "Biji Kopi Robusta (Normal)",
            "grade": "Grade 1 Commercial",
            "deskripsi": "Biji kopi dalam kondisi utuh dengan bentuk cenderung bulat bundar. Memiliki kadar kafein tinggi, bodi tebal (*bold body*), dan rasa cenderung pahit-cokelat (*nutty/chocolaty*).",
            "rekomendasi": "Sangat baik untuk profil *medium-to-dark roast*, ideal digunakan sebagai bahan baku espresso blend atau kopi susu kekinian.",
            "color": "success"
        }
    # 4. Kategori Mutu Baik / Premium Umum
    elif "good" in label_lower or "premium" in label_lower or "bagus" in label_lower:
        return {
            "status": "Biji Kopi Utuh & Sehat",
            "grade": "Mutu Tinggi (Grade A)",
            "deskripsi": "Biji kopi menunjukkan warna dan tekstur permukaan yang seragam tanpa ada tanda-tanda kerusakan fisik atau serangan hama.",
            "rekomendasi": "Siap untuk tahap penyangraian (*roasting*) sesuai target profil rasa yang diinginkan.",
            "color": "success"
        }
    # 5. Kategori Default / Umum
    else:
        return {
            "status": f"Kategori Kopi {label_name.upper()}",
            "grade": "Mutu Teridentifikasi",
            "deskripsi": f"Biji kopi berhasil teridentifikasi dalam kelompok kelas `{label_name}` berdasarkan fitur warna dan tekstur permukaannya.",
            "rekomendasi": "Lakukan verifikasi sortir lanjutan sesuai standar operasional penanganan pascapanen.",
            "color": "info"
        }

# UI Header
st.title("☕ Sistem Analisis & Deteksi Kualitas Biji Kopi")
st.markdown("Implementasi Metodologi CRISP-DM: Optimasi Parameter Support Vector Machine (SVM) Menggunakan **Algoritma Genetika (GA)**.")
st.divider()

if not artifacts_loaded:
    st.error("⚠️ File model tidak ditemukan di folder `save_models/`! Pastikan file `.pkl` sudah ter-upload.")
else:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📷 Unggah Sampel Biji Kopi")
        uploaded_file = st.file_uploader("Upload gambar biji kopi (Format: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Input", use_container_width=True)

    with col_right:
        st.subheader("📊 Hasil Prediksi & Analisis Detail")
        if uploaded_file is not None:
            img_np = np.array(image.convert("RGB"))

            with st.spinner("Mengekstraksi fitur dan melakukan klasifikasi..."):
                features, edges, v_val, s_val, g_std_val = extract_features_single(img_np)
                features_scaled = scaler.transform([features])

                pred_idx = model.predict(features_scaled)[0]
                pred_label = le.inverse_transform([pred_idx])[0]

                # Ambil keterangan detail mutu
                info_kualitas = get_quality_info(pred_label)

            # 1. Output Kategori Utama
            if info_kualitas["color"] == "error":
                st.error(f"### **Hasil Klasifikasi:** `{pred_label.upper()}`")
            else:
                st.success(f"### **Hasil Klasifikasi:** `{pred_label.upper()}`")

            # 2. Keterangan Kualitas Lengkap (Markdown Card)
            st.markdown(f"""
            #### 📝 **Keterangan Hasil Analisis:**
            * **Status Kelayakan:** `{info_kualitas['status']}`
            * **Tingkat Kualitas (Grade):** **{info_kualitas['grade']}**
            * **Deskripsi Karakteristik:** {info_kualitas['deskripsi']}
            * **Rekomendasi Penanganan:** {info_kualitas['rekomendasi']}
            """)

            st.divider()

            # 3. Detail Parameter Ekstraksi Fitur
            st.markdown("#### **Detail Parameter Ekstraksi Fitur:**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Kecerahan (HSV-V)", f"{v_val:.2f}")
            m2.metric("Saturasi Warna (HSV-S)", f"{s_val:.2f}")
            m3.metric("Tekstur (Std Dev)", f"{g_std_val:.2f}")

            # 4. Visualisasi Edges
            st.markdown("#### **Visualisasi Tepi (Canny Edge Detection):**")
            st.image(edges, caption="Hasil Deteksi Kontur & Tepi", use_container_width=True)
        else:
            st.info("Silakan unggah gambar pada panel di sebelah kiri untuk menampilkan analisis lengkap.")
