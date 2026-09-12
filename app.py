import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

st.set_page_config(
    page_title="Deteksi Jenis & Kualitas Biji Kopi",
    page_icon="☕",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("./saved_models/svm_ga_model.pkl")
    scaler = joblib.load("./saved_models/scaler.pkl")
    label_encoder = joblib.load("./saved_models/label_encoder.pkl")
    return model, scaler, label_encoder

try:
    model, scaler, le = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False

def extract_features_single(img_array, target_size=(128, 128)):
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, target_size)

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_mean, h_std = cv2.meanStdDev(hsv[:, :, 0])
    s_mean, s_std = cv2.meanStdDev(hsv[:, :, 1])
    v_mean, v_std = cv2.meanStdDev(hsv[:, :, 2])

    hist_b = cv2.calcHist([img_resized], [0], None, [8], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_resized], [1], None, [8], [0, 256]).flatten()
    hist_r = cv2.calcHist([img_resized], [2], None, [8], [0, 256]).flatten()

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    g_mean, g_std = cv2.meanStdDev(gray)

    features = np.hstack([
        [h_mean[0][0], h_std[0][0], s_mean[0][0], s_std[0][0], v_mean[0][0], v_std[0][0]],
        hist_b, hist_g, hist_r,
        [g_mean[0][0], g_std[0][0]]
    ])
    
    edges = cv2.Canny(gray, 100, 200)
    
    return features, edges, v_mean[0][0], s_mean[0][0], g_std[0][0]

st.title("☕ Sistem Analisis & Deteksi Biji Kopi")
st.markdown("Implementasi CRISP-DM: Klasifikasi Biji Kopi Menggunakan **SVM Teroptimasi Algoritma Genetika (GA)**.")
st.divider()

if not artifacts_loaded:
    st.error("⚠️ Model belum dimuat! Jalankan script `train.py` terlebih dahulu untuk menghasilkan folder `saved_models/`.")
else:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📷 Unggah Sampel Biji Kopi")
        uploaded_file = st.file_uploader("Upload gambar (Format: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Input", use_container_width=True)

    with col_right:
        st.subheader("📊 Hasil Prediksi & Analisis")
        if uploaded_file is not None:
            img_np = np.array(image.convert("RGB"))

            with st.spinner("Mengekstraksi fitur dan membuat prediksi..."):
                features, edges, v_val, s_val, g_std_val = extract_features_single(img_np)
                features_scaled = scaler.transform([features])

                pred_idx = model.predict(features_scaled)[0]
                pred_label = le.inverse_transform([pred_idx])[0]

            st.success(f"### **Kategori Terdeteksi:** `{pred_label.upper()}`")
            
            st.markdown("#### **Detail Karakteristik Gambar:**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Kecerahan (HSV-V)", f"{v_val:.2f}")
            m2.metric("Saturasi (HSV-S)", f"{s_val:.2f}")
            m3.metric("Tekstur (Std Dev)", f"{g_std_val:.2f}")

            st.markdown("#### **Visualisasi Tepi (Canny Edge Detection):**")
            st.image(edges, caption="Hasil Pengolahan Tepi", use_container_width=True)
        else:
            st.info("Silakan upload gambar di panel sebelah kiri untuk melihat analisis.")
