import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import time
import os

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="EcoScan AI - Deteksi Sampah",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #00CC66;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
    }
    .stButton>button:hover {
        background-color: #00994d;
        border: none;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2e7d32;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (ROBUST VERSION)
# ==========================================
def build_model(neuron_layer_1, neuron_layer_2):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(224, 224, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(neuron_layer_1, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(neuron_layer_2, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    return model

@st.cache_resource
def load_learner():
    model_path = 'model_klasifikasi_sampah.keras' # Pastikan nama file benar
    
    if not os.path.exists(model_path):
        return None

    try:
        model = build_model(256, 128)
        model.load_weights(model_path)
        return model
    except:
        try:
            model = build_model(64, 32)
            model.load_weights(model_path)
            return model
        except:
            return None

model = load_learner()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
class_names = ['Cardboard 📦', 'Glass 🥃', 'Metal ⚙️', 'Paper 📄', 'Plastic 🥤', 'Trash 🗑️']
clean_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image).astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def get_recycling_info(label):
    info = {
        'cardboard': ("Kardus/Karton", "Lipat hingga pipih agar hemat tempat. Pastikan kering dan tidak kena minyak.", "✅ Bisa Daur Ulang"),
        'glass': ("Kaca/Beling", "Cuci bersih botol kaca. Lepaskan tutup jika terbuat dari logam/plastik.", "✅ Bisa Daur Ulang (Selamanya)"),
        'metal': ("Logam/Kaleng", "Bersihkan sisa makanan/minuman. Kaleng aluminium sangat bernilai tinggi.", "✅ Bisa Daur Ulang"),
        'paper': ("Kertas", "Pisahkan dari kertas kotor (tisu bekas/kertas minyak). Kertas koran & HVS sangat baik.", "✅ Bisa Daur Ulang"),
        'plastic': ("Plastik", "Cek kode daur ulang di bawah botol. Remukkan botol untuk menghemat ruang tong sampah.", "⚠️ Cek Kode Plastik"),
        'trash': ("Residu/Lainnya", "Sampah ini sulit didaur ulang atau terkontaminasi. Buang di tong sampah residu.", "⛔ Buang ke TPA")
    }
    return info.get(label, ("Unknown", "Tidak ada data", "N/A"))

# ==========================================
# 4. SIDEBAR (PROFIL AUTHOR)
# ==========================================
with st.sidebar:
    st.title("EcoScan AI")
    st.markdown("---")
    
    # --- FOTO PROFIL (Ganti dengan file fotomu sendiri) ---
    # Jika belum ada foto, pakai link avatar dummy ini dulu
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
    
    # --- DATA DIRI ---
    st.header("Alif Addarisalam Wibowo")
    st.caption("Mahasiswa Teknik Informatika")
    st.caption("Universitas Muhammadiyah Riau")
    
    st.markdown("---")
    
    # --- NAVIGASI UPLOAD ---
    st.write("**Mulai Analisis:**")
    uploaded_file = st.file_uploader("Upload Gambar Sampah", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    
    # --- KONTAK / SOSMED ---
    st.write("📫 **Hubungi Saya:**")
    
    # Gunakan Columns agar tombolnya sejajar
    col_sosmed1, col_sosmed2 = st.columns(2)
    
    with col_sosmed1:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/alif-addarisalam-ba61632a7/", use_container_width=True)
    with col_sosmed2:
        st.link_button("GitHub", "https://github.com/Aldwib", use_container_width=True)
        
    st.link_button(
        "📧 Email Saya", 
        "mailto:alifaddarisalam@student.umri.ac.id?subject=Diskusi%20Proyek%20EcoScan", 
        use_container_width=True
    )

# ==========================================
# MAIN LAYOUT DENGAN TABS
# ==========================================
tab_aplikasi, tab_porto = st.tabs(["🚀 Aplikasi Deteksi", "👨‍💻 Tentang Pembuat"])

# --- TAB 1: APLIKASI UTAMA ---
with tab_aplikasi:
    if uploaded_file is None:
        # Tampilan Awal (Welcome Screen)
        st.header("👋 Selamat Datang di EcoScan AI")
        st.markdown("""
        Sistem cerdas untuk membantu Anda memilah sampah dengan benar.
        
        **Cara Penggunaan:**
        1. Buka Sidebar di sebelah kiri 👈
        2. Unggah foto sampah (Botol, Kertas, Kaleng, dll)
        3. AI akan menganalisis dan memberikan saran daur ulang
        """)
        
        # Tampilkan galeri contoh (biar cantik)
        c1, c2, c3 = st.columns(3)
        with c1: st.image("contoh_sampahPlastik.jpg", caption="Sampah Plastik", use_container_width=True)
        with c2: st.image("contoh_sampahKardus.jpg", caption="Sampah Kardus", use_container_width=True)
        with c3: st.image("contoh_sampahLogam.jpg", caption="Sampah Logam", use_container_width=True)

    else:
        # Jika file diupload
        if model is None:
            st.error("❌ Model gagal dimuat. Cek file .keras/.h5 Anda.")
        else:
            # Layout 2 Kolom: Gambar vs Hasil
            col_img, col_result = st.columns([1, 1.5])
            
            image = Image.open(uploaded_file).convert('RGB')
            
            with col_img:
                st.image(image, caption='Foto Sampah', use_container_width=True, channels="RGB")
                
            with col_result:
                st.subheader("🔍 Hasil Analisis")
                
                # Loading effect biar keren
                with st.spinner('Sedang memindai tekstur objek...'):
                    time.sleep(1) # Efek delay buatan (opsional)
                    processed = preprocess_image(image)
                    prediction = model.predict(processed)
                    
                    # Ambil data
                    idx = np.argmax(prediction)
                    label_clean = clean_names[idx]
                    label_display = class_names[idx]
                    confidence = prediction[0][idx] * 100
                    
                    # Data Info
                    indo_name, tips, status = get_recycling_info(label_clean)
                
                # Tampilkan Hasil Utama dalam Kartu
                st.success(f"**Terdeteksi:** {label_display}")
                
                # Metrik Keyakinan
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Jenis", indo_name)
                with metric_col2:
                    st.metric("Akurasi AI", f"{confidence:.1f}%")
                
                # Progress Bar Keyakinan
                st.write("Tingkat Keyakinan:")
                st.progress(int(confidence))
                
                # Pesan Status Daur Ulang
                if "Bisa" in status:
                    st.success(f"💡 {tips}")
                else:
                    st.warning(f"⚠️ {tips}")

            st.markdown("---")
            
            # Bagian Bawah: Grafik Statistik
            st.subheader("📊 Statistik Probabilitas")
            
            # Buat DataFrame untuk Grafik
            chart_data = pd.DataFrame({
                'Jenis Sampah': clean_names,
                'Probabilitas': prediction[0]
            })
            
            # Tampilkan Bar Chart
            st.bar_chart(chart_data.set_index('Jenis Sampah'), color="#00CC66")

            # Efek Balon jika akurasi tinggi
            if confidence > 85:
                st.balloons()

# --- TAB 2: PORTOFOLIO KAMU ---
with tab_porto:
    st.header("Tentang Pengembang")
    
    col_foto, col_bio = st.columns([1, 3])
    
    with col_foto:
        # Ganti dengan fotomu misal: "alif.jpg"
        st.image("foto_profil.jpeg", caption="Alif Addarisalam Wibowo", use_container_width=True)
        
    with col_bio:
        st.markdown("""
        ### Alif Addarisalam Wibowo
        **Mahasiswa Teknik Informatika - Universitas Muhammadiyah Riau**
        
        Halo! Saya adalah pengembang software yang berfokus pada *Artificial Intelligence* dan *Computer Vision*. 
        Proyek ini dikembangkan sebagai bagian dari penelitian Tugas Akhir untuk memecahkan masalah pemilahan sampah menggunakan Deep Learning.
        
        **Keahlian:**
        - 🐍 Python (TensorFlow, Keras, Streamlit)
        - 🌐 Web Development
        - 🤖 Machine Learning Algorithm
        """)
        
        st.markdown("---")
        st.subheader("Jejak Digital")
        st.markdown("""
        - 💼 [LinkedIn](https://www.linkedin.com/in/alif-addarisalam-ba61632a7/)
        - 🐙 [GitHub](https://github.com/Aldwib)
        - 📸 [Instagram](https://www.instagram.com/alif_addarisalam.w/)
        """)


