import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import time
from pathlib import Path


st.set_page_config(
    page_title="SafeFlow AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .alert-box {
        background-color: #ff4444;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #000000;
    }
    /* Sekme Tasarımı */
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .stTabs [data-baseweb="tab"] { height: 50px; padding: 0 2rem; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="main-header">
        <h1>🔍 SafeFlow AI</h1>
        <p>Endüstriyel Sızıntı & Çatlak Tespit Sistemi</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">Powered by YOLOv8 </p>
    </div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"❌ Model yüklenemedi: {e}")
        st.info("💡 'best.pt' dosyasının script ile aynı dizinde olduğundan emin olun.")
        st.stop()

model = load_model()


def process_image(image, conf_threshold=0.15):
    start_time = time.time()
    img_array = np.array(image)
    
    results = model.predict(source=img_array, conf=conf_threshold, iou=0.45, verbose=False)
    
    annotated_img = results[0].plot(line_width=4, font_size=1.5)
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    boxes = results[0].boxes
    detections = []
    for box in boxes:
        detections.append({
            'class': results[0].names[int(box.cls[0])],
            'confidence': float(box.conf[0])
        })
    
    processing_time = time.time() - start_time
    return annotated_img_rgb, detections, processing_time


def process_video(video_path, conf_threshold=0.15, progress_bar=None, st_frame=None):
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
   
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
      
        results = model.predict(source=frame, conf=conf_threshold, iou=0.45, verbose=False)
        
       
        annotated_frame = results[0].plot(line_width=4, font_size=1.5)
        
      
        total_detections += len(results[0].boxes)
        
        
        out.write(annotated_frame)
        
    
        if st_frame:
            
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, caption=f"Analiz Ediliyor: Kare {frame_count}/{total_frames}", use_container_width=True)
        
        frame_count += 1
        
        
        if progress_bar:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
    
    cap.release()
    out.release()
    
    return output_path, total_detections, frame_count


tab1, tab2 = st.tabs(["📷 Fotoğraf Analizi", "🎥 Video Analizi"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Görsel Yükle")
        uploaded_file = st.file_uploader("Bir fotoğraf seçin...", type=['jpg', 'jpeg', 'png'], key="image_uploader")
        
        conf_threshold = st.slider("Hassasiyet Ayarı", 0.05, 0.50, 0.15, 0.05, help="Düşük değerler en ufak sızıntıları bile yakalar.")
    
    with col2:
        st.subheader("Analiz Sonuçları")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        with st.spinner('🔍 Görüntü taranıyor...'):
            annotated_img, detections, proc_time = process_image(image, conf_threshold)
        
        
        if detections:
            st.markdown(f"""<div class="alert-box">🚨 {len(detections)} SIZINTI TESPİT EDİLDİ!</div>""", unsafe_allow_html=True)
        else:
            st.success("✅ Sistem Temiz: Sızıntı bulunamadı.")
        
        col_res1, col_res2 = st.columns([2, 1])
        with col_res1:
            st.image(annotated_img, caption='İşlenmiş Görüntü', use_container_width=True)
        with col_res2:
            st.markdown("### 📊 Metrikler")
            st.markdown(f"""<div class="metric-card"><h4>Tespit Sayısı</h4><h2 style="color: #000000;">{len(detections)}</h2></div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="metric-card" style="margin-top:1rem;"><h4>İşlem Süresi</h4><h2 style="color: #000000;">{proc_time:.2f}sn</h2></div>""", unsafe_allow_html=True)


with tab2:
    st.subheader("Video Yükle ve Canlı İzle")
    
    uploaded_video = st.file_uploader("Video dosyasını seçin...", type=['mp4', 'avi', 'mov'], key="video_uploader")
    video_conf = st.slider("Video Hassasiyeti", 0.05, 0.50, 0.15, 0.05, key="video_conf")
    
    if uploaded_video:
       
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            video_path = tmp_file.name
        
        if st.button("🚀 Canlı Analizi Başlat", type="primary", use_container_width=True):
            
            
            vid_col1, vid_col2 = st.columns([2, 1])
            
            with vid_col2:
                st.markdown("### 📊 Durum Paneli")
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.info("Video motoru hazırlanıyor...")
            
            with vid_col1:
               
                st_frame_placeholder = st.empty()
            
           
            output_path, total_det, total_frames = process_video(
                video_path,
                video_conf,
                progress_bar,
                st_frame_placeholder 
            )
            
            status_text.success("✅ Analiz tamamlandı!")
            
            
            with vid_col2:
                st.markdown("---")
                if total_det > 0:
                     st.markdown(f"""<div class="alert-box" style="font-size:1rem;">🚨 TOPLAM {total_det} KAREDE SIZINTI!</div>""", unsafe_allow_html=True)
                else:
                    st.success("✅ Video temiz.")
                
                
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Analiz Videosunu İndir",
                        data=f,
                        file_name="safeflow_analiz.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )


st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>SafeFlow AI © 2025 | Endüstriyel Güvenlik ve İzleme Sistemi</p>
    </div>
""", unsafe_allow_html=True)

# python -m streamlit run app.py 