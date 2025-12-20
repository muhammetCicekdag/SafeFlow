import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
from datetime import datetime
import pandas as pd
import time

# --- 1. PROFESYONEL SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Safe-Flow AI | Kontrol Merkezi",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. KURUMSAL ARAYÜZ TASARIMI (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #05070a; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #0b111b; border-right: 1px solid #1f2937; }
    
    /* KPI Kart Tasarımları */
    .kpi-card { 
        background-color: #0f172a; padding: 20px; border-radius: 15px; 
        border: 1px solid #1e293b; text-align: center;
        transition: transform 0.3s;
    }
    .kpi-card:hover { transform: translateY(-5px); border-color: #38bdf8; }
    .kpi-value { font-size: 32px; font-weight: bold; color: #38bdf8; }
    .kpi-label { font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

    /* Dinamik Alarm Paneli */
    .status-panel {
        padding: 30px; border-radius: 15px; text-align: center; font-weight: 800;
        font-size: 24px; letter-spacing: 2px; margin-bottom: 25px; border: 3px solid transparent;
    }
    .status-safe { background: rgba(16, 185, 129, 0.15); color: #10b981; border-color: #10b981; }
    .status-danger { 
        background: rgba(239, 68, 68, 0.25); color: #ef4444; border-color: #ef4444;
        animation: alert-pulse 1s infinite;
    }
    @keyframes alert-pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. YAN PANEL: SİSTEM KONTROLLERİ ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🛡️ SAFE-FLOW</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>Endüstriyel Sızıntı Takip Sistemi</p>", unsafe_allow_html=True)
    st.divider()
    
    st.subheader("🛠️ Hassasiyet Protokolleri")
    # Recall oranını desteklemek için AI hassasiyeti ayarı
    ai_conf = st.slider("AI Doğrulama Hassasiyeti", 0.01, 1.0, 0.20)
    # Piksel bazlı hareket hassasiyeti
    motion_sens = st.slider("Damla Algılama Eşiği", 5, 200, 45)
    
    st.divider()
    yuklenen_video = st.file_uploader("📂 Gözlem Videosu Yükle", type=['mp4', 'avi', 'mov'])
    
    st.divider()
    st.markdown("**Proje Yürütücüsü:** Muhammet ÇİÇEKDAĞ")
    st.caption("MAKÜ - Yönetim Bilişim Sistemleri")

# --- 4. ANA DASHBOARD ---
st.markdown("<h1 style='margin-bottom: 0;'>🚀 Gerçek Zamanlı Analiz Ekranı</h1>", unsafe_allow_html=True)

if yuklenen_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(yuklenen_video.read())
    
    # YOLO Model Yükleme
    try:
        model = YOLO("best.pt")
    except:
        st.error("Kritik Hata: 'best.pt' dosyası bulunamadı. Lütfen klasörü kontrol edin.")
        st.stop()

    # Arka Plan Çıkarıcı (MOG2) - Her damlayı yakalamak için
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=motion_sens, detectShadows=False)
    cap = cv2.VideoCapture(tfile.name)
    
    # Görsel Bileşenler
    status_placeholder = st.empty()
    col1, col2, col3, col4 = st.columns(4)
    with col1: k_ai = st.empty()
    with col2: k_mot = st.empty()
    with col3: k_loc = st.empty()
    with col4: k_tim = st.empty()

    v_col, l_col = st.columns([2, 1])
    with v_col: video_screen = st.empty()
    with l_col:
        st.subheader("📋 Olay Kayıtları")
        event_log = st.empty()
        st.subheader("📊 Analiz Grafiği")
        chart_placeholder = st.empty()

    olaylar = []
    grafik_verisi = []

    if st.button("SİSTEM ANALİZİNİ BAŞLAT"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # ADIM 1: AI (YOLO) TESPİTİ
            results = model(frame, conf=ai_conf, verbose=False)
            ai_kutulari = []
            max_conf = 0
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    max_conf = max(max_conf, conf)
                    ai_kutulari.append((x1, y1, x2, y2))

            # ADIM 2: HAREKET ANALİZİ (Her damlayı yakalar)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            mask = fgbg.apply(blur)
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            sızıntı_durumu = False
            lokasyon = "---"

            # ADIM 3: ÇİFT DOĞRULAMA VE İŞARETLEME
            for cnt in contours:
                if cv2.contourArea(cnt) > 25: # Küçük gürültüleri ele
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    
                    # KOŞUL: Ya AI onaylayacak ya da hareket çok bariz olacak (Recall desteği)
                    #
                    ai_onay = any(ax1 < center[0] < ax2 and ay1 < center[1] < ay2 for (ax1, ay1, ax2, ay2) in ai_kutulari)
                    
                    if ai_onay or cv2.contourArea(cnt) > 300: # Bariz sızıntı ise AI görmese de çiz
                        sızıntı_durumu = True
                        lokasyon = f"{center[0]},{center[1]}"
                        # İSTEDİĞİN ÖZELLİK: Sızıntıyı Kırmızı Daire içine al
                        cv2.circle(frame, center, int(radius) + 10, (0, 0, 255), 3)
                        cv2.putText(frame, "TEHLIKE: SIZINTI", (center[0]-50, center[1]-int(radius)-20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ADIM 4: DASHBOARD GÜNCELLEME
            if sızıntı_durumu:
                status_placeholder.markdown('<div class="status-panel status-danger">⚠️ KRİTİK ALARM: SIZINTI TESPİT EDİLDİ</div>', unsafe_allow_html=True)
                if len(olaylar) == 0 or (datetime.now() - olaylar[-1]["_t"]).seconds > 2:
                    olaylar.append({"Zaman": datetime.now().strftime("%H:%M:%S"), "Tür": "DOĞRULANMIŞ", "Konum": lokasyon, "_t": datetime.now()})
            else:
                status_placeholder.markdown('<div class="status-panel status-safe">✅ SİSTEM GÜVENLİ: AKIŞ NORMAL</div>', unsafe_allow_html=True)

            # KPI Kartları
            k_ai.markdown(f'<div class="kpi-card"><div class="kpi-label">AI GÜVEN</div><div class="kpi-value">%{int(max_conf*100)}</div></div>', unsafe_allow_html=True)
            k_mot.markdown(f'<div class="kpi-card"><div class="kpi-label">HAREKET</div><div class="kpi-value">{len(contours)}</div></div>', unsafe_allow_html=True)
            k_loc.markdown(f'<div class="kpi-card"><div class="kpi-label">LOKASYON</div><div class="kpi-value">{lokasyon}</div></div>', unsafe_allow_html=True)
            k_tim.markdown(f'<div class="kpi-card"><div class="kpi-label">SÜRE</div><div class="kpi-value">{datetime.now().strftime("%H:%M:%S")}</div></div>', unsafe_allow_html=True)

            # Video ve Analitikler
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_screen.image(frame_rgb, use_container_width=True)
            
            if len(olaylar) > 0:
                event_log.dataframe(pd.DataFrame(olaylar)[["Zaman", "Tür", "Konum"]].tail(8), use_container_width=True)
            
            grafik_verisi.append(max_conf)
            if len(grafik_verisi) > 50: grafik_verisi.pop(0)
            chart_placeholder.line_chart(grafik_verisi)

        cap.release()
else:
    st.info("💡 Başlamak için lütfen sol panelden bir analiz videosu yükleyin.")
    st.image("https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&q=80&w=1200")