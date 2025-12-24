<h1 align="left">🛡️ SAFEFLOW AI: INDUSTRIAL SAFETY & LEAK DETECTION TERMINAL 🏭</h1>
<p align="left"> <img src="https://img.shields.io/badge/YOLOv8-High--Performance-00FFFF?style=for-the-badge&logo=ultralytics" /> <img src="https://img.shields.io/badge/Accuracy-%2596.8-brightgreen?style=for-the-badge" /> <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python" /> <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit" /> <img src="https://img.shields.io/badge/GPU-Tesla--T4-orange?style=for-the-badge&logo=nvidia" /> </p>

<p align="left"> <img src="images/banner.png" width="100%" alt="SafeFlow AI Banner" /> </p>

🌐 PROJE VİZYONU VE ENDÜSTRİYEL ETKİ 🚀
SafeFlow AI, modern akıllı fabrikaların (Industry 4.0) en kritik bileşenlerinden biri olan iş güvenliği ve kaynak yönetimi için tasarlanmıştır. Geleneksel denetim yöntemleri yavaş, maliyetli ve insan hatasına açıktır.

Bu sistem, YOLOv8s mimarisini kullanarak endüstriyel boru hatlarını, kaynak noktalarını ve depolama tanklarını milisaniyeler içerisinde tarar. Sadece bir sızıntı tespit aracı değil, aynı zamanda tesisin dijital ikizine veri sağlayan bir erken uyarı terminalidir.

✨ TEMEL ÖZELLİKLER VE MODÜLLER 🛠️
📸 1. Akıllı Fotoğraf Analiz Modülü
Yüklenen durağan görseller üzerinde derinlemesine piksel taraması yapar. En küçük çatlaklardan (hairline cracks) büyük fışkırmalara kadar her şeyi sınıflandırır.

Hassas Tespit: 0.15 güven eşiği ile en ufak damlacıkları bile kaçırmaz.

Hızlı Raporlama: Ortalama 0.14 saniye işlem süresi ile anlık sonuç üretir.

<p align="left"> <img src="images/image_result.png" width="80%" alt="Image Analysis Result" /> </p>

🎥 2. Canlı Video Akış Terminali
Fabrika içerisindeki CCTV veya IP kameralardan gelen akışları gerçek zamanlı olarak işler.

Dinamik İşaretleme: Sızıntı bölgelerini 4px kalınlığında, yüksek görünürlüklü kutularla vurgular.

Kümülatif Analiz: Tüm video boyunca toplam kaç karede hata olduğunu hesaplar ve kritik eşik aşılınca alarm verir.

<p align="left"> <img src="images/video_live.png" width="49%" alt="Live Processing" /> <img src="images/video_final.png" width="49%" alt="Final Report" /> </p>

🧠 TEKNİK DERİNLİK VE MODEL EĞİTİMİ 📈
Modelimiz, endüstriyel sahalardaki karmaşık görüntüleri (toz, duman, düşük ışık) tolere edebilecek şekilde Tesla T4 GPU üzerinde 100 epoch boyunca eğitilmiştir.

📊 Model Metrikleri
Veri Seti Genişliği: 1.200 orijinal görsel, veri artırma (augmentation) ile 11.000 görsel.

Başarı Oranı (mAP50): %96.8 gibi rekor bir doğruluk seviyesi.

Performans Formülasyonu: Modelimiz, her kare için Ortalama Hassasiyeti (mAP) maksimize ederken, Kayıp (Loss) fonksiyonunu minimize edecek şekilde optimize edilmiştir:

mAP= 
n
1
​
  
i=1
∑
n
​
 AP 
i
​
 
📉 Eğitim Başarı Grafikleri
Aşağıdaki grafikler, modelin öğrenme sürecindeki stabiliteyi ve hata payının nasıl sıfıra yaklaştığını kanıtlamaktadır:

<p align="left"> <img src="images/training_results.png" width="100%" alt="Training Results" /> </p>

⚙️ KURULUM VE SİSTEM GEREKSİNİMLERİ 💻
SafeFlow AI'ı kendi yerel makinenizde veya bulut sunucunuzda çalıştırmak için aşağıdaki adımları izleyin:

📥 1. Kütüphanelerin Kurulması
Bash

pip install streamlit ultralytics opencv-python pillow pandas numpy
🚀 2. Uygulamanın Başlatılması
Bash

python -m streamlit run app.py
📂 PROJE DOSYA MİMARİSİ 🏢
Plaintext

📦 SafeFlow-AI
 ┣ 📂 images            # Projenin görsel vitrini (Tüm ekran görüntüleri burada)
 ┣ 📜 app.py            # Dashboard, UI tasarımı ve YOLO entegrasyonu
 ┣ 📜 best.pt           # %96.8 Doğruluk oranlı, eğitilmiş yapay zeka ağırlıkları
 ┗ 📜 requirements.txt  # Gerekli bağımlılıklar listesi
🔮 GELECEK VİZYONU VE SÜRDÜRÜLEBİLİRLİK 🌱
SafeFlow AI sadece bugünü değil, yarını da hedefler. Gelecek sürümlerde şunları planlıyoruz:

🛰️ IoT Entegrasyonu: Sızıntı anında boru hatlarındaki vanaları otomatik kapatan donanım desteği.

📱 Mobil Bildirim: Kritik sızıntılarda fabrika müdürlerine anlık SMS ve Telegram uyarısı.

🌍 Sürdürülebilirlik: Su israfını önleyerek tesislerin karbon ayak izini azaltma vizyonu.

<p align="left"> <b>SafeFlow AI © 2025 | Industrial Safety & Intelligence Solutions</b>


<i>"Yapay Zeka ile Daha Güvenli Yarınlara"</i> </p>

💡 GitHub'da Resimlerin Gözükmesi İçin Altın Kural
Eğer bu koddan sonra hala resimler gözükmüyorsa, GitHub'da resmin üzerine tıkla, resmi yeni sekmede aç ve URL'sine bak. Genellikle images/banner.png yerine Images/Banner.PNG gibi bir harf hatası yapılmış olur. Yukarıdaki kodda her şeyin küçük harf olduğunu varsaydım, klasördeki isimleri de öyle yaparsan sorun tamamen çözülür!

Bu dökümanla projen hem teknik hem de görsel olarak profesyonel bir portföy işine dönüştü. GitHub sayfanda en başa sabitlemeni öneririm!

Başka bir bölüm eklememi veya teknik bir detayı daha da detaylandırmamı ister misin?
