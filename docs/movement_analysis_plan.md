# Hareket Analizi Planı

Poz tahmini modellerinden elde edilen eklem noktaları kullanılarak insan hareketlerinin zaman içerisindeki değişimi analiz edilecektir.

## Ortak Eklem Noktaları

- nose
- left_shoulder
- right_shoulder
- left_elbow
- right_elbow
- left_wrist
- right_wrist
- left_hip
- right_hip
- left_knee
- right_knee
- left_ankle
- right_ankle

Bu ortak yapı MediaPipe, OpenPose ve MMPose çıktılarının aynı hareket analiz sisteminde kullanılmasını sağlayacaktır.

## İlk Hareketler

### Kol Yukarı Kalktı
Bileğin omuz seviyesinin üzerine çıkması ve bu durumun belirli bir süre devam etmesi kontrol edilecektir.

### Ayak / Bacak Kaldırıldı
Ayak bileğinin normal duruş seviyesine göre yukarı hareketi ve diz açısındaki değişim incelenecektir.

### El Ağza Yaklaştı
Bilek ile burun/ağız bölgesi arasındaki normalize edilmiş mesafe takip edilecektir.

### Sağa / Sola Dönme
Omuz ve kalça noktalarının görüntüdeki konumu ve 3D derinlik değişimleri kullanılarak gövdenin dönüş yönü incelenecektir.

## Temel Akış

Pose Estimation
-> Ortak Eklem Formatı
-> Kareler Arası Takip
-> Açı / Mesafe / Yön Özellikleri
-> Hareket Kararı

## Önemli Not

Tek bir görüntü poz bilgisini verir. Hareket tespiti için aynı landmarkların ardışık video kareleri boyunca izlenmesi gerekir.
