# Human Pose Analysis

OpenPose, MediaPipe ve MMPose kullanılarak 2D/3D insan poz tahmini ve görüntü/video tabanlı hareket analizi çalışması.

## Amaç

İnsan vücudundaki eklem noktalarının görüntü ve videolardan çıkarılması, 2D ve 3D poz tahminlerinin karşılaştırılması ve bu verilerden hareket örüntülerinin belirlenmesi amaçlanmaktadır.

## Kullanılan Yöntemler

### OpenPose
- BODY_25
- 25 keypoint
- 2D poz tahmini

### MediaPipe
- Pose Landmarker
- 33 landmark
- 2D poz tahmini
- 3D world landmark tahmini

### MMPose
- RTMDet ile insan tespiti
- RTMPose-M ile 2D poz tahmini
- SimpleBaseline3D ile 2D → 3D pose lifting
- Human3.6M formatında 17 keypoint

## Durum

| Yöntem | 2D | 3D |
|---|---|---|
| OpenPose | ✅ | Çoklu kamera gerektirir |
| MediaPipe | ✅ | ✅ |
| MMPose | ✅ | ✅ |

## Çıktılar

### OpenPose 2D
![OpenPose 2D](outputs/openpose_2d.png)

### MediaPipe 2D
![MediaPipe 2D](outputs/mediapipe_2d.jpg)

### MediaPipe 3D
![MediaPipe 3D](outputs/mediapipe_3d.png)

### MMPose 2D
![MMPose 2D](outputs/mmpose_2d.jpg)

### MMPose 3D
![MMPose 3D](outputs/mmpose_3d.jpg)

## Sonraki Aşamalar

- Video üzerinde 2D ve 3D poz tahmini
- Landmarkların zaman içindeki değişiminin takibi
- Hareket özelliklerinin çıkarılması
- Hareket örüntülerinin sınıflandırılması

Model dosyaları, sanal ortamlar ve harici kütüphane kaynakları repoya dahil edilmemektedir.
