# Human Pose Analysis

Bu projede farklı human pose estimation yöntemlerini aynı bebek videosu üzerinde çalıştırıp sonuçlarını karşılaştırıyorum.

Çalışmanın temel amacı videodan insan vücudundaki eklem noktalarını çıkarmak ve daha sonraki aşamada bu noktaların zaman içerisindeki değişimlerinden hareketleri analiz etmek.

Genel çalışma yapısı:

```text
Video
  ↓
Pose Estimation
  ↓
Keypoint / Landmark
  ↓
Zamansal Değişim
  ↓
Hareket Analizi
```

Testlerde bebek videosu kullandım. İlerleyen aşamada sağa dönme, sola dönme, elini ağzına götürme veya ayağını kaldırma gibi hareketleri keypoint verilerinden çıkarmayı hedefliyorum.

## Kullandığım Yöntemler

### MediaPipe

MediaPipe Pose ile video üzerindeki vücut noktalarını çıkardım.

- 33 landmark kullanıyor
- Her frame için landmark bilgileri alınıyor
- Sonuçlar video üzerine çiziliyor
- Keypoint bilgileri CSV olarak kaydediliyor

Kaynak:  
https://github.com/google-ai-edge/mediapipe

### OpenPose

OpenPose tarafında BODY_25 modelini kullandım.

- 25 body keypoint
- Apple Silicon üzerinde CPU_ONLY olarak derlendi
- Her frame için keypoint sonuçları JSON olarak kaydediliyor
- Sonuç video üzerine skeleton olarak çiziliyor

Kaynak:  
https://github.com/CMU-Perceptual-Computing-Lab/openpose

### MMPose

MMPose tarafında 2D pose estimation kullandım.

Kullandığım yapı:

```text
RTMDet
  ↓
Person Detection
  ↓
RTMPose-M
  ↓
17 COCO Keypoint
```

Bu çalışmada MMPose için sadece 2D sonuç kullandım.

Kaynak:  
https://github.com/open-mmlab/mmpose

### DeeperCut

DeeperCut için yazarların yayınladığı `pose-tensorflow` reposunu kullandım.

Hazır COCO multiperson modeli ile frame bazlı pose estimation yaptım.

Genel pipeline:

```text
CNN
  ↓
Joint Detection
  ↓
Pairwise Relations
  ↓
Graph / Multicut
  ↓
Person Pose
```

Diğer yöntemlere göre daha yavaş çalıştığı için bebek videosunun ilk 10 saniyelik kısmında test ettim.

Kaynak:  
https://github.com/eldar/pose-tensorflow

## Bebek Videosu Testi

Ana test videosu:

```text
videos/input/baby_test.mp4
```

Video yaklaşık 182 saniye ve 30 FPS.

MediaPipe, OpenPose ve MMPose videonun tamamında çalıştırıldı.

DeeperCut ise ilk 10 saniye üzerinde çalıştırıldı.

## Karşılaştırma Videosu

Dört yöntemin aynı 10 saniyelik bölüm üzerindeki sonuçlarını tek videoda birleştirdim.

Yerleşim:

```text
MediaPipe | OpenPose
----------|----------
MMPose    | DeeperCut
```

Karşılaştırma videosu:

```text
outputs/baby_test_10s_comparison.mp4
```

Videoları birleştirmek için kullandığım script:

```text
scripts/combine_pose_comparison.py
```

## Güncel Durum

| Yöntem | Durum | Test |
|---|---|---|
| MediaPipe | Tamamlandı | Tam video |
| OpenPose | Tamamlandı | Tam video |
| MMPose 2D | Tamamlandı | Tam video |
| DeeperCut | Tamamlandı | İlk 10 saniye |

## Proje Yapısı

```text
Human_Pose_Analysis/
│
├── videos/
│   └── input/
│
├── models/
│
├── outputs/
│   ├── keypoints/
│   ├── mediapipe_video/
│   ├── openpose_video/
│   ├── mmpose_video/
│   └── deepercut_multicut/
│
├── scripts/
│   ├── mediapipe_video.py
│   ├── openpose_video.sh
│   ├── mmpose_video_2d.py
│   └── combine_pose_comparison.py
│
├── external/
│
└── README.md
```

## Sonraki Aşama

Şu an farklı pose estimation yöntemlerinin aynı video üzerindeki sonuçlarını elde etmiş durumdayım.

Sonraki aşamada çıkarılan keypointlerin zaman içerisindeki değişimlerini kullanarak hareket analizi üzerine çalışacağım.
