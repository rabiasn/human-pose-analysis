# Görsel Üzerinde İnsan Poz Tahmini Karşılaştırması

Aynı test görüntüsü üzerinde OpenPose, MediaPipe ve MMPose ile 2D ve 3D insan poz tahmini incelenmiştir.

## Sonuç Özeti

| Yöntem | 2D | 3D | Nokta Sayısı |
|---|---|---|---|
| OpenPose BODY_25 | ✅ | Tek görüntüde uygulanmadı | 25 |
| MediaPipe Pose | ✅ | ✅ | 33 |
| MMPose RTMPose | ✅ | ✅ | 17 |

## OpenPose

BODY_25 modeli ile 25 adet 2D vücut anahtar noktası çıkarılmıştır.
Native OpenPose 3D, senkronize ve kalibre edilmiş birden fazla kamera gerektirdiği için tek test görüntüsünde uygulanmamıştır.

## MediaPipe

Pose Landmarker ile 33 landmark çıkarılmıştır.
2D landmarkların yanında pose_world_landmarks kullanılarak tek görüntüden tahmini 3D vücut koordinatları elde edilmiştir.

## MMPose

RTMDet ile insan tespiti, RTMPose-M ile 17 adet 2D keypoint çıkarılmıştır.
Bu noktalar SimpleBaseline3D PoseLifter ile Human3.6M formatında 17 adet 3D keypointe dönüştürülmüştür.

Akış: Görüntü -> RTMDet -> RTMPose-M -> 2D Keypoint -> SimpleBaseline3D -> 3D Pose

## Genel Değerlendirme

- OpenPose BODY_25 ayrıntılı bir 2D iskelet yapısı sağlar.
- MediaPipe 33 landmark ile daha ayrıntılı landmark bilgisi sunar.
- MMPose 2D ve 3D modellerini ayrı aşamalarda kullanan modüler bir yapı sunar.
- MediaPipe ve MMPose ile tek görüntüden elde edilen 3D sonuçlar fiziksel derinlik ölçümü değildir; model tahminidir.
