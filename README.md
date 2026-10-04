# Human Pose Analysis

Bilgisayar mühendisliği stajımda MediaPipe, OpenPose, MMPose ve DeeperCut ile bebek görüntülerinden vücut noktalarını çıkardım. Aynı görüntülerdeki sonuçları karşılaştırdım ve bu noktalardan sayısal özellikler hesaplayarak CSV dosyalarına kaydettim.

## Video karşılaştırması

`baby_test.mp4` üzerinde MediaPipe, OpenPose ve MMPose'u tam videoda, DeeperCut'ı ilk 10 saniyede çalıştırdım. Dört yöntemin aynı bölümdeki sonuçlarını [karşılaştırma videosunda](outputs/baby_test_10s_comparison.mp4) birleştirdim.

## Resimlerden veri seti oluşturma

Test videosundan beş sırtüstü (SUPINE), beş yan yatış (SIDE_LYING) örneği seçtim. Bu pozisyon adları resim seçimi için kullandığım gözlem notlarıdır; model tahmini değildir.

Dört kütüphanede sağ ve sol omuz, dirsek, el bileği, kalça, diz ve ayak bileğini eşledim. Aynı 12 eklem için yalnız x,y kullandım; z değerini almadım.

Her resim için üç şekilde özellik çıkardım:

1. **Koordinatlar:** 24 x,y değeri, piksel cinsinden.
2. **Açılar:** Kosinüs teoremiyle hesaplanan 12 eklem ve gövde açısı.
3. **Oranlar:** 21 mesafenin gövde uzunluğuna oranı. Referans olarak omuzların orta noktası ile kalçaların orta noktası arasındaki mesafeyi kullandım.

Her kütüphane için üç ayrı CSV ve bu 57 özelliği bir arada tutan bir birleşik CSV oluşturdum. Her resim bir satırdır. Hazır pose modellerini kullandım; bu resim çalışmasında yeni bir sınıflandırıcı eğitmedim ve noktaları elle düzeltmedim.

## Kod ve örnek sonuçlar

[Ana script](scripts/build_pose_image_dataset_v2.py) verilen resim klasörünü işler. Mevcut model dosyaları ve kütüphane ortamları gerekir; DeeperCut için Docker kullanılır.

```bash
.venv_mediapipe/bin/python -B scripts/build_pose_image_dataset_v2.py \
  --input "/resim/klasoru" \
  --output "/yeni/sonuc/klasoru"
```

[10 resimlik sonuçlarda](outputs/image_feature_datasets/final_pilot_v2_release/) `csv/` ayrı yöntem dosyalarını, `combined/` birleşik dosyaları, `previews/` yan yana çizimleri içerir. [Kullanım ayrıntıları](docs/image_feature_dataset_v2.md) ayrı dosyadadır.

## Kontroller

Dokuz otomatik kontrol geçti. Eksik noktaları ve hesaplanamayan değerleri boş bıraktım. Boş ve bozuk resimleri ayrıca denedim. CSV değerlerini ham noktalar ve bağımsız açı/mesafe hesabıyla kontrol ettim.

MediaPipe ve MMPose 120 ortak noktanın tamamında, OpenPose 119'unda, DeeperCut 75'inde sayısal değer verdi. Bu sayılar doğruluk oranı değildir. Örneğin 10. resimde MediaPipe'ın diz noktası yanlış yerde görünüyor; yüksek görünürlük değeri doğru yerleşimi garanti etmiyor. Açı ve oranlar eşit ölçeklemeye dayanıklıdır, farklı kamera açılarında değişebilir.
