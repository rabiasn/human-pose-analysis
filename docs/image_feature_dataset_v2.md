# Resim klasöründen ortak öznitelik veri seti

Bu aşama hocanın istediği **resim → hazır pose modeli → noktalar → CSV** işlemidir.
MediaPipe, OpenPose, MMPose ve DeeperCut aynı resimleri işler. Yeni hareket veya
posture sınıflandırıcısı eğitilmez; SUPINE/SIDE_LYING tahmini bu aşamada yapılmaz.

## Kullanım

Proje dizininde mevcut ortamla çalıştır:

```bash
.venv_mediapipe/bin/python -B scripts/build_pose_image_dataset_v2.py \
  --input "/resim/klasorunun/tam/yolu" \
  --output "/yeni/sonuc/klasorunun/tam/yolu"
```

DeeperCut için Docker açık ve mevcut `pose-tf14:latest` imajı hazır olmalı.
Script paket kurmaz veya imaj indirmez. Çıktı klasörü önceden mevcut olmamalı ve
girdi klasörünün dışında olmalı; eski sonuçların üzerine yazılmaz.
`--output` verilmezse projenin `outputs/image_feature_datasets/` dizininde tarihli
yeni bir klasör açılır. PNG, JPG/JPEG, BMP, TIFF ve WEBP desteklenir. Alt klasörler
için `--recursive`, yalnız CSV/ham veri için `--no-previews` eklenir. İstenirse
`--libraries mediapipe openpose` gibi bir alt küme seçilebilir; varsayılan dört
kütüphanenin tamamıdır. HEIC gibi destek dışı dosyalar resim listesine alınmaz.

Resimler kayıpsız PNG olarak yeni çıktı klasörüne kopyalanır. Modelden önce
ölçekleme veya kırpma yapılmaz; bütün kütüphaneler aynı çözümlenmiş pikselleri
görür. Her resmin kendi genişliği/yüksekliği kullanılır. Aynı ada sahip alt klasör
resimleri göreli kaynak yolu ve benzersiz resim kimliğiyle ayrılır.

## Üç öznitelik yöntemi

Ortak 12 eklem: sağ/sol omuz, dirsek, el bileği, kalça, diz, ayak bileği.
Sağ/sol bebeğin anatomik tarafıdır. Nokta numaraları kütüphaneye göre eşlenir.
Yalnız **x,y** kullanılır; **z kullanılmaz**.

1. **Koordinatlar:** 12 noktanın 24 x,y değeri, resim pikseli cinsinden.
2. **Açılar:** 12 açı. A–B–C açısı B noktasındadır:
   `acos((|AB|² + |BC|² − |AC|²) / (2 |AB| |BC|))`, dereceye çevrilir.
   İki dirsek, iki omuz, iki kalça, iki diz ve gövde dörtgeninin dört köşesi.
3. **Uzunluk oranları:** 21 düz çizgi mesafesi / gövde referans uzunluğu.
   Gövde referansı omuzların orta noktası ile kalçaların orta noktası arasındadır.
   Eklem bağlantıları, gövde genişlikleri/köşegenleri, iki uzuv arasındaki
   mesafeler ve omuz–el bileği/kalça–ayak bileği açıklıkları kullanılır.

Her kütüphane için `xy`, `angles`, `ratios` CSV'si oluşur: **toplam 12 CSV**.
Hocanın açıları ve uzunlukları aynı satırda istemesi için ayrıca `combined/`
klasöründe **kütüphane başına bir birleşik CSV** bulunur. Bu CSV'de aynı 24
koordinat, 12 açı ve 21 oran birlikte yer alır; yeni bir yöntem değildir.
Her CSV'de her girdi resmi bir satırdır. Sayısal sütunların isimleri/formülleri
`feature_schema.json` içinde tanımlıdır. Kaynak dosya, model durumu ve QC
sütunları metadatadır; ileride sınıflandırıcıya sayısal özellik diye verilmez.

## Hazır modeller ve nokta seçimi

- MediaPipe: PoseLandmarker full, bağımsız IMAGE modu, en çok bir pose.
- OpenPose: mevcut BODY_25 CPU modeli, önceki deneyle aynı `-1x256` ağ boyutu;
  ortak eklemlerde ortalama native skoru en yüksek kişi seçilir.
- MMPose: RTMDet-M kişi tespiti + RTMPose-M COCO17; 0.3 üzeri en yüksek kişi
  tespit skoru seçilir. Manuel kutu veya başarısız tespitte ek pose yoktur.
- DeeperCut: projedeki pose-tensorflow DeeperCut/ArtTrack uygulaması, hazır COCO
  ResNet-101 + location/pairwise regression + multicut. En çok ortak eklem
  içeren küme seçilir; eşitlikte unary skor toplamı, sonra küme numarası kullanılır.
  Farklı kümelerin eklemleri sonradan birleştirilmez.

## Eksik noktalar ve doğrulama

Kütüphanenin noktaları elle değiştirilmez. Eksik nokta veya tanımsız hesaplama
**boş hücre** üretir; sıfırla doldurulmaz. Gerçek sıfır mesafe sıfır kalır. Bir
resim açı/uzunluk hesaplamak için yeterli noktayı vermeyebilir. Okunamayan resim
`INPUT_ERROR`, resme özgü model hatası `INFERENCE_ERROR`, pose bulunamaması
`NO_POSE` olarak kaydedilir. Modelin başlatılamaması tüm çalışmada açık hata
verir ve `run_failure.json` oluşturur; başarı olarak sunulmaz.

MP görünürlük 0.5, OP/MM native skor 0.3 seviyeleri yalnız ölçüm kontrolü
işaretidir; posture karar eşikleri değildir. Bu skorlar birbirinin yerine
kullanılamaz; RTMPose'un 1 üzeri native skorları değiştirilmez. DeeperCut'ın
native kişi koordinatlarında ayrı confidence kanalı yoktur. Ayrı kaydedilen
skor, ilgili ekleme atanmış detection unary skorlarının ortalamasıdır. Bir
ekleme birden fazla detection atanması veya birden çok body kümesi QC'de görünür.

Yeni bir klasörde görsel kontrol otomatik yapılmış sayılmaz. `visual_review`
alanı **NOT_REVIEWED** kalır. `NO_FLAG`, doğru nokta veya doğru posture kanıtı
değildir. Yan yana çizimler kontrol içindir. Kıyafet, örtüşme ve düşük görünürlük
özellikle bebeklerde eklem yerini belirsiz bırakabilir.

Açı ve oranlar aynı 2D noktaların eşit ölçeklenmesinden etkilenmez; kamera bakış
açısı değiştiğinde sabit kalmaları garanti değildir. Ham x,y ölçeğe bağlıdır.
Bu CSV'ler bir hareket tanıma başarısı veya bağımsız test doğruluğu göstermez.

Çıktıda `manifest.json` kaynak eşlemesini, `native/` ham model sonuçlarını,
`common_keypoints_xy.csv` ortak noktaları ve native skorları, `csv/` 12 yöntem
veri setini, `combined/` dört birleşik veri setini, `previews/` çizimleri,
`logs/` çalışma kayıtlarını içerir. Girdi/model
hash'leri ve CSV sayısal kayıpsızlık kontrolleri çalışma özetinde saklanır.

Video için sabit süreli kliplerden kare örnekleyip ölçümleri aynı satırda yan
yana yazma ve ardından sınıflandırma, hocanın tarif ettiği **sonraki aşamalardır**.

## Son sürümün doğrulaması

`outputs/image_feature_datasets/final_pilot_v2_release/` içinde aynı 10 resim
üzerinden 12 yöntem CSV'si ve dört birleşik CSV üretildi. Dokuz otomatik kontrol
geçti. Ayrıca altı farklı dosyayla alt klasörler, aynı dosya adları, Türkçe
karakterler, farklı boyutlar, EXIF yönü, boş resim ve bozuk resim sınandı.
Boş resim dört modelde de NO_POSE, bozuk resim INPUT_ERROR olarak kaldı.
CSV'ler bağımsız açı/mesafe hesaplamasıyla ve önceki pilot ölçümleriyle
karşılaştırıldı. Önceki proje kaynakları ve sonuçlarının 198 hash kontrolü geçti.

10 resimde **sayısal olarak elde edilebilen** değer sayıları:

| Kütüphane | Ortak nokta / 120 | x,y / 240 | Açı / 120 | Oran / 210 |
|---|---:|---:|---:|---:|
| MediaPipe | 120 | 240 | 120 | 210 |
| OpenPose | 119 | 238 | 119 | 207 |
| MMPose | 120 | 240 | 120 | 210 |
| DeeperCut | 75 | 150 | 56 | 62 |

Bu tablo doğruluk veya hareket tanıma başarısı değildir. Özellikle DeeperCut
iskeleti bazı resimlerde parçalı çıkarmıştır. Diğer modellerde de mevcut nokta
yanlış yere yerleşmiş olabilir; sayısal doluluk görsel doğrulama yerine geçmez.
Sınıflandırıcı eğitilmedi, orijinal resimler/noktalar değiştirilmedi. Ayrıntılı
doğrulama kaydı son çıktı klasörünün `validation.json` dosyasındadır.
