# Hortum — xBD (Joplin, Moore, Tuscaloosa) ve Rolling Fork 2023 (hava fotoğrafı)

## Özet
- **Uydu (Maxar, xBD) — 3 hortum olayı:** bina bulma F1 0.78, hasarlı/hasarsız bina F1 **0.77**, yıkılmış bina recall 0.71.
- **Hava fotoğrafı (NAIP uçak görüntüsü, 2021 öncesi / Ağustos 2023 sonrası) — Rolling Fork EF4 hortumu (24 Mart 2023):** xBD uydu modeli hiç eğitim almadan uygulandı; ayırt etme gücü zayıf ama yönü doğru (ortalama skor EF0'dan EF4'e artıyor): NWS saha ekiplerinin 341 yapı hasar noktasına karşı **EF3+ ile EF0–1 ayrımı AUC 0.57**, EF2+ ile hasarsız kontrol binaları AUC 0.58; EF derecesi ile model skoru Spearman ρ=0.14.

## Veri ve yer gerçeği
| set | platform | çözünürlük | yer gerçeği |
|---|---|---|---|
| xBD tier3: joplin, moore, tuscaloosa | uydu: Maxar | ~0,5 m (modelde ~1 m) | bina hasar seviyeleri |
| NAIP Mississippi 2021-11 ve 2023-08 (Planetary Computer) | hava: uçak (USDA NAIP) | 0,3 m / 0,6 m → 1 m'ye örneklendi | NOAA NWS Damage Assessment Toolkit: EF dereceli yapı noktaları |

## Sonuçlar — xBD hortum olayları
**Model:** 6 kanallı (öncesi+sonrası RGB) U-Net/ResNet18, 5 sınıf (arka plan, hasarsız, az, ağır, yıkılmış). xBD train+tier3'ten olay başına ≤250 görüntü (3304 görüntü), 1024→512 küçültülerek (~1 m/px) 12 epoch eğitildi; en iyi doğrulama xView2 skoru 0.628. Test: xBD test bölümü + tier3 olaylarının ayrılmış %20'si (1313 görüntü). *Hasar sınıf F1* = 4 hasar sınıfının harmonik ortalaması (xView2 tanımı); *bina* metrikleri bağlantılı bina bileşenleri üzerinden.

| olay | bina | bina bulma F1 | hasar sınıf F1 | hasarlı/hasarsız F1 (bina) | yıkılmış recall | gerçek hasarlı payı |
|---|---|---|---|---|---|---|
| joplin-tornado | 2795 | 0.772 | 0.652 | 0.840 | 0.760 | 0.423 |
| moore-tornado | 4756 | 0.804 | 0.330 | 0.705 | 0.664 | 0.097 |
| tuscaloosa-tornado | 2424 | 0.754 | 0.497 | 0.701 | 0.504 | 0.256 |
| **tornado (toplam)** | 9975 | 0.783 | 0.596 | 0.772 | 0.705 | 0.227 |

![xBD hortum olayları — rastgele test kareleri](img/tornado_xbd_samples.png)
*xBD hortum olayları — rastgele test kareleri*


## Vaka — Rolling Fork, Mississippi (EF4, 24 Mart 2023), hava fotoğrafı
Her NWS hasar noktası için 256 m × 256 m öncesi/sonrası NAIP kesiti alındı; skor = noktanın 20 m çevresinde modelin *ağır hasar + yıkılmış* olasılığı (bina pikselleri ağırlıklı). Kontroller: hasar noktalarına >1,5 km uzaklıkta, modelin bina bulduğu rastgele noktalar (93 adet).
| seviye | nokta sayısı | ortalama skor |
|---|---|---|
| none | 93 | 0.296 |
| EF0 | 21 | 0.249 |
| EF1 | 130 | 0.291 |
| EF2 | 118 | 0.359 |
| EF3 | 45 | 0.351 |
| EF4 | 27 | 0.373 |

![Rolling Fork: EF derecesine göre model skoru ve AUC](img/tornado_rf_metrics.png)
*Rolling Fork: EF derecesine göre model skoru ve AUC*

![Rastgele NAIP kesitleri (camgöbeği daire = 20 m değerlendirme çevresi)](img/tornado_rf_samples.png)
*Rastgele NAIP kesitleri (camgöbeği daire = 20 m değerlendirme çevresi)*


## Sınırlar
- NAIP sonrası görüntü olaydan ~4,5 ay sonra: enkaz kaldırılmış, bazı yapılar onarılmış olabilir → hasar düşük görünür.
- NAIP (uçak, nadir) ile Maxar (uydu, eğik bakış) arasında renk/ölçek farkı; model hava fotoğrafı ile eğitilmedi.
- NWS noktaları yalnızca hasar görmüş yapılardır; "hasarsız" sınıfı kontrol noktalarıyla yaklaşık temsil edildi.
- **Gözlenen başarısızlık modu:** EF4 noktalarında enkaz Ağustos 2023'e kadar kaldırılmış; model boş arsayı "bina yok" olarak okuyor ve hasar skoru düşük kalıyor (örnek görsellerin ilk ve üçüncü satırı). xBD'de yıkılmış bina hep enkazla görüldüğü için model "kaybolan binayı" hasar olarak öğrenmemiş. Öncesi görüntüdeki bina konumuna göre puanlama bu hatayı azaltır.

## Kaynaklar
- xBD / xView2 (Gupta ve ark., 2019), Maxar Open Data görüntüleri, CC BY-NC-SA 4.0 — HF mirror `hannan022/xview2-xbd`
- NAIP: Microsoft Planetary Computer `naip`
- NWS Damage Assessment Toolkit: https://apps.dat.noaa.gov/stormdamage/damageviewer/
