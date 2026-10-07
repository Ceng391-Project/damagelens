# Sel — Sen1Floods11 (11 ülke), Valencia DANA 2024 ve xBD

## Özet
- **Uydu (Sentinel-2 optik + Sentinel-1 radar, 10 m), 11 ülke:** en iyi model **U-Net S1+S2 IoU 0.82** (F1 0.90); eğitimsiz NDWI eşiği bile IoU 0.78. Eğitimde hiç görülmeyen Bolivya olayında U-Net S1+S2 IoU 0.79.
- **Yakın tarihli büyük sel — Valencia, 29 Ekim 2024:** Copernicus EMS resmi taşkın alanına (≈324 km² inceleme alanında) karşı en iyi yöntem **U-Net S2 (Sen1Floods11'den aktarım): F1 0.66**. İndeks eşikleri çok isabetli (precision ≈0,9+) ama taşkının yalnızca ~üçte birini buluyor.
- **Kentsel sel uydudan zor:** Valencia'da yerleşim alanlarında recall 0.60, tarım alanında 0.83.
- **Radar tek başına:** 2 gün sonraki Sentinel-1 geçişinde su çekilmeye başladığı için recall 0.18; zamanlama kritik.
- **Bina hasarı (xBD sel/kasırga olayları):** sel kaynaklı bina hasarı en zor türlerden — ayrıntı aşağıda.

## Veri ve yer gerçeği
| set | platform | çözünürlük | yer gerçeği |
|---|---|---|---|
| Sen1Floods11 (HF `blumenstiel/Sen1Floods11`) | uydu: Sentinel-1 GRD + Sentinel-2 L1C | 10 m | 446 elle etiketli 512×512 kare, 11 olay |
| Valencia DANA, Ekim 2024 | uydu: Sentinel-2 L2A (31 Eki), Sentinel-1 RTC (25 Eki, 1 Kas) — Planetary Computer | 10 m | Copernicus EMS EMSR773 AOI01 taşkın sınırı |
| xBD: midwest-flooding, nepal-flooding, 4 kasırga | uydu: Maxar | ~0,5 m | bina hasar seviyeleri |

## Sonuçlar — Sen1Floods11 test (90 kare) ve Bolivya (eğitimde görülmemiş, 15 kare)
Eşikler doğrulama bölümünde seçildi (NDWI>-0.05, MNDWI>0.15, VV<-17 dB). "Sadece taşkın" = JRC kalıcı su pikselleri hariç.
| yöntem | test IoU | test F1 | test sadece-taşkın IoU | Bolivya IoU | Bolivya F1 |
|---|---|---|---|---|---|
| NDWI | 0.776 | 0.874 | 0.688 | 0.816 | 0.898 |
| MNDWI | 0.773 | 0.872 | 0.687 | 0.703 | 0.825 |
| S1 VV threshold | 0.468 | 0.638 | 0.391 | 0.388 | 0.559 |
| U-Net S1 | 0.659 | 0.794 | 0.533 | 0.633 | 0.775 |
| U-Net S2 | 0.810 | 0.895 | 0.734 | 0.793 | 0.884 |
| U-Net S1+S2 | 0.818 | 0.900 | 0.742 | 0.791 | 0.883 |

![Yöntem karşılaştırması](img/flood_iou_by_method.png)
*Yöntem karşılaştırması*

![Ülke/olay bazında IoU](img/flood_iou_by_country.png)
*Ülke/olay bazında IoU*

![Rastgele test kareleri](img/flood_samples.png)
*Rastgele test kareleri*

![U-Net eğitim eğrileri](img/flood_training.png)
*U-Net eğitim eğrileri*


## Vaka — Valencia DANA (29 Ekim 2024)
Modeller **yeniden eğitilmeden** Sen1Floods11'den aktarıldı (S2 modeli L1C ile eğitildi, burada L2A'ya uygulandı; B10 bandı eğitim ortalamasıyla dolduruldu). Kalıcı su (ESA WorldCover) hariç.
| yöntem | precision | recall | F1 | IoU | recall (yerleşim) | recall (tarım) |
|---|---|---|---|---|---|---|
| NDWI eşik (S2, 31 Eki) | 0.930 | 0.332 | 0.489 | 0.324 | 0.104 | 0.515 |
| MNDWI eşik (S2, 31 Eki) | 0.994 | 0.346 | 0.513 | 0.345 | 0.014 | 0.551 |
| U-Net S2 (Sen1Floods11'den aktarım) | 0.575 | 0.767 | 0.658 | 0.490 | 0.599 | 0.830 |
| S1 VV eşik (1 Kas) | 0.992 | 0.179 | 0.304 | 0.179 | 0.003 | 0.265 |
| S1 değişim VV(1 Kas)−VV(25 Eki) < −3 dB | 0.460 | 0.340 | 0.391 | 0.243 | 0.166 | 0.547 |
| U-Net S1 (Sen1Floods11'den aktarım) | 0.984 | 0.180 | 0.304 | 0.179 | 0.005 | 0.254 |

![Valencia: yöntemler ve arazi türüne göre recall](img/flood_valencia_metrics.png)
*Valencia: yöntemler ve arazi türüne göre recall*

![Valencia: görüntüler, EMS yer gerçeği ve en iyi tahmin](img/flood_valencia_maps.png)
*Valencia: görüntüler, EMS yer gerçeği ve en iyi tahmin*


## Sonuçlar — sel/kasırga kaynaklı bina hasarı (xBD)
**Model:** 6 kanallı (öncesi+sonrası RGB) U-Net/ResNet18, 5 sınıf (arka plan, hasarsız, az, ağır, yıkılmış). xBD train+tier3'ten olay başına ≤250 görüntü (3304 görüntü), 1024→512 küçültülerek (~1 m/px) 12 epoch eğitildi; en iyi doğrulama xView2 skoru 0.628. Test: xBD test bölümü + tier3 olaylarının ayrılmış %20'si (1313 görüntü). *Hasar sınıf F1* = 4 hasar sınıfının harmonik ortalaması (xView2 tanımı); *bina* metrikleri bağlantılı bina bileşenleri üzerinden.

| olay | bina | bina bulma F1 | hasar sınıf F1 | hasarlı/hasarsız F1 (bina) | yıkılmış recall | gerçek hasarlı payı |
|---|---|---|---|---|---|---|
| hurricane-florence | 1934 | 0.724 | 0.000 | 0.822 | 0.000 | 0.216 |
| hurricane-harvey | 6001 | 0.780 | 0.096 | 0.853 | 0.018 | 0.461 |
| hurricane-matthew | 2889 | 0.637 | 0.246 | 0.804 | 0.392 | 0.841 |
| hurricane-michael | 5305 | 0.752 | 0.239 | 0.527 | 0.052 | 0.352 |
| midwest-flooding | 2069 | 0.747 | 0.095 | 0.449 | 0.089 | 0.052 |
| nepal-flooding | 1884 | 0.726 | 0.185 | 0.586 | 0.000 | 0.265 |
| **flood (toplam)** | 3953 | 0.737 | 0.129 | 0.562 | 0.048 | 0.153 |
| **hurricane (toplam)** | 16129 | 0.755 | 0.420 | 0.742 | 0.260 | 0.464 |

![xBD sel olayları — rastgele test kareleri](img/flood_xbd_samples.png)
*xBD sel olayları — rastgele test kareleri*


## Sınırlar
- EMS yer gerçeği 30–31 Ekim Landsat-8/Sentinel-2'den üretildi; aynı Sentinel-2 sahnesini kullanan yöntemler bir miktar avantajlı, Sentinel-1 (1 Kas) ise dezavantajlı (su çekilmiş).
- EMS "gözlenen olay" katmanı çamur/taşkın izlerini de içerir; açık su indeksleri bunları görmez → düşük recall'ın bir kısmı tanım farkı.
- Sen1Floods11 kareleri 10 m; ince kentsel sokak taşkınları bu çözünürlükte görünmez.

## Kaynaklar
- Sen1Floods11 (Bonafilia ve ark., 2020): https://github.com/cloudtostreet/Sen1Floods11
- Copernicus EMS EMSR773: https://rapidmapping.emergency.copernicus.eu/EMSR773
- Sentinel-1/2, ESA WorldCover: Microsoft Planetary Computer
- xBD / xView2 (Gupta ve ark., 2019), Maxar Open Data görüntüleri, CC BY-NC-SA 4.0 — HF mirror `hannan022/xview2-xbd`
