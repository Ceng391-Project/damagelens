# Deprem — 6 Şubat 2023 Kahramanmaraş ve xBD depremleri

## Özet
- **6 Şubat (KATE-CD, Maxar + Pleiades, 0,3–0,5 m — Google Earth sınıfı görüntü):** Türkiye verisiyle eğitilen model test setinde **F1 0.55** (IoU 0.38); basit görüntü farkı 0.08.
- **Başka depremlerden öğrenme Türkiye'ye taşınmıyor:** xBD (Meksika 2017, Palu 2018 + diğer afetler) ile eğitilen 5 sınıflı model 6 Şubat'ta doğrudan **F1 0.11**.
- **xBD içi deprem/tsunami performansı:** Meksika depreminde bina bulma F1 0.76, ancak binaların yalnızca 0.9%'i hasarlı → hasar sınıfları çok seyrek. Palu tsunamisinde hasarlı/hasarsız bina F1 0.77.
- **Rastgele Maxar kareleri (Antakya, Kahramanmaraş, Gaziantep, İslahiye/Nurdağı):** yer gerçeği olmadığı için nitel test. İki bulgu: (1) Şubat görüntülerindeki kar ve bulut her iki modelde sahte hasarı artırıyor; (2) temiz kentsel karelerde iki model çok farklı tahmin veriyor (KATE modeli piksellerin ~%0,3'ünü, xBD modeli binaların ~%10'unu hasarlı sayıyor) — ağır hasarlı Antakya'da bile KATE modelinin neredeyse hiç hasar işaretlememesi, etiketli test setindeki F1'in yeni sahnelere doğrudan taşınmadığını gösteriyor (KATE-CD eğitim karelerinin hepsinde hasar vardı; model yüksek eşikle çalışıyor).

## Veri ve yer gerçeği
| set | platform | çözünürlük | yer gerçeği |
|---|---|---|---|
| KATE-CD (HF `CSCRS/kate-cd`) | uydu: Maxar Open Data + Airbus Pleiades | 0,3–0,5 m | elle çizilmiş hasarlı bina poligonları (486 çift, 7 il) |
| xBD: mexico-earthquake, palu-tsunami, sunda-tsunami | uydu: Maxar | ~0,5 m (modelde ~1 m) | bina poligonu + 4 seviye hasar |
| Maxar Open Data `Kahramanmaras-turkey-earthquake-23` | uydu: Maxar (WorldView/GeoEye) | ~0,3–0,5 m | yok (nitel test) |

> Google Earth görüntüleri kullanım koşulları gereği programla indirilemez; Google Earth'teki deprem sonrası çok yüksek çözünürlüklü görüntülerin büyük kısmı zaten Maxar kaynaklıdır. Burada aynı sınıftaki Maxar Open Data kullanıldı.

## Sonuçlar — 6 Şubat (KATE-CD test, 38 kare, piksel bazlı)
| yöntem | precision | recall | F1 | IoU |
|---|---|---|---|---|
| Görüntü farkı (Lab + histogram eşleme) | 0.045 | 0.502 | 0.083 | 0.043 |
| U-Net (ikili), sadece xBD (4 shard) → sıfır-atış | 0.096 | 0.130 | 0.110 | 0.058 |
| U-Net 5 sınıf, tüm xBD → sıfır-atış, 512 ölçek | 0.062 | 0.150 | 0.088 | 0.046 |
| U-Net 5 sınıf, tüm xBD → sıfır-atış, 256 ölçek (çözünürlük eşlenmiş) | 0.060 | 0.523 | 0.108 | 0.057 |
| U-Net, sadece KATE-CD (404 eğitim karesi) | 0.624 | 0.496 | 0.552 | 0.382 |
| U-Net, xBD ön eğitim + KATE-CD ince ayar | 0.571 | 0.444 | 0.500 | 0.333 |

![KATE-CD test: öncesi, sonrası, etiket ve üç modelin tahmini](img/eq_kate_predictions.png)
*KATE-CD test: öncesi, sonrası, etiket ve üç modelin tahmini*

![xBD ile eğitilmiş 5 sınıflı modelin 6 Şubat'a doğrudan uygulanması](img/eq_kate_zero_shot.png)
*xBD ile eğitilmiş 5 sınıflı modelin 6 Şubat'a doğrudan uygulanması*


## Sonuçlar — xBD deprem / tsunami olayları
**Model:** 6 kanallı (öncesi+sonrası RGB) U-Net/ResNet18, 5 sınıf (arka plan, hasarsız, az, ağır, yıkılmış). xBD train+tier3'ten olay başına ≤250 görüntü (3304 görüntü), 1024→512 küçültülerek (~1 m/px) 12 epoch eğitildi; en iyi doğrulama xView2 skoru 0.628. Test: xBD test bölümü + tier3 olaylarının ayrılmış %20'si (1313 görüntü). *Hasar sınıf F1* = 4 hasar sınıfının harmonik ortalaması (xView2 tanımı); *bina* metrikleri bağlantılı bina bileşenleri üzerinden.

| olay | bina | bina bulma F1 | hasar sınıf F1 | hasarlı/hasarsız F1 (bina) | yıkılmış recall | gerçek hasarlı payı |
|---|---|---|---|---|---|---|
| mexico-earthquake | 4745 | 0.763 | 0.000 | 0.043 | 0.000 | 0.009 |
| palu-tsunami | 3649 | 0.782 | 0.263 | 0.767 | 0.719 | 0.168 |
| sunda-tsunami | 906 | 0.729 | 0.000 | 0.093 | 0.069 | 0.040 |
| **earthquake (toplam)** | 4745 | 0.763 | 0.000 | 0.043 | 0.000 | 0.009 |
| **tsunami (toplam)** | 4555 | 0.776 | 0.253 | 0.741 | 0.686 | 0.142 |

![xBD Meksika depremi rastgele test kareleri](img/eq_xbd_samples.png)
*xBD Meksika depremi rastgele test kareleri*


## Rastgele vakalar — Maxar Open Data, 6 Şubat (Google Earth sınıfı)
Her şehirden rastgele 256 m × 256 m kareler (0,5 m/px), en yakın deprem öncesi ve ilk deprem sonrası Maxar görüntüsü.
Kareler WorldCover'a göre ikiye ayrıldı: *temiz kentsel* (yerleşim ≥%40, kar/bulut <%10) ve *diğer* (kırsal, karlı ya da bulutlu — Şubat 2023 görüntülerinin bir kısmında kar ve bulut var).
| şehir | temiz kentsel kare | KATE modeli: hasarlı piksel | xBD modeli: ağır+yıkık bina payı | diğer kare | KATE (diğer) | xBD (diğer) |
|---|---|---|---|---|---|---|
| Antakya (Hatay) | 3 | 0.005 | 0.087 | 10 | 0.002 | 0.175 |
| Kahramanmaraş merkez | 10 | 0.003 | 0.099 | 3 | 0.058 | 0.120 |
| Gaziantep merkez | 4 | 0.000 | 0.126 | 12 | 0.013 | 0.296 |
| İslahiye / Nurdağı | 0 | – | – | 16 | 0.015 | 0.063 |

**Başarısızlık modu:** kar/bulut ≥%10 olan 13 karede KATE modelinin ortalama hasar tahmini 0.012, xBD modelininki 0.25; temiz kentsel 17 karede sırasıyla 0.003 ve 0.10. Kar, öncesi–sonrası farkını büyüttüğü için "hasar" olarak okunuyor; operasyonel kullanımda kar/bulut maskesi şart.
![Şehir bazında ortalama tahmin](img/eq_maxar_cities.png)
*Şehir bazında ortalama tahmin*

![Her şehirden en yüksek KATE tahmini alan iki temiz kentsel kare + iki kar/bulutlu kare](img/eq_maxar_samples.png)
*Her şehirden en yüksek KATE tahmini alan iki temiz kentsel kare + iki kar/bulutlu kare*


## Sınırlar
- KATE-CD testi 38 kare; tek seed → sayılar ±birkaç puan oynayabilir. KATE-CD'de hasarsız kare yok, yanlış alarm oranı sınırlı ölçülür.
- xBD görüntüleri 2× küçültüldü (~1 m/px); orijinal çözünürlükte bina/hasar skorları daha yüksek olur.
- Maxar rastgele karelerinde yer gerçeği yok; öncesi/sonrası görüntüler farklı bakış açısı ve mevsimden — yanlış alarmlar görsel olarak kontrol edilmeli.

## Kaynaklar
- KATE-CD: https://huggingface.co/datasets/CSCRS/kate-cd (ITÜ CSCRS)
- xBD / xView2 (Gupta ve ark., 2019), Maxar Open Data görüntüleri, CC BY-NC-SA 4.0 — HF mirror `hannan022/xview2-xbd`
- Maxar Open Data Program: https://maxar-opendata.s3.amazonaws.com/events/catalog.json
