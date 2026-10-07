# Dolu — Nebraska/Iowa, Haziran 2022 (Sentinel-2 vs radar MESH)

## Özet
- Dolu doğrudan "görülmez"; uydu yalnızca **bitki örtüsündeki hasarı** (NDVI düşüşü) görebilir. Yer gerçeği olarak NOAA MRMS **MESH** (radardan tahmini en büyük dolu çapı, ~1 km) kullanıldı; konumu SPC dolu raporlarıyla doğrulandı (raporlardaki MESH medyanı ~36 mm).
- **Küçük/orta dolu (≥25 mm) neredeyse ayırt edilemiyor:** AUC 0.51–0.61.
- **Büyük dolu görülebiliyor:** ≥60 mm için AUC **0.84**'e kadar; dolu çapı arttıkça ΔNDVI düzenli artıyor.
- Sonuç tarih penceresine çok duyarlı: Haziran'da ekin hızla büyüdüğü için "öncesi" ile "sonrası" arası uzadıkça sinyal bozuluyor.

## Veri ve yer gerçeği
| veri | platform | çözünürlük |
|---|---|---|
| Sentinel-2 L2A (B04, B08, SCL bulut maskesi) — Planetary Computer | uydu | 10 m → 0,0025° (~250 m) hücreye ortalandı |
| ESA WorldCover 2021 (tarım alanı maskesi) | uydu ürünü | 10 m |
| NOAA MRMS MESH_Max_1440min (IEM arşivi) | yer radarı | 0,01° (~1 km) |
| NOAA SPC dolu raporları | gözlemci | nokta |

## Sonuçlar
Pozitif: MESH ≥ 25 mm; negatif: aynı dönemde MESH < 10 mm. Skor: ΔNDVI = NDVI(öncesi) − NDVI(sonrası), sadece tarım hücreleri.
| konfigürasyon | geçerli hücre | dolu hücresi | AUC ≥25 mm | AUC ≥40 mm | AUC ≥60 mm | Spearman (MESH, ΔNDVI) | en iyi F1* |
|---|---|---|---|---|---|---|---|
| A: tek olay (14 Haz), öncesi 11–14 / sonrası 16–19 Haz | 467573 | 71535 | 0.512 | 0.597 | 0.704 | 0.111 | 0.415 |
| C: 6–14 Haz olayları, öncesi 1–5 / sonrası 11–14 Haz | 283310 | 65626 | 0.612 | 0.715 | 0.836 | 0.135 | 0.525 |
| B: 6–19 Haz olayları, öncesi 1–5 / sonrası 16–19 Haz | 374634 | 142313 | 0.526 | 0.626 | 0.783 | 0.126 | 0.576 |

\* F1 eşiği aynı veride seçildi (iyimser); asıl karşılaştırma ölçütü eşikten bağımsız AUC.

![Konfigürasyon C: ROC, MESH sınıfına göre ΔNDVI, eşik eğrileri](img/hail_C_metrics.png)
*Konfigürasyon C: ROC, MESH sınıfına göre ΔNDVI, eşik eğrileri*

![Konfigürasyon C: NDVI öncesi/sonrası, MESH ve ΔNDVI haritaları](img/hail_C_maps.png)
*Konfigürasyon C: NDVI öncesi/sonrası, MESH ve ΔNDVI haritaları*

![Konfigürasyon A: 14 Haziran olayı](img/hail_A_maps.png)
*Konfigürasyon A: 14 Haziran olayı*


## Sınırlar
- MESH bir radar tahminidir (dolu çapını sıklıkla abartır, yere düşen doluyu doğrudan ölçmez) — "gerçek" değil, en iyi mevcut alan ölçümü.
- ~250 m hücreler tarla sınırlarını karıştırır; tarla bazlı analiz ve daha uzun süreli zaman serisi sinyali güçlendirir.
- Haziran başında mısır/soya küçük olduğu için hasar NDVI'da sınırlı görünür; Temmuz–Ağustos olayları daha belirgin olabilir.

## Kaynaklar
- NOAA MRMS (IEM arşivi): https://mtarchive.geol.iastate.edu/
- NOAA SPC Storm Reports: https://www.spc.noaa.gov/climo/reports/
- CIMSS uydu blogu, Nebraska/Iowa dolu izleri (Haziran 2022): https://cimss.ssec.wisc.edu/satellite-blog/archives/46975
