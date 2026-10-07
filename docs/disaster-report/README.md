# DamageLens — afet türlerine göre uydu ve hava görüntüsü analizi

Her rapor: veri ve yer gerçeği, yöntemler, metrik tabloları, grafikler, rastgele vaka görselleri, sınırlar ve kaynaklar.
Kod ve ham çıktılar: `experiments/multi-hazard/` (her deney bir betik, sonuçlar `experiments/multi-hazard/outputs/`).

![Genel bakış](img/overview.png)

| afet | rapor | platformlar | ana sonuç |
|---|---|---|---|
| Deprem | [01-deprem.md](01-deprem.md) | uydu VHR (Maxar/Pleiades — Google Earth sınıfı) | 6 Şubat F1 0.55; xBD'den aktarım F1 0.11 |
| Sel | [02-sel.md](02-sel.md) | uydu (Sentinel-1/2), uydu VHR (xBD) | 11 ülke IoU 0.82; Valencia 2024 F1 0.66 |
| Heyelan | [03-heyelan.md](03-heyelan.md) | uydu (Sentinel-2+DEM), hava (İHA) | uydu F1 0.63; İHA F1 0.82 |
| Hortum | [04-hortum.md](04-hortum.md) | uydu VHR (xBD), hava (NAIP uçak) | xBD bina F1 0.77; Rolling Fork EF3+ AUC 0.57 |
| Dolu | [05-dolu.md](05-dolu.md) | uydu (Sentinel-2) + radar MESH | ≥60 mm AUC 0.84; ≥25 mm AUC 0.61 |
| Aşırı sıcak | [06-asiri-sicak.md](06-asiri-sicak.md) | uydu (MODIS LST) + istasyon | r 0.65; gün tespiti AUC 0.82 |
| Yangın | [07-yangin.md](07-yangin.md) | uydu (Sentinel-2), uydu VHR (xBD) | yanık alanı IoU 0.54–0.86; xBD bina F1 0.79 |

**Google Earth hakkında:** Google Earth görüntüleri kullanım koşulları gereği toplu/otomatik indirilemez. Google Earth'teki afet sonrası çok yüksek çözünürlüklü görüntülerin önemli kısmı Maxar kaynaklıdır; bu çalışmada aynı sınıftaki Maxar Open Data (xBD, KATE-CD, Kahramanmaraş 2023 olayı) kullanıldı. Hava görüntüsü olarak İHA (heyelan) ve uçak (USDA NAIP, hortum) verisi test edildi; helikopter görüntüsü için açık, etiketli bir set bulunamadı.
