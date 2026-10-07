# Disaster Damage Assessment from Satellite and Aerial Images

CENG391 Introduction to Image Understanding, 2026 Fall, term project **#30**, group G15 (3 kişi).

> Develop a change/damage assessment system using pre-event and post-event aerial/satellite images. Align image pairs, identify changed regions, classify or segment damaged structures/areas, and summarize damage spatially. Compare simple image-difference/feature baselines with a learned change-detection approach.

Ana vaka: **6 Şubat 2023 Kahramanmaraş depremleri**. Ek olarak sistem sel, heyelan, hortum, dolu, aşırı sıcak ve yangın üzerinde de denendi (`disaster-eval/`).

## Durum

Fizibilite tamamlandı: ödevin her adımı açık veriyle uçtan uca çalıştırıldı.

| Ödev adımı | Kodda | Şu anki sonuç |
|---|---|---|
| Öncesi/sonrası görüntü | `download_data.sh`, `disaster-eval/maxar_index.py` | KATE-CD (Maxar + Pleiades, 0,3–0,5 m), xBD, ham Maxar Open Data, NAIP uçak görüntüsü |
| Görüntü çiftlerini hizalama | `feasibility/04_gaps.py`, `05_spatial_summary.py` | Faz korelasyonu. Ham Maxar çiftlerinde medyan 16 px (~8 m) kayma ölçüldü |
| Değişen bölgeleri bulma, hasarı segmentleme | `feasibility/02_train.py`, `disaster-eval/xbd_train.py` | KATE-CD test F1 **0.55**; xBD 5 sınıf xView2 skoru **0.62** |
| Mekânsal özet | `feasibility/05_spatial_summary.py` | Kahramanmaraş merkezi 1,5 km², 48 m hücre ısı haritası |
| Baseline vs öğrenilmiş model | `feasibility/01_stats_baseline.py`, `04_gaps.py` | 5 klasik yöntem F1 0.08–0.09, U-Net 0.55 |

![KATE-CD tahminleri](docs/figures/kate_predictions.png)
![Klasik yöntemler ve hizalama hatası](docs/figures/baselines_alignment.png)
![Ham Maxar görüntüsünden mekânsal özet](docs/figures/spatial_summary_kahramanmaras.png)

### 6 Şubat — KATE-CD test (38 kare, piksel bazlı)
| Yöntem | F1 | IoU |
|---|---|---|
| Görüntü farkı (Lab + histogram eşleme) | 0.083 | 0.043 |
| CVA / 1−SSIM / PCA-kmeans / CVA+Otsu | 0.083–0.094 | 0.043–0.050 |
| U-Net, xBD ile eğitilmiş, Türkiye'ye doğrudan | 0.108 | 0.057 |
| U-Net, xBD ön eğitim + KATE-CD ince ayar | 0.500 | 0.333 |
| **U-Net, sadece KATE-CD** | **0.552** | **0.382** |

### Diğer afet türleri (özet)
Ayrıntılı raporlar (her afet için yöntem, tablolar, grafikler, rastgele vakalar, sınırlar): [docs/disaster-report/](docs/disaster-report/README.md). Tek sayfalık sürüm: `docs/disaster-report/index.html` (yerelde tarayıcıda açılır).

![Afet türlerine göre](docs/figures/disaster_overview.png)

| Afet | Veri | Ana sonuç |
|---|---|---|
| Sel | Sen1Floods11 (11 ülke), Valencia DANA 2024 vs Copernicus EMS | IoU 0.82; Valencia F1 0.66 |
| Heyelan | Landslide4Sense (uydu), İHA seti | F1 0.63 / 0.82 |
| Hortum | xBD; Rolling Fork 2023 NAIP uçak görüntüsü vs NWS EF noktaları | bina F1 0.77; uçak görüntüsünde AUC 0.57 |
| Dolu | Sentinel-2 ΔNDVI vs NOAA MESH, Nebraska 2022 | ≥60 mm AUC 0.84, ≥25 mm ≈0.6 |
| Aşırı sıcak | MODIS LST vs istasyon, 10 şehir, yaz 2023 | r 0.65, gün tespiti AUC 0.82 |
| Yangın | Sentinel-2 dNBR vs NIFC/EFFIS sınırları; xBD | IoU 0.54–0.86; bina F1 0.79 |

![xBD afet türüne göre](docs/figures/xbd_per_type.png)

## Kurulum

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
./download_data.sh kate        # ~450 MB, ana deney için yeterli
./download_data.sh maxar       # ham Maxar sahneleri için STAC indeksi (görüntü anında okunur)
```

Diğer setler: `./download_data.sh xbd` (~24 GB), `flood`, `landslide`, `valencia`, `all`. Dolu, sıcak ve yangın verileri kendi betiklerinde Planetary Computer, NOAA ve Meteostat'tan gerektiğinde çekilir.

## Çalıştırma

```bash
python feasibility/01_stats_baseline.py           # istatistik + görüntü farkı baseline
bash feasibility/run_all.sh                       # U-Net: KATE, xBD, xBD→KATE (~16 dk/koşu, M4)
python feasibility/03_figure.py
python feasibility/04_gaps.py                     # klasik baseline'lar + hizalama deneyi
python feasibility/05_spatial_summary.py          # ham Maxar → hizalama → model → hasar haritası
```

Çoklu afet deneyleri ve rapor üretimi: `disaster-eval/README.md`. Önce `xbd_prep.py → xbd_train.py` (~1,5 saat), sonra `gpu_queue.sh`. GPU işlerini aynı anda çalıştırmayın, 16 GB'lık bir Mac swap'a düşüyor.

## Repo yapısı

```
feasibility/      KATE-CD deneyleri: baseline'lar, U-Net, hizalama, mekânsal özet
disaster-eval/    xBD 5 sınıf model + sel, heyelan, hortum, dolu, sıcak, yangın deneyleri, rapor üreticileri
docs/figures/     README grafikleri
download_data.sh  veri indirme
```
`data/` ve `outputs/` klasörleri, model ağırlıkları (`*.pt`) ve ara dosyalar git'e girmez.

## Dikkat edilmesi gerekenler

- **Etiketli testteki başarı ham sahneye taşınmıyor.** KATE-CD'de F1 0.55 alan model, ham Maxar görüntüsünde ağır hasarlı Kahramanmaraş merkezinde piksellerin yalnızca ~%0,4'ünü hasarlı buluyor. Projenin asıl problemi bu.
- **KATE-CD'nin her karesinde hasar var.** Model hiç hasarsız sahne görmedi, yanlış alarm oranı ölçülemiyor. Eğitime hasarsız negatifler eklenmeli.
- **KATE-CD koordinatsız**, mekânsal özetin sayısal doğrulaması için koordinatlı bir yer gerçeği lazım. Türkiye için bina bazlı başka açık bir set bulamadık: HOT OSM dosyası boş çıktı, Copernicus EMS EMSR648 giriş istiyor.
- **Mevsim, bakış açısı, kar ve bulut:** öncesi görüntüler yaz 2022, sonrası Şubat 2023. Kar ve bulut her iki modelde sahte hasar üretiyor. Mümkünse Aralık 2022 ve Ocak 2023 öncesi kareleri kullanın.
- **Hizalama:** 16 px kayma F1'i 0.55'ten 0.36'ya düşürüyor. Faz korelasyonu 0.49'a geri getiriyor; yüksek binalarda parallaks yüzünden tek bir kaydırma yetmiyor.
- **Başka afetlerden öğrenme taşınmıyor:** xBD'den Türkiye'ye doğrudan uygulama F1 0.11, xBD ön eğitimi ince ayarda da fayda vermedi.
- **Test seti küçük (38 kare):** sonuçları 3 tekrar ya da k-fold ile, ortalama ± standart sapma olarak verin.
- **Lisanslar:** xBD CC BY-NC-SA 4.0, Maxar Open Data CC BY-NC 4.0, ikisi de yalnız ticari olmayan kullanım için. KATE-CD'nin lisansı belirtilmemiş, yazarlara sorulmalı.

## Yapılacaklar

Rol önerisi: **(A)** veri + hizalama + baseline'lar · **(B)** öğrenilmiş model · **(C)** değerlendirme + mekânsal özet + arayüz + rapor.

- [ ] (A) KATE-CD lisansını ve orijinal karelerde koordinat olup olmadığını ITÜ CSCRS'e sor
- [ ] (A) Öznitelik tabanlı hizalama ekle (SIFT/ORB + RANSAC), faz korelasyonuyla karşılaştır
- [ ] (A) Kar/bulut maskesi ve depreme yakın tarihli öncesi görüntü seçimi
- [ ] (A) Hasarsız negatif kareler: Maxar'ın hasar görmemiş mahallelerinden ve xBD'nin hasarsız binalarından
- [ ] (B) Eğitimde ±16 px kaydırma, renk ve gölge augmentation'ı
- [ ] (B) Siamese / değişim odaklı bir model (örn. ChangeFormer) ile 6 kanallı U-Net'i karşılaştır
- [ ] (B) Tam çözünürlüklü xBD ile ön eğitim ve sınıf dengesizliği için focal loss
- [ ] (C) 3 seed veya k-fold değerlendirme, ortalama ± standart sapma
- [ ] (C) Ham Maxar sahneleri için etiketli küçük bir test alanı (birkaç yüz bina, elle) ya da koordinatlı yer gerçeği
- [ ] (C) Mahalle/il düzeyinde hasar özeti ve basit bir harita arayüzü
- [ ] (C) Final rapor ve sunum

## Veri kaynakları

- KATE-CD: https://huggingface.co/datasets/CSCRS/kate-cd (ITÜ CSCRS)
- xBD / xView2: https://xview2.org, mirror https://huggingface.co/datasets/hannan022/xview2-xbd
- Maxar Open Data Program: https://maxar-opendata.s3.amazonaws.com/events/catalog.json
- Sen1Floods11, Landslide4Sense, İHA heyelan seti, Copernicus EMS EMSR773, NOAA MRMS/SPC/NWS DAT, MODIS LST, Meteostat, NIFC, EFFIS: ayrıntılar `disaster-eval/` betiklerinde
