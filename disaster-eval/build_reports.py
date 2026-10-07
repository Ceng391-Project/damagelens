import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
O = ROOT / "outputs"
F = ROOT.parent / "feasibility" / "outputs"
R = ROOT.parent / "docs" / "disaster-report"
IMG = R / "img"
IMG.mkdir(parents=True, exist_ok=True)
J = lambda p: json.load(open(p))


def img(src, name, caption):
    shutil.copy(src, IMG / name)
    return f"![{caption}](img/{name})\n*{caption}*\n"


def table(header, rows):
    out = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        out += "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n"
    return out


xe = J(O / "xbd/eval_xbd_unet5.json")
xtl = J(O / "xbd/train_log.json")


def xbd_rows(types):
    rows = []
    for ev, m in sorted(xe["per_event"].items()):
        if m["type"] in types:
            rows.append([ev, m["buildings"], m["loc_f1"], m["dmg_f1"], m["building_damaged_f1"], m["destroyed_recall"], m["gt_damaged_share"]])
    for t in types:
        if t in xe["per_type"]:
            m = xe["per_type"][t]
            rows.append([f"**{t} (toplam)**", m["buildings"], m["loc_f1"], m["dmg_f1"], m["building_damaged_f1"], m["destroyed_recall"], m["gt_damaged_share"]])
    return table(["olay", "bina", "bina bulma F1", "hasar sınıf F1", "hasarlı/hasarsız F1 (bina)", "yıkılmış recall", "gerçek hasarlı payı"], rows)


XBD_NOTE = (f"**Model:** 6 kanallı (öncesi+sonrası RGB) U-Net/ResNet18, 5 sınıf (arka plan, hasarsız, az, ağır, yıkılmış). "
            f"xBD train+tier3'ten olay başına ≤250 görüntü ({xtl['n_train']} görüntü), 1024→512 küçültülerek (~1 m/px) 12 epoch eğitildi; "
            f"en iyi doğrulama xView2 skoru {max(h['val_xview2'] for h in xtl['hist']):.3f}. Test: xBD test bölümü + tier3 olaylarının ayrılmış %20'si "
            f"({xe['n_test_images']} görüntü). *Hasar sınıf F1* = 4 hasar sınıfının harmonik ortalaması (xView2 tanımı); *bina* metrikleri bağlantılı bina bileşenleri üzerinden.\n")
SRC_XBD = "- xBD / xView2 (Gupta ve ark., 2019), Maxar Open Data görüntüleri, CC BY-NC-SA 4.0 — HF mirror `hannan022/xview2-xbd`\n"
reports = {}

# ---------------- EARTHQUAKE ----------------
fb = J(F / "01_stats_baseline.json"); k1, k2, k3 = J(F / "kate_only.json"), J(F / "xbd_only.json"), J(F / "xbd_then_kate.json")
kz = xe["kate_zero_shot"]; mx = J(O / "earthquake_maxar/results.json")
eq = xe["per_type"].get("earthquake", {}); ts = xe["per_type"].get("tsunami", {})
best_kz = max(kz.values(), key=lambda v: v["f1"])
reports["01-deprem"] = f"""# Deprem — 6 Şubat 2023 Kahramanmaraş ve xBD depremleri

## Özet
- **6 Şubat (KATE-CD, Maxar + Pleiades, 0,3–0,5 m — Google Earth sınıfı görüntü):** Türkiye verisiyle eğitilen model test setinde **F1 {k1['test']['f1']:.2f}** (IoU {k1['test']['iou']:.2f}); basit görüntü farkı {fb['baseline_test']['f1']:.2f}.
- **Başka depremlerden öğrenme Türkiye'ye taşınmıyor:** xBD (Meksika 2017, Palu 2018 + diğer afetler) ile eğitilen 5 sınıflı model 6 Şubat'ta doğrudan **F1 {best_kz['f1']:.2f}**.
- **xBD içi deprem/tsunami performansı:** Meksika depreminde bina bulma F1 {xe['per_event'].get('mexico-earthquake', {}).get('loc_f1', float('nan')):.2f}, ancak binaların yalnızca {xe['per_event'].get('mexico-earthquake', {}).get('gt_damaged_share', float('nan')):.1%}'i hasarlı → hasar sınıfları çok seyrek. Palu tsunamisinde hasarlı/hasarsız bina F1 {xe['per_event'].get('palu-tsunami', {}).get('building_damaged_f1', float('nan')):.2f}.
- **Rastgele Maxar kareleri (Antakya, Kahramanmaraş, Gaziantep, İslahiye/Nurdağı):** yer gerçeği olmadığı için nitel test. İki bulgu: (1) Şubat görüntülerindeki kar ve bulut her iki modelde sahte hasarı artırıyor; (2) temiz kentsel karelerde iki model çok farklı tahmin veriyor (KATE modeli piksellerin ~%0,3'ünü, xBD modeli binaların ~%10'unu hasarlı sayıyor) — ağır hasarlı Antakya'da bile KATE modelinin neredeyse hiç hasar işaretlememesi, etiketli test setindeki F1'in yeni sahnelere doğrudan taşınmadığını gösteriyor (KATE-CD eğitim karelerinin hepsinde hasar vardı; model yüksek eşikle çalışıyor).

## Veri ve yer gerçeği
| set | platform | çözünürlük | yer gerçeği |
|---|---|---|---|
| KATE-CD (HF `CSCRS/kate-cd`) | uydu: Maxar Open Data + Airbus Pleiades | 0,3–0,5 m | elle çizilmiş hasarlı bina poligonları (486 çift, 7 il) |
| xBD: mexico-earthquake, palu-tsunami, sunda-tsunami | uydu: Maxar | ~0,5 m (modelde ~1 m) | bina poligonu + 4 seviye hasar |
| Maxar Open Data `Kahramanmaras-turkey-earthquake-23` | uydu: Maxar (WorldView/GeoEye) | ~0,3–0,5 m | yok (nitel test) |

> Google Earth görüntüleri kullanım koşulları gereği programla indirilemez; Google Earth'teki deprem sonrası çok yüksek çözünürlüklü görüntülerin büyük kısmı zaten Maxar kaynaklıdır. Burada aynı sınıftaki Maxar Open Data kullanıldı.

## Sonuçlar — 6 Şubat (KATE-CD test, 38 kare, piksel bazlı)
{table(["yöntem", "precision", "recall", "F1", "IoU"], [
    ["Görüntü farkı (Lab + histogram eşleme)", fb['baseline_test']['precision'], fb['baseline_test']['recall'], fb['baseline_test']['f1'], fb['baseline_test']['iou']],
    ["U-Net (ikili), sadece xBD (4 shard) → sıfır-atış", k2['test']['precision'], k2['test']['recall'], k2['test']['f1'], k2['test']['iou']],
    [f"U-Net 5 sınıf, tüm xBD → sıfır-atış, 512 ölçek", kz['scale_512']['precision'], kz['scale_512']['recall'], kz['scale_512']['f1'], kz['scale_512']['iou']],
    [f"U-Net 5 sınıf, tüm xBD → sıfır-atış, 256 ölçek (çözünürlük eşlenmiş)", kz['scale_256']['precision'], kz['scale_256']['recall'], kz['scale_256']['f1'], kz['scale_256']['iou']],
    ["U-Net, sadece KATE-CD (404 eğitim karesi)", k1['test']['precision'], k1['test']['recall'], k1['test']['f1'], k1['test']['iou']],
    ["U-Net, xBD ön eğitim + KATE-CD ince ayar", k3['test']['precision'], k3['test']['recall'], k3['test']['f1'], k3['test']['iou']],
])}
{img(F / "03_predictions.png", "eq_kate_predictions.png", "KATE-CD test: öncesi, sonrası, etiket ve üç modelin tahmini")}
{img(O / "xbd/kate_zero_shot.png", "eq_kate_zero_shot.png", "xBD ile eğitilmiş 5 sınıflı modelin 6 Şubat'a doğrudan uygulanması")}

## Sonuçlar — xBD deprem / tsunami olayları
{XBD_NOTE}
{xbd_rows(["earthquake", "tsunami"])}
{img(O / "xbd/samples_earthquake.png", "eq_xbd_samples.png", "xBD Meksika depremi rastgele test kareleri")}

## Rastgele vakalar — Maxar Open Data, 6 Şubat (Google Earth sınıfı)
Her şehirden rastgele 256 m × 256 m kareler (0,5 m/px), en yakın deprem öncesi ve ilk deprem sonrası Maxar görüntüsü.
Kareler WorldCover'a göre ikiye ayrıldı: *temiz kentsel* (yerleşim ≥%40, kar/bulut <%10) ve *diğer* (kırsal, karlı ya da bulutlu — Şubat 2023 görüntülerinin bir kısmında kar ve bulut var).
{table(["şehir", "temiz kentsel kare", "KATE modeli: hasarlı piksel", "xBD modeli: ağır+yıkık bina payı", "diğer kare", "KATE (diğer)", "xBD (diğer)"],
       [[c, v['clean_urban']['tiles'], v['clean_urban']['kate_damage_frac_mean'] if v['clean_urban']['tiles'] else '–', v['clean_urban']['xbd_damaged_share_mean'] if v['clean_urban']['tiles'] else '–',
         v['other']['tiles'], v['other']['kate_damage_frac_mean'] if v['other']['tiles'] else '–', v['other']['xbd_damaged_share_mean'] if v['other']['tiles'] else '–'] for c, v in mx.items() if not c.startswith('_')])}
**Başarısızlık modu:** kar/bulut ≥%10 olan {mx['_all']['n_snow_cloud']} karede KATE modelinin ortalama hasar tahmini {mx['_all']['kate_on_snow_cloud']:.3f}, xBD modelininki {mx['_all']['xbd_on_snow_cloud']:.2f}; temiz kentsel {mx['_all']['n_clean']} karede sırasıyla {mx['_all']['kate_on_clean']:.3f} ve {mx['_all']['xbd_on_clean']:.2f}. Kar, öncesi–sonrası farkını büyüttüğü için "hasar" olarak okunuyor; operasyonel kullanımda kar/bulut maskesi şart.
{img(O / "earthquake_maxar/city_summary.png", "eq_maxar_cities.png", "Şehir bazında ortalama tahmin")}
{img(O / "earthquake_maxar/samples.png", "eq_maxar_samples.png", "Her şehirden en yüksek KATE tahmini alan iki temiz kentsel kare + iki kar/bulutlu kare")}

## Sınırlar
- KATE-CD testi 38 kare; tek seed → sayılar ±birkaç puan oynayabilir. KATE-CD'de hasarsız kare yok, yanlış alarm oranı sınırlı ölçülür.
- xBD görüntüleri 2× küçültüldü (~1 m/px); orijinal çözünürlükte bina/hasar skorları daha yüksek olur.
- Maxar rastgele karelerinde yer gerçeği yok; öncesi/sonrası görüntüler farklı bakış açısı ve mevsimden — yanlış alarmlar görsel olarak kontrol edilmeli.

## Kaynaklar
- KATE-CD: https://huggingface.co/datasets/CSCRS/kate-cd (ITÜ CSCRS)
{SRC_XBD}- Maxar Open Data Program: https://maxar-opendata.s3.amazonaws.com/events/catalog.json
"""

# ---------------- FLOOD ----------------
fl = J(O / "flood/results.json"); va = J(O / "flood_valencia/results.json")
fr = fl["results"]
meths = ["NDWI", "MNDWI", "S1 VV threshold", "U-Net S1", "U-Net S2", "U-Net S1+S2"]
bestv = max(va["results"], key=lambda k: va["results"][k]["f1"])
reports["02-sel"] = f"""# Sel — Sen1Floods11 (11 ülke), Valencia DANA 2024 ve xBD

## Özet
- **Uydu (Sentinel-2 optik + Sentinel-1 radar, 10 m), 11 ülke:** en iyi model **U-Net S1+S2 IoU {fr['U-Net S1+S2|test']['all']['iou']:.2f}** (F1 {fr['U-Net S1+S2|test']['all']['f1']:.2f}); eğitimsiz NDWI eşiği bile IoU {fr['NDWI|test']['all']['iou']:.2f}. Eğitimde hiç görülmeyen Bolivya olayında U-Net S1+S2 IoU {fr['U-Net S1+S2|bolivia']['all']['iou']:.2f}.
- **Yakın tarihli büyük sel — Valencia, 29 Ekim 2024:** Copernicus EMS resmi taşkın alanına (≈{va['gt_flood_km2']:.0f} km² inceleme alanında) karşı en iyi yöntem **{bestv}: F1 {va['results'][bestv]['f1']:.2f}**. İndeks eşikleri çok isabetli (precision ≈0,9+) ama taşkının yalnızca ~üçte birini buluyor.
- **Kentsel sel uydudan zor:** Valencia'da yerleşim alanlarında recall {va['results'][bestv]['recall_urban']:.2f}, tarım alanında {va['results'][bestv]['recall_cropland']:.2f}.
- **Radar tek başına:** 2 gün sonraki Sentinel-1 geçişinde su çekilmeye başladığı için recall {va['results']['S1 VV eşik (1 Kas)']['recall']:.2f}; zamanlama kritik.
- **Bina hasarı (xBD sel/kasırga olayları):** sel kaynaklı bina hasarı en zor türlerden — ayrıntı aşağıda.

## Veri ve yer gerçeği
| set | platform | çözünürlük | yer gerçeği |
|---|---|---|---|
| Sen1Floods11 (HF `blumenstiel/Sen1Floods11`) | uydu: Sentinel-1 GRD + Sentinel-2 L1C | 10 m | 446 elle etiketli 512×512 kare, 11 olay |
| Valencia DANA, Ekim 2024 | uydu: Sentinel-2 L2A (31 Eki), Sentinel-1 RTC (25 Eki, 1 Kas) — Planetary Computer | 10 m | Copernicus EMS EMSR773 AOI01 taşkın sınırı |
| xBD: midwest-flooding, nepal-flooding, 4 kasırga | uydu: Maxar | ~0,5 m | bina hasar seviyeleri |

## Sonuçlar — Sen1Floods11 test (90 kare) ve Bolivya (eğitimde görülmemiş, 15 kare)
Eşikler doğrulama bölümünde seçildi (NDWI>{fl['thresholds']['NDWI']:.2f}, MNDWI>{fl['thresholds']['MNDWI']:.2f}, VV<{fl['thresholds']['S1 VV threshold']:.0f} dB). "Sadece taşkın" = JRC kalıcı su pikselleri hariç.
{table(["yöntem", "test IoU", "test F1", "test sadece-taşkın IoU", "Bolivya IoU", "Bolivya F1"], [[m, fr[f'{m}|test']['all']['iou'], fr[f'{m}|test']['all']['f1'], fr[f'{m}|test']['flood_only']['iou'], fr[f'{m}|bolivia']['all']['iou'], fr[f'{m}|bolivia']['all']['f1']] for m in meths])}
{img(O / "flood/iou_by_method.png", "flood_iou_by_method.png", "Yöntem karşılaştırması")}
{img(O / "flood/iou_by_country.png", "flood_iou_by_country.png", "Ülke/olay bazında IoU")}
{img(O / "flood/samples.png", "flood_samples.png", "Rastgele test kareleri")}
{img(O / "flood/training_curves.png", "flood_training.png", "U-Net eğitim eğrileri")}

## Vaka — Valencia DANA (29 Ekim 2024)
Modeller **yeniden eğitilmeden** Sen1Floods11'den aktarıldı (S2 modeli L1C ile eğitildi, burada L2A'ya uygulandı; B10 bandı eğitim ortalamasıyla dolduruldu). Kalıcı su (ESA WorldCover) hariç.
{table(["yöntem", "precision", "recall", "F1", "IoU", "recall (yerleşim)", "recall (tarım)"], [[k, v['precision'], v['recall'], v['f1'], v['iou'], v['recall_urban'], v['recall_cropland']] for k, v in va['results'].items()])}
{img(O / "flood_valencia/metrics.png", "flood_valencia_metrics.png", "Valencia: yöntemler ve arazi türüne göre recall")}
{img(O / "flood_valencia/maps.png", "flood_valencia_maps.png", "Valencia: görüntüler, EMS yer gerçeği ve en iyi tahmin")}

## Sonuçlar — sel/kasırga kaynaklı bina hasarı (xBD)
{XBD_NOTE}
{xbd_rows(["flood", "hurricane"])}
{img(O / "xbd/samples_flood.png", "flood_xbd_samples.png", "xBD sel olayları — rastgele test kareleri")}

## Sınırlar
- EMS yer gerçeği 30–31 Ekim Landsat-8/Sentinel-2'den üretildi; aynı Sentinel-2 sahnesini kullanan yöntemler bir miktar avantajlı, Sentinel-1 (1 Kas) ise dezavantajlı (su çekilmiş).
- EMS "gözlenen olay" katmanı çamur/taşkın izlerini de içerir; açık su indeksleri bunları görmez → düşük recall'ın bir kısmı tanım farkı.
- Sen1Floods11 kareleri 10 m; ince kentsel sokak taşkınları bu çözünürlükte görünmez.

## Kaynaklar
- Sen1Floods11 (Bonafilia ve ark., 2020): https://github.com/cloudtostreet/Sen1Floods11
- Copernicus EMS EMSR773: https://rapidmapping.emergency.copernicus.eu/EMSR773
- Sentinel-1/2, ESA WorldCover: Microsoft Planetary Computer
{SRC_XBD}"""

# ---------------- LANDSLIDE ----------------
ls = J(O / "landslide/results.json"); lu = J(O / "landslide_uav/results.json")
lsr, lur = ls["results"], lu["results"]
bl = max(lsr, key=lambda k: lsr[k]["f1"]); bu = max(lur, key=lambda k: lur[k]["f1"])
reports["03-heyelan"] = f"""# Heyelan — Landslide4Sense (uydu) ve İHA heyelan seti (hava)

## Özet
- **Uydu (Sentinel-2 14 bant: 12 spektral + eğim + DEM, 10 m), Landslide4Sense test (800 kare):** en iyi **{bl}: F1 {lsr[bl]['f1']:.2f}**, IoU {lsr[bl]['iou']:.2f}. Sadece RGB (Google Earth benzeri bilgi) kullanan model F1 {lsr['U-Net RGB (Google Earth benzeri)']['f1']:.2f} → çok bantlı + topografya katkısı ölçülebilir.
- **Kural tabanlı NDVI + eğim** F1 {lsr['NDVI + eğim kuralı']['f1']:.2f}: tek başına yetersiz.
- **Hava (İHA, RGB, cm–dm çözünürlük), {lu['n']['test']} test karesi:** **{bu}: F1 {lur[bu]['f1']:.2f}**, IoU {lur[bu]['iou']:.2f}; RGB toprak indeksi kuralı F1 {lur['RGB toprak indeksi (kural)']['f1']:.2f}.
- Heyelan pikselleri azınlıkta (uyduda {ls['landslide_px_frac']['test']:.1%}, İHA'da {lu['landslide_px_frac']:.1%}) — precision/recall dengesi eşiğe çok duyarlı.

## Veri ve yer gerçeği
| set | platform | çözünürlük | yer gerçeği |
|---|---|---|---|
| Landslide4Sense (HF `ibm-nasa-geospatial/Landslide4sense`) | uydu: Sentinel-2 + ALOS PALSAR eğim/DEM | ~10 m, 128×128 | piksel maskesi; {ls['split_sizes']['train']}/{ls['split_sizes']['validation']}/{ls['split_sizes']['test']} train/val/test |
| İHA heyelan seti (HF `syeddhasnainn/landslide-uav-all`, alt küme) | hava: İHA RGB | cm–dm (384×384'e küçültüldü) | piksel maskesi; {lu['n']['train']}/{lu['n']['val']}/{lu['n']['test']} kullanıldı |

## Sonuçlar — uydu (Landslide4Sense)
Kural eşiği doğrulamada seçildi: NDVI < {ls['rule']['ndvi_lt']:.2f} ve eğim > {ls['rule']['slope_gt']:.2f} (normalize).
{table(["yöntem", "precision", "recall", "F1", "IoU"], [[k, v['precision'], v['recall'], v['f1'], v['iou']] for k, v in lsr.items()])}
{img(O / "landslide/metrics.png", "landslide_l4s_metrics.png", "Landslide4Sense: metrikler ve precision–recall eğrileri")}
{img(O / "landslide/samples.png", "landslide_l4s_samples.png", "Rastgele test kareleri (RGB, etiket, kural, U-Net RGB, U-Net çok bantlı)")}

## Sonuçlar — hava (İHA)
{table(["yöntem", "precision", "recall", "F1", "IoU"], [[k, v['precision'], v['recall'], v['f1'], v['iou']] for k, v in lur.items()])}
{img(O / "landslide_uav/metrics.png", "landslide_uav_metrics.png", "İHA heyelan: metrikler, kare başına IoU dağılımı, eğitim eğrisi")}
{img(O / "landslide_uav/samples.png", "landslide_uav_samples.png", "Rastgele İHA test kareleri")}

## Sınırlar
- İHA setinin yalnızca bir alt kümesi kullanıldı (19 eğitim parçasından 4'ü); görüntüler 384 px'e küçültüldü.
- Landslide4Sense test bölgeleri eğitimle benzer coğrafyalardan; yeni bir bölgeye (örn. 6 Şubat'ın tetiklediği heyelanlar) aktarım ayrıca test edilmeli.

## Kaynaklar
- Landslide4Sense (Ghorbanzadeh ve ark., 2022): https://github.com/iarai/Landslide4Sense-2022
- İHA heyelan seti: https://huggingface.co/datasets/syeddhasnainn/landslide-uav-all
"""

# ---------------- TORNADO ----------------
rf = J(O / "tornado_rollingfork/results.json")
tt = xe["per_type"].get("tornado", {})
reports["04-hortum"] = f"""# Hortum — xBD (Joplin, Moore, Tuscaloosa) ve Rolling Fork 2023 (hava fotoğrafı)

## Özet
- **Uydu (Maxar, xBD) — 3 hortum olayı:** bina bulma F1 {tt.get('loc_f1', float('nan')):.2f}, hasarlı/hasarsız bina F1 **{tt.get('building_damaged_f1', float('nan')):.2f}**, yıkılmış bina recall {tt.get('destroyed_recall', float('nan')):.2f}.
- **Hava fotoğrafı (NAIP uçak görüntüsü, 2021 öncesi / Ağustos 2023 sonrası) — Rolling Fork EF4 hortumu (24 Mart 2023):** xBD uydu modeli hiç eğitim almadan uygulandı; ayırt etme gücü zayıf ama yönü doğru (ortalama skor EF0'dan EF4'e artıyor): NWS saha ekiplerinin {rf['n_damage_points']} yapı hasar noktasına karşı **EF3+ ile EF0–1 ayrımı AUC {rf['auc_EF3plus_vs_EF0_1']:.2f}**, EF2+ ile hasarsız kontrol binaları AUC {rf['auc_EF2plus_vs_control']:.2f}; EF derecesi ile model skoru Spearman ρ={rf['spearman_ef_vs_score']:.2f}.

## Veri ve yer gerçeği
| set | platform | çözünürlük | yer gerçeği |
|---|---|---|---|
| xBD tier3: joplin, moore, tuscaloosa | uydu: Maxar | ~0,5 m (modelde ~1 m) | bina hasar seviyeleri |
| NAIP Mississippi 2021-11 ve 2023-08 (Planetary Computer) | hava: uçak (USDA NAIP) | 0,3 m / 0,6 m → 1 m'ye örneklendi | NOAA NWS Damage Assessment Toolkit: EF dereceli yapı noktaları |

## Sonuçlar — xBD hortum olayları
{XBD_NOTE}
{xbd_rows(["tornado"])}
{img(O / "xbd/samples_tornado.png", "tornado_xbd_samples.png", "xBD hortum olayları — rastgele test kareleri")}

## Vaka — Rolling Fork, Mississippi (EF4, 24 Mart 2023), hava fotoğrafı
Her NWS hasar noktası için 256 m × 256 m öncesi/sonrası NAIP kesiti alındı; skor = noktanın 20 m çevresinde modelin *ağır hasar + yıkılmış* olasılığı (bina pikselleri ağırlıklı). Kontroller: hasar noktalarına >1,5 km uzaklıkta, modelin bina bulduğu rastgele noktalar ({rf['n_controls']} adet).
{table(["seviye", "nokta sayısı", "ortalama skor"], [[k, rf['n_by_level'][k], rf['mean_score_by_level'][k]] for k in rf['mean_score_by_level']])}
{img(O / "tornado_rollingfork/metrics.png", "tornado_rf_metrics.png", "Rolling Fork: EF derecesine göre model skoru ve AUC")}
{img(O / "tornado_rollingfork/samples.png", "tornado_rf_samples.png", "Rastgele NAIP kesitleri (camgöbeği daire = 20 m değerlendirme çevresi)")}

## Sınırlar
- NAIP sonrası görüntü olaydan ~4,5 ay sonra: enkaz kaldırılmış, bazı yapılar onarılmış olabilir → hasar düşük görünür.
- NAIP (uçak, nadir) ile Maxar (uydu, eğik bakış) arasında renk/ölçek farkı; model hava fotoğrafı ile eğitilmedi.
- NWS noktaları yalnızca hasar görmüş yapılardır; "hasarsız" sınıfı kontrol noktalarıyla yaklaşık temsil edildi.
- **Gözlenen başarısızlık modu:** EF4 noktalarında enkaz Ağustos 2023'e kadar kaldırılmış; model boş arsayı "bina yok" olarak okuyor ve hasar skoru düşük kalıyor (örnek görsellerin ilk ve üçüncü satırı). xBD'de yıkılmış bina hep enkazla görüldüğü için model "kaybolan binayı" hasar olarak öğrenmemiş. Öncesi görüntüdeki bina konumuna göre puanlama bu hatayı azaltır.

## Kaynaklar
{SRC_XBD}- NAIP: Microsoft Planetary Computer `naip`
- NWS Damage Assessment Toolkit: https://apps.dat.noaa.gov/stormdamage/damageviewer/
"""

# ---------------- HAIL ----------------
hc = {c: J(O / f"hail/{c}/results.json") for c in ["A_single_event_14jun", "C_early_6to14jun", "B_season_6to19jun"]}
cname = {"A_single_event_14jun": "A: tek olay (14 Haz), öncesi 11–14 / sonrası 16–19 Haz",
         "C_early_6to14jun": "C: 6–14 Haz olayları, öncesi 1–5 / sonrası 11–14 Haz",
         "B_season_6to19jun": "B: 6–19 Haz olayları, öncesi 1–5 / sonrası 16–19 Haz"}
reports["05-dolu"] = f"""# Dolu — Nebraska/Iowa, Haziran 2022 (Sentinel-2 vs radar MESH)

## Özet
- Dolu doğrudan "görülmez"; uydu yalnızca **bitki örtüsündeki hasarı** (NDVI düşüşü) görebilir. Yer gerçeği olarak NOAA MRMS **MESH** (radardan tahmini en büyük dolu çapı, ~1 km) kullanıldı; konumu SPC dolu raporlarıyla doğrulandı (raporlardaki MESH medyanı ~36 mm).
- **Küçük/orta dolu (≥25 mm) neredeyse ayırt edilemiyor:** AUC {min(v['auc'] for v in hc.values()):.2f}–{max(v['auc'] for v in hc.values()):.2f}.
- **Büyük dolu görülebiliyor:** ≥60 mm için AUC **{max(v['auc_by_hail_size'].get('>=60mm', 0) for v in hc.values()):.2f}**'e kadar; dolu çapı arttıkça ΔNDVI düzenli artıyor.
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
{table(["konfigürasyon", "geçerli hücre", "dolu hücresi", "AUC ≥25 mm", "AUC ≥40 mm", "AUC ≥60 mm", "Spearman (MESH, ΔNDVI)", "en iyi F1*"],
       [[cname[c], v['valid_cells'], v['hail_cells'], v['auc'], v['auc_by_hail_size'].get('>=40mm', float('nan')), v['auc_by_hail_size'].get('>=60mm', float('nan')), v['spearman_mesh_vs_dndvi'], v['best_f1']] for c, v in hc.items()])}
\\* F1 eşiği aynı veride seçildi (iyimser); asıl karşılaştırma ölçütü eşikten bağımsız AUC.

{img(O / "hail/C_early_6to14jun/metrics.png", "hail_C_metrics.png", "Konfigürasyon C: ROC, MESH sınıfına göre ΔNDVI, eşik eğrileri")}
{img(O / "hail/C_early_6to14jun/maps.png", "hail_C_maps.png", "Konfigürasyon C: NDVI öncesi/sonrası, MESH ve ΔNDVI haritaları")}
{img(O / "hail/A_single_event_14jun/maps.png", "hail_A_maps.png", "Konfigürasyon A: 14 Haziran olayı")}

## Sınırlar
- MESH bir radar tahminidir (dolu çapını sıklıkla abartır, yere düşen doluyu doğrudan ölçmez) — "gerçek" değil, en iyi mevcut alan ölçümü.
- ~250 m hücreler tarla sınırlarını karıştırır; tarla bazlı analiz ve daha uzun süreli zaman serisi sinyali güçlendirir.
- Haziran başında mısır/soya küçük olduğu için hasar NDVI'da sınırlı görünür; Temmuz–Ağustos olayları daha belirgin olabilir.

## Kaynaklar
- NOAA MRMS (IEM arşivi): https://mtarchive.geol.iastate.edu/
- NOAA SPC Storm Reports: https://www.spc.noaa.gov/climo/reports/
- CIMSS uydu blogu, Nebraska/Iowa dolu izleri (Haziran 2022): https://cimss.ssec.wisc.edu/satellite-blog/archives/46975
"""

# ---------------- HEAT ----------------
he = J(O / "heat/results.json"); pc = he["per_city"]
reports["06-asiri-sicak"] = f"""# Aşırı sıcak — Yaz 2023 sıcak hava dalgaları (MODIS yüzey sıcaklığı vs istasyon)

## Özet
- **10 şehir** (Phoenix, Las Vegas, El Paso, Sevilla, Roma, Palermo, Atina, Antalya, Adana, Pekin), Haziran–Ağustos 2023; kalibrasyon yalnızca 2022 yazıyla.
- Uydu gündüz yüzey sıcaklığı (LST) ile istasyon günlük maksimum hava sıcaklığı ortalama **r = {he['mean_r']:.2f}**; doğrusal kalibrasyon hatası ortalama **{he['mean_rmse']:.1f} °C** RMSE.
- **Aşırı sıcak günü tespiti** (istasyon Tmax ≥ 2013–2022 yaz P90): sıralama gücü iyi (**havuz AUC {he['pooled_auc_zscore']:.2f}**), fakat eşikleme yöntemi belirleyici: doğrusal kalibrasyon F1 {he['pooled_f1']:.2f} (tahminler ortalamaya çekiliyor), yüzdelik eşleme F1 **{he['pooled_q_f1']:.2f}** (precision {he['pooled_q_precision']:.2f}, recall {he['pooled_q_recall']:.2f}).
- Akdeniz LST anomali haritası Temmuz 2023 "Cerberus" dalgasını Sicilya, Sardunya, Yunanistan, Kuzey Afrika ve Adana çevresinde net gösteriyor (kara ortalaması {he.get('anomaly_mean_land_c', float('nan')):+.1f} °C).

## Veri ve yer gerçeği
| veri | platform | çözünürlük |
|---|---|---|
| MODIS Terra MOD11A1 günlük LST (gündüz) — Planetary Computer | uydu | 1 km |
| MODIS Terra MOD11A2 8 günlük LST (anomali haritası) | uydu | 1 km → 0,05° |
| Meteostat (NOAA ISD/DWD istasyonları) günlük Tmax, 2013–2023 | yer istasyonu | nokta |

## Sonuçlar — şehir bazında (test yılı 2023)
{table(["şehir", "istasyon", "gün (bulutsuz)", "aşırı sıcak gün", "r", "RMSE °C", "AUC", "F1 doğrusal", "F1 yüzdelik"],
       [[c, v['station'], v['n_days'], v['heat_days'], v['pearson_r'], v['rmse_c'], v['auc'] if v['auc'] is not None else 'n/a', v['f1'], v['q_f1']] for c, v in pc.items()])}
{img(O / "heat/metrics.png", "heat_metrics.png", "LST–Tmax ilişkisi, aşırı sıcak günü ROC ve şehir bazında performans")}
{img(O / "heat/timeseries.png", "heat_timeseries.png", "Yaz 2023 zaman serileri: istasyon Tmax ve uydu tahmini")}
{img(O / "heat/anomaly_map.png", "heat_anomaly_map.png", "12–19 Temmuz 2023 LST anomalisi (2018–2022 ortalamasına göre)")}

## Sınırlar
- LST yüzey sıcaklığıdır, hava sıcaklığı değil; kentsel yüzeyler/çıplak toprak gündüz 10–20 °C daha sıcak olabilir.
- Bulutlu günlerde ölçüm yok (gün sayısı sütunu); Pekin'de muson nedeniyle yaz günlerinin ~yarısı kayıp.
- 2023, 2022'den belirgin sıcaktı: 2022 oranıyla kalibre edilen eşikler 2023'teki aşırı gün sayısını eksik tahmin ediyor (recall düşük).

## Kaynaklar
- MODIS LST (Wan ve ark.), Microsoft Planetary Computer `modis-11A1-061`, `modis-11A2-061`
- Meteostat: https://meteostat.net
"""

# ---------------- FIRE ----------------
fi = J(O / "fire/results.json")
fx = xe["per_type"].get("fire", {})
k010 = "dNBR>0.10 (USGS düşük şiddet)"
reports["07-yangin"] = f"""# Yangın — yanık alanı (Sentinel-2 dNBR) ve bina hasarı (xBD)

## Özet
- **Yanık alanı, 4 yangın, resmi sınırlara karşı:** {', '.join(f"{n.split(' (')[0]} IoU {max(v['iou'] for v in r['scores'].values()):.2f}" for n, r in fi.items())} (en iyi eşik).
- Tek bir standart eşik (USGS dNBR > 0,10) eğitim olmadan çoğu yangında iyi çalışıyor; hatalar resmi sınır içindeki yanmamış adacıklar, hasat edilen tarlalar ve bulut kaynaklı.
- **Yangın kaynaklı bina hasarı (xBD, 5 yangın olayı):** hasarlı/hasarsız bina F1 **{fx.get('building_damaged_f1', float('nan')):.2f}**, yıkılmış bina recall {fx.get('destroyed_recall', float('nan')):.2f} — yangın, xBD'de en iyi tespit edilen afet türlerinden (yanmış binalar ikili: ya ayakta ya kül).

## Veri ve yer gerçeği
| veri | platform | çözünürlük | yer gerçeği |
|---|---|---|---|
| Sentinel-2 L2A B08/B12 öncesi–sonrası mozaik — Planetary Computer | uydu | 20 m (~25 m grid) | NIFC WFIGS / InterAgency yangın sınırları (ABD), EFFIS yanık alanı (Manavgat) |
| xBD: socal, santa-rosa, woolsey, portugal, pinery | uydu: Maxar | ~0,5 m (modelde ~1 m) | bina hasar seviyeleri |

## Sonuçlar — yanık alanı
{table(["yangın", "yöntem", "precision", "recall", "F1", "IoU", "resmi alan km²", "uydu alanı km²**"],
       [[n, k, v['precision'], v['recall'], v['f1'], v['iou'], v['area_official_km2'], v['area_pred_km2']] for n, r in fi.items() for k, v in r['scores'].items()])}
\\*\\* Uydu alanı, sınırın %25 tamponlu kutusunun tamamında ölçüldü (kutudaki başka yanıklar da sayılır).

{img(O / "fire/metrics.png", "fire_metrics.png", "IoU, alan karşılaştırması, sınır içi yanma şiddeti dağılımı")}
{img(O / "fire/maps.png", "fire_maps.png", "NBR öncesi/sonrası, dNBR ve hata haritası")}

## Sonuçlar — bina hasarı (xBD yangın olayları)
{XBD_NOTE}
{xbd_rows(["fire"])}
{img(O / "xbd/samples_fire.png", "fire_xbd_samples.png", "xBD yangın olayları — rastgele test kareleri")}

## Sınırlar
- Resmi sınırlar yanmamış adacıkları da kapsar; IoU üst sınırı 1'in altındadır.
- EFFIS sınırı MODIS/VIIRS + Sentinel-2 tabanlı, NIFC sınırları hava/saha ölçümlü — kaynaklar arası tutarlılık farklı.
- Camp yangınında "sonrası" Aralık mozaiğinde bulut/kar ve hasat edilmiş tarlalar yanlış alarm üretiyor.

## Kaynaklar
- NIFC WFIGS Interagency Perimeters: https://data-nifc.opendata.arcgis.com
- EFFIS: https://effis.jrc.ec.europa.eu
{SRC_XBD}"""

for name, text in reports.items():
    (R / f"{name}.md").write_text(text)

# cross-type overview chart + index
heads = [
    ("Deprem\n(6 Şubat, KATE-CD)", J(F / "kate_only.json")["test"]["f1"], "F1 (piksel)"),
    ("Sel\n(Sen1Floods11)", fr["U-Net S1+S2|test"]["all"]["f1"], "F1 (piksel)"),
    ("Sel\n(Valencia 2024)", va["results"][bestv]["f1"], "F1 (piksel)"),
    ("Heyelan\n(uydu)", lsr[bl]["f1"], "F1 (piksel)"),
    ("Heyelan\n(İHA)", lur[bu]["f1"], "F1 (piksel)"),
    ("Hortum\n(xBD bina)", tt.get("building_damaged_f1", 0), "F1 (bina)"),
    ("Hortum\n(NAIP, EF3+ vs EF0–1)", rf["auc_EF3plus_vs_EF0_1"], "AUC"),
    ("Dolu\n(≥60 mm)", max(v["auc_by_hail_size"].get(">=60mm", 0) for v in hc.values()), "AUC"),
    ("Dolu\n(≥25 mm)", max(v["auc"] for v in hc.values()), "AUC"),
    ("Aşırı sıcak\n(gün tespiti)", he["pooled_auc_zscore"], "AUC"),
    ("Yangın\n(yanık alanı)", float(np.mean([max(v['f1'] for v in r['scores'].values()) for r in fi.values()])), "F1 (piksel)"),
    ("Yangın\n(xBD bina)", fx.get("building_damaged_f1", 0), "F1 (bina)"),
]
fig, ax = plt.subplots(figsize=(15, 5))
cols = {"F1 (piksel)": "#3b6ea5", "F1 (bina)": "#6a994e", "AUC": "#e07b39"}
for i, (n, v, m) in enumerate(heads):
    ax.bar(i, v, color=cols[m]); ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(range(len(heads)), [h[0] for h in heads], fontsize=8); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=l) for l, c in cols.items()], loc="upper right")
ax.set_title("Afet türlerine göre en iyi yöntemin ana metriği (farklı metrikler doğrudan karşılaştırılamaz)")
plt.tight_layout(); plt.savefig(IMG / "overview.png", dpi=100); plt.close()

index = """# Afet türlerine göre uydu ve hava görüntüsü analizi — rapor dizini

Her rapor: veri ve yer gerçeği, yöntemler, metrik tabloları, grafikler, rastgele vaka görselleri, sınırlar ve kaynaklar.
Kod ve ham çıktılar: `disaster-eval/` (her deney bir betik, sonuçlar `disaster-eval/outputs/`).

![Genel bakış](img/overview.png)

| afet | rapor | platformlar | ana sonuç |
|---|---|---|---|
"""
rows = [("Deprem", "01-deprem.md", "uydu VHR (Maxar/Pleiades — Google Earth sınıfı)", f"6 Şubat F1 {J(F / 'kate_only.json')['test']['f1']:.2f}; xBD'den aktarım F1 {best_kz['f1']:.2f}"),
        ("Sel", "02-sel.md", "uydu (Sentinel-1/2), uydu VHR (xBD)", f"11 ülke IoU {fr['U-Net S1+S2|test']['all']['iou']:.2f}; Valencia 2024 F1 {va['results'][bestv]['f1']:.2f}"),
        ("Heyelan", "03-heyelan.md", "uydu (Sentinel-2+DEM), hava (İHA)", f"uydu F1 {lsr[bl]['f1']:.2f}; İHA F1 {lur[bu]['f1']:.2f}"),
        ("Hortum", "04-hortum.md", "uydu VHR (xBD), hava (NAIP uçak)", f"xBD bina F1 {tt.get('building_damaged_f1', 0):.2f}; Rolling Fork EF3+ AUC {rf['auc_EF3plus_vs_EF0_1']:.2f}"),
        ("Dolu", "05-dolu.md", "uydu (Sentinel-2) + radar MESH", f"≥60 mm AUC {max(v['auc_by_hail_size'].get('>=60mm', 0) for v in hc.values()):.2f}; ≥25 mm AUC {max(v['auc'] for v in hc.values()):.2f}"),
        ("Aşırı sıcak", "06-asiri-sicak.md", "uydu (MODIS LST) + istasyon", f"r {he['mean_r']:.2f}; gün tespiti AUC {he['pooled_auc_zscore']:.2f}"),
        ("Yangın", "07-yangin.md", "uydu (Sentinel-2), uydu VHR (xBD)", f"yanık alanı IoU {min(max(v['iou'] for v in r['scores'].values()) for r in fi.values()):.2f}–{max(max(v['iou'] for v in r['scores'].values()) for r in fi.values()):.2f}; xBD bina F1 {fx.get('building_damaged_f1', 0):.2f}")]
for a, f, p, s in rows:
    index += f"| {a} | [{f}]({f}) | {p} | {s} |\n"
index += """
**Google Earth hakkında:** Google Earth görüntüleri kullanım koşulları gereği toplu/otomatik indirilemez. Google Earth'teki afet sonrası çok yüksek çözünürlüklü görüntülerin önemli kısmı Maxar kaynaklıdır; bu çalışmada aynı sınıftaki Maxar Open Data (xBD, KATE-CD, Kahramanmaraş 2023 olayı) kullanıldı. Hava görüntüsü olarak İHA (heyelan) ve uçak (USDA NAIP, hortum) verisi test edildi; helikopter görüntüsü için açık, etiketli bir set bulunamadı.
"""
(R / "README.md").write_text(index)
print("written", list(reports))
