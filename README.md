# Deprem GPT: a transformer point process trained on several earthquake catalogs

Code, pre-registrations, result summaries and figures for the study
*A transformer trained on several earthquake catalogs outperforms ETAS after large earthquakes*
(Yıldırım, in preparation for *Geophysical Research Letters*).

A small decoder-only transformer treats each earthquake as a token and forecasts the time and magnitude of the
next event. It is trained on the catalog of a target region together with other regional catalogs and is
compared with regionally fitted ETAS models in prospective tests on the 2023 Kahramanmaraş, 2024 Noto and
2024 Hualien sequences. Every test had a written pre-registration with an explicit decision rule.

## Repository layout

| Folder | Content |
|---|---|
| `kod/` | All data preparation, training, ETAS, simulation and analysis code (Python) |
| `on_kayit/` | Pre-registrations `ON_KAYIT*.md` (Turkish), one per step |
| `sonuclar/` | Result summaries (`*_ozet.json`) from which all reported numbers are taken, including the pre-registered decisions |
| `sekiller/` | Output folder for the figures written by the analysis scripts (empty) |
| `makale_sekilleri/` | Main figures of the manuscript |

Raw catalogs, per-event result arrays and model weights are not included (about 1 GB, and the catalogs are
subject to the terms of their providers). The scripts below download or rebuild them. Job scripts for the
GPU server and run logs are also omitted.

## Steps

| Step | Question | Pre-registration | Main scripts |
|---|---|---|---|
| 1 | Context length 20, 512, 4096 (California) | `ON_KAYIT.md` | `hazirla.py`, `egit.py`, `analiz.py` |
| 2 | Multi-catalog training (California) | `ON_KAYIT_2.md` | `hazirla_cok.py`, `analiz2.py`, `etas_kiyas_ca.py` |
| 3 | Kahramanmaraş 2023 | `ON_KAYIT_3.md` | `indir_koeri.py`, `hazirla_tr.py`, `etas_zaman.py`, `analiz3.py` |
| 4 | Stronger ETAS variants | `ON_KAYIT_4.md` | `etas_guclu.py`, `analiz4.py`, `etas_json_onar.py` |
| 5 | Decomposition of the first-day gain (exploratory) | none | `etas_ayristir.py`, `model_ayristir.py`, `analiz5.py` |
| 6 | ETAS with short-term incompleteness | `ON_KAYIT_6.md` | `etas_eksik.py`, `analiz6.py` |
| 7 | Noto 2024 | `ON_KAYIT_7.md` | `hazirla_jp.py`, `analiz7.py` |
| 8 | Architecture variants | `ON_KAYIT_8.md` | `egit2.py`, `hazirla_konum.py`, `sentetik_etas.py`, `analiz8.py` |
| 9 | Probabilistic forecasts | `ON_KAYIT_9.md` | `olasilik_model.py`, `olasilik_etas.py`, `analiz9.py` |
| 10 | Location output | `ON_KAYIT_10.md` | `egit3.py`, `etas_uzay.py`, `harita_*.py`, `analiz10.py` |
| 11 | Hualien 2024 | `ON_KAYIT_11.md` | `indir_isc_tayvan.py`, `hazirla_tw.py`, `analiz11.py` |

`makale_sekiller.py` produces the manuscript figures and the numbers used in the text and in Tables S4 and S5 of the
Supporting Information (written to `makale/sayilar.json`, a copy of which is `sonuclar/makale_sayilar.json`).
The plotted values of Figures 1 and 2a are in `sonuclar/sekil_verisi.npz`. Figure 2b uses `sonuclar/c5_ozet.json`
and Figure 3 uses `sonuclar/c9_ozet.json`.

## Data

| Catalog | Source | Script |
|---|---|---|
| California (ComCat, M ≥ 2.5) | [EarthquakeNPP](https://github.com/ss15859/EarthquakeNPP), `Experiments/ETAS/output_data_ComCat_25/augmented_catalog.csv`, placed under `veri/earthquakenpp/` | `hazirla.py` |
| Japan (JMA unified hypocenter catalog, 2000–2024) | Yearly files `h2000.zip` … `h2024.zip` from the Japan Meteorological Agency, placed under `veri/jma/` | `hazirla_cok.py`, `hazirla_jp.py` |
| Italy (INGV) and New Zealand (GeoNet) | FDSN event services (text format), placed under `veri/ingv/` and `veri/geonet/` | `hazirla_cok.py` |
| Türkiye (KOERI, 2006–2024) | KOERI search form, downloaded month by month | `indir_koeri.py`, `hazirla_tr.py` |
| Taiwan (CWA local magnitudes via ISC, 2005–2024) | ISC event catalogue search, downloaded month by month | `indir_isc_tayvan.py`, `hazirla_tw.py` |

Please cite the data providers and respect their terms of use. Use of ISC data requires citation of the ISC
Bulletin.

## Running

Install the Python dependencies (Python 3.11+):

```bash
pip install -r requirements.txt
```

Data preparation and analysis run on a CPU (`python3 kod/<script>.py` from the repository root). Training and
the ETAS fits use PyTorch on a CUDA GPU. The work was done on NVIDIA H100 GPUs (PyTorch 2.6, Python 3.11).
Dense ETAS likelihoods for the largest catalogs need a large-memory GPU. Example calls:

```bash
python kod/egit.py --ana turkiye.npz --ek ek_tr.npz --L 512 --tohum 1 --ad c3_T1_t1
python kod/etas_guclu.py --ana turkiye.npz --n_max 0.99 --ad etas4_E2
python kod/analiz3.py
```

GPU training is not bitwise reproducible. Repeated runs with the same seed differ by about 0.003 nats per event
in temporal log-likelihood.

## License

Code is released under the MIT License (see `LICENSE`). Pre-registrations, result summaries and figures are released
under CC BY 4.0.
