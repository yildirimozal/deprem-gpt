"""Noto testi verisi (ON_KAYIT_7.md).

1) veri/japonya.npz: JMA, ikinci testteki hazırlıkla aynı (J kayıtları, F/N hariç, 30–46°K, 128–148°D,
   derinlik <= 70 km, büyüklük 0.1'e yuvarlı), M >= 3.0.
   bolum: 0 geçmiş (2000–2001), 1 eğitim (2002–2020), 2 doğrulama (2021 – 2024-01-01),
          3 test (2024-01-01 – 2025-01-01).
   Maskeler: Noto (2024-01-01, M7.6) ve Hyuganada (2024-08-08, M7.1) sonrası pencereler, Noto kutusu.
2) veri/ek_jp.npz: California (EarthquakeNPP, hepsi), INGV, GeoNet (veri/ek_kataloglar.npz) ve Türkiye
   (veri/turkiye.npz, M >= 3.0), hepsi 2024-01-01 öncesi. JMA burada yok (ana katalog).
Mac'te: /usr/bin/python3 kod/hazirla_jp.py
"""
import glob
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
V = KOK / "veri"
T0 = pd.Timestamp("1971-01-01")
MC = 3.0
DERIN_MAX = 70.0
SINIR = {"egitim": "2002-01-01", "dogrulama": "2021-01-01", "test": "2024-01-01", "son": "2025-01-01"}
NOTO_KUTU = dict(enlem=(36.5, 38.0), boylam=(136.0, 138.0))


def gun(ts):
    return (pd.Timestamp(ts) - T0).total_seconds() / 86400.0


def jma():
    """hazirla_cok.py'deki ayrıştırıcının aynısı."""
    satirlar = []
    for z in sorted(glob.glob(str(V / "jma/h*.zip"))):
        with zipfile.ZipFile(z) as zf:
            for ad in zf.namelist():
                satirlar.extend(zf.read(ad).decode("ascii", errors="replace").splitlines())
    kayit = []
    for s in satirlar:
        if len(s) < 96 or s[0] != "J":
            continue
        try:
            yil, ay, g = int(s[1:5]), int(s[5:7]), int(s[7:9])
            sa, dk = int(s[9:11]), int(s[11:13])
            sn = float(s[13:17]) / 100
            lat = int(s[21:24]) + float(s[24:28]) / 6000
            lon = int(s[32:36]) + float(s[36:40]) / 6000
            dz = s[44:49]
            derin = float(dz) / 100 if dz[3:].strip() else float(dz[:3])
            mz = s[52:54]
            if not mz.strip() or not mz.strip().lstrip("-").isdigit():
                continue
            m = int(mz) / 10
        except ValueError:
            continue
        if s[95] in "FN":
            continue
        kayit.append((yil, ay, g, sa, dk, sn, lat, lon, derin, m))
    d = pd.DataFrame(kayit, columns="yil ay gun sa dk sn lat lon derin m".split())
    zaman = pd.to_datetime(dict(year=d.yil, month=d.ay, day=d.gun, hour=d.sa, minute=d.dk)) \
        + pd.to_timedelta(d.sn, unit="s") - pd.Timedelta(hours=9)          # JST → UTC
    d = pd.DataFrame(dict(zaman=zaman, lat=d.lat, lon=d.lon, derin=d.derin, m=d.m))
    return d[(d.lat.between(30, 46)) & (d.lon.between(128, 148))]


d = jma()
d = d[d.derin <= DERIN_MAX].copy()
d["m"] = np.floor(d.m.values * 10 + 0.5 + 1e-9) / 10
d = d[(d.m >= MC - 1e-9) & (d.zaman < SINIR["son"])].sort_values("zaman").reset_index(drop=True)
t = (d.zaman - T0).dt.total_seconds().values / 86400.0
m = d.m.values
bolum = np.zeros(len(d), np.int8)
bolum[t >= gun(SINIR["egitim"])] = 1
bolum[t >= gun(SINIR["dogrulama"])] = 2
bolum[t >= gun(SINIR["test"])] = 3
noto = ((d.zaman >= "2024-01-01 07:00") & (d.zaman < "2024-01-01 07:20") & (m >= 7.0)).values
hyuga = ((d.zaman >= "2024-08-08 07:30") & (d.zaman < "2024-08-08 08:00") & (m >= 6.8)).values
assert noto.sum() == 1 and hyuga.sum() == 1, (noto.sum(), hyuga.sum())
ana, hy = int(np.where(noto)[0][0]), int(np.where(hyuga)[0][0])
kutu = (d.lat.between(*NOTO_KUTU["enlem"]) & d.lon.between(*NOTO_KUTU["boylam"])).values
s_ana, s_hy = t - t[ana], t - t[hy]
maske = {"dizi_kutusu": kutu, "sonra_1g": (s_ana > 0) & (s_ana <= 1), "sonra_7g": (s_ana > 0) & (s_ana <= 7),
         "sonra_30g": (s_ana > 0) & (s_ana <= 30)}
maske["dizi_30g"] = maske["sonra_30g"] & kutu
maske["dizi_disi"] = (bolum == 3) & ~kutu
maske["hyuga_1g"] = (s_hy > 0) & (s_hy <= 1)
maske["hyuga_7g"] = (s_hy > 0) & (s_hy <= 7)
np.savez(V / "japonya.npz", t_gun=t, buyukluk=m, bolum=bolum, mc=MC, ana_sok_idx=ana, hyuga_idx=hy,
         enlem=d.lat.values, boylam=d.lon.values, **maske)

yil_maxc = {}
for y, g_ in d[d.zaman.dt.year.between(2002, 2023)].groupby(d.zaman.dt.year):
    u, n = np.unique(np.round(g_.m.values * 10).astype(int), return_counts=True)
    yil_maxc[int(y)] = float(u[np.argmax(n)] / 10 + 0.2)
ozet = {"mc": MC, "olay": int(len(t)), "bolum_sayisi": {int(k): int((bolum == k).sum()) for k in range(4)},
        "yillik_maxc_arti_0.2_M3_ustu_veriyle": yil_maxc,
        "noto": {"idx": ana, "zaman": str(d.zaman[ana]), "m": float(m[ana]), "enlem": float(d.lat[ana]), "boylam": float(d.lon[ana])},
        "hyuganada": {"idx": hy, "zaman": str(d.zaman[hy]), "m": float(m[hy])},
        "egitim_en_buyuk": float(m[bolum == 1].max()), "egitim_M7_ustu": int((m[bolum == 1] >= 7.0).sum()),
        "test_M6_ustu": [[str(z), float(x)] for z, x in zip(d.zaman[(bolum == 3) & (m >= 6.0)], m[(bolum == 3) & (m >= 6.0)])],
        "maske_sayisi": {k: int((x & (bolum == 3)).sum()) for k, x in maske.items()}, "sinirlar": SINIR, "noto_kutusu": NOTO_KUTU}

# ---------- ek katalog: California + INGV + GeoNet + Türkiye, 2024-01-01 öncesi ----------
kes = gun(SINIR["test"])
vc = np.load(V / "olaylar.npz")
e = np.load(V / "ek_kataloglar.npz")
vt = np.load(V / "turkiye.npz")
tl, ml, sb, ss, sa = [vc["t_gun"]], [np.round(vc["buyukluk"] - 2.5, 1)], [0], [len(vc["t_gun"])], ["California_ComCat"]
n = len(vc["t_gun"])
for a, b, ad in zip(e["seg_bas"], e["seg_son"], e["seg_ad"]):
    if str(ad) == "Japonya_JMA":
        continue
    k = e["t_gun"][a:b] < kes
    tl.append(e["t_gun"][a:b][k])
    ml.append(e["m_goreli"][a:b][k])
    sb.append(n)
    n += int(k.sum())
    ss.append(n)
    sa.append(str(ad))
k = vt["t_gun"] < kes
tl.append(vt["t_gun"][k])
ml.append(np.round(vt["buyukluk"][k] - float(vt["mc"]), 1))
sb.append(n)
n += int(k.sum())
ss.append(n)
sa.append("Turkiye_KOERI")
assert max(x.max() for x in tl) < kes
np.savez(V / "ek_jp.npz", t_gun=np.concatenate(tl), m_goreli=np.concatenate(ml),
         kat=np.concatenate([np.full(len(x), i, np.int8) for i, x in enumerate(tl)]),
         seg_bas=np.array(sb), seg_son=np.array(ss), seg_ad=np.array(sa))
ozet["ek_jp"] = {a: int(s2 - s1) for a, s1, s2 in zip(sa, sb, ss)}
ozet["ek_jp_toplam"] = int(n)
with open(KOK / "sonuclar/jp_veri_ozeti.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)
print(json.dumps(ozet, indent=1, ensure_ascii=False))
