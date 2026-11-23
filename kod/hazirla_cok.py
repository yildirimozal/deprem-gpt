"""Japonya (JMA), İtalya (INGV), Yeni Zelanda (GeoNet) kataloglarını ek eğitim verisi olarak hazırlar.

Çıktı: veri/ek_kataloglar.npz  (t_gun, m_goreli, kat, seg_bas, seg_son, seg_ad)
       sonuclar/ek_veri_ozeti.json
Kurallar ON_KAYIT_2.md'de. Büyüklük katalog eşiğine göre göreli (M − Mc), 0.1'e yuvarlı.
"""
import glob
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
V = KOK / "veri"
DERIN_MAX = 70.0
KATI = "--kati" in sys.argv          # yalnız 2007 öncesi ek veri


def yuvarla(x):
    return np.floor(np.asarray(x, float) * 10 + 0.5 + 1e-9) / 10


def jma():
    satirlar = []
    for z in sorted(glob.glob(str(V / "jma/h*.zip"))):
        with zipfile.ZipFile(z) as zf:
            for ad in zf.namelist():
                metin = zf.read(ad).decode("ascii", errors="replace")
                satirlar.extend(metin.splitlines())
    kayit = []
    for s in satirlar:
        if len(s) < 96 or s[0] != "J":
            continue
        try:
            yil, ay, gun = int(s[1:5]), int(s[5:7]), int(s[7:9])
            sa, dk = int(s[9:11]), int(s[11:13])
            sn = float(s[13:17]) / 100
            lat = int(s[21:24]) + float(s[24:28]) / 6000
            lon = int(s[32:36]) + float(s[36:40]) / 6000
            dz = s[44:49]
            derin = float(dz) / 100 if dz[3:].strip() else float(dz[:3])   # F5.2; sabit derinlikte F3.0
            mz = s[52:54]
            if not mz.strip() or not mz.strip().lstrip("-").isdigit():
                continue                       # boş ya da harfli (negatif) büyüklük
            m = int(mz) / 10
        except ValueError:
            continue
        bayrak = s[95]
        if bayrak in "FN":
            continue
        kayit.append((yil, ay, gun, sa, dk, sn, lat, lon, derin, m))
    d = pd.DataFrame(kayit, columns="yil ay gun sa dk sn lat lon derin m".split())
    zaman = pd.to_datetime(dict(year=d.yil, month=d.ay, day=d.gun, hour=d.sa, minute=d.dk)) \
        + pd.to_timedelta(d.sn, unit="s") - pd.Timedelta(hours=9)          # JST → UTC
    d = pd.DataFrame(dict(zaman=zaman, lat=d.lat, lon=d.lon, derin=d.derin, m=d.m))
    return d[(d.lat.between(30, 46)) & (d.lon.between(128, 148))]


def fdsn(desen, tur_sutun=True):
    parca = []
    for f in sorted(glob.glob(str(V / desen))):
        x = pd.read_csv(f, sep="|", comment=None, skipinitialspace=True)
        x.columns = [c.strip().lstrip("#").strip() for c in x.columns]
        parca.append(x)
    x = pd.concat(parca)
    x = x[x.EventType.astype(str).str.strip() == "earthquake"]
    d = pd.DataFrame(dict(zaman=pd.to_datetime(x.Time.str.strip(), format="ISO8601"),
                          lat=x.Latitude.astype(float), lon=x.Longitude.astype(float),
                          derin=x["Depth/km" if "Depth/km" in x else "Depth/Km"].astype(float),
                          m=x.Magnitude.astype(float)))
    return d.drop_duplicates()


def mc_maxc(m):
    """En büyük eğrilik yöntemi (0.1 bölme) + 0.2 düzeltme."""
    b = np.round(m * 10).astype(int)
    u, n = np.unique(b, return_counts=True)
    return u[np.argmax(n)] / 10 + 0.2


KATALOG = {
    "Japonya_JMA": dict(oku=jma, mc=3.0, bolge=None),
    "Italya_INGV": dict(oku=lambda: fdsn("ingv/ingv_*.txt"), mc=2.5, bolge=(35, 48, 6, 19)),
    "YeniZelanda_GeoNet": dict(oku=lambda: fdsn("geonet/geonet_*.txt"), mc=3.0, bolge=(-48, -34, 165, 179)),
}

t0 = pd.Timestamp("1971-01-01")
tum_t, tum_m, tum_k, seg_bas, seg_son, seg_ad = [], [], [], [], [], []
ozet = {"kati": KATI, "derin_max_km": DERIN_MAX, "katalog": {}}
n = 0
for k, (ad, c) in enumerate(KATALOG.items()):
    d = c["oku"]()
    ham = len(d)
    if c["bolge"]:
        a, b, e, f = c["bolge"]
        d = d[d.lat.between(a, b) & d.lon.between(e, f)]
    d = d[d.derin <= DERIN_MAX].copy()
    d["m"] = yuvarla(d.m)
    if KATI:
        d = d[d.zaman < pd.Timestamp("2007-01-01")]
    # Mc denetimi: her yıl için MAXC+0.2; ön kayıtlı eşikten büyükse eşik yükseltilir
    yil_mc = d[d.m >= 1.0].groupby(d.zaman.dt.year).m.apply(lambda x: mc_maxc(x.values))
    mc = max(c["mc"], float(yil_mc.max())) if len(yil_mc) else c["mc"]
    d = d[d.m >= mc - 1e-9].sort_values("zaman")
    if len(d) == 0:
        ozet["katalog"][ad] = {"olay": 0}
        continue
    t = (d.zaman - t0).dt.total_seconds().values / 86400.0
    tum_t.append(t)
    tum_m.append(np.round(d.m.values - mc, 1))
    tum_k.append(np.full(len(t), k, np.int8))
    seg_bas.append(n)
    n += len(t)
    seg_son.append(n)
    seg_ad.append(ad)
    dt = np.diff(t)
    ozet["katalog"][ad] = {
        "ham_satir": int(ham), "olay": int(len(t)), "mc_on_kayit": c["mc"], "mc_kullanilan": mc,
        "yillik_mc_maxc": {int(y): float(v) for y, v in yil_mc.items()},
        "ilk": str(d.zaman.iloc[0]), "son": str(d.zaman.iloc[-1]),
        "m_max_mutlak": float(d.m.max()), "sifir_dt": int((dt <= 0).sum()),
        "M6_ustu": int((d.m >= 6.0).sum()),
    }

np.savez(V / ("ek_kataloglar_kati.npz" if KATI else "ek_kataloglar.npz"),
         t_gun=np.concatenate(tum_t), m_goreli=np.concatenate(tum_m), kat=np.concatenate(tum_k),
         seg_bas=np.array(seg_bas), seg_son=np.array(seg_son), seg_ad=np.array(seg_ad))
ozet["toplam_olay"] = int(n)
with open(KOK / ("sonuclar/ek_veri_ozeti_kati.json" if KATI else "sonuclar/ek_veri_ozeti.json"), "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)
print(json.dumps({a: {k: v for k, v in b.items() if k != "yillik_mc_maxc"} for a, b in ozet["katalog"].items()},
                 indent=1, ensure_ascii=False))
for a, b in ozet["katalog"].items():
    print(a, b.get("yillik_mc_maxc"))
print("toplam", n)
