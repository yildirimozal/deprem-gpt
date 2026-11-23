"""Kahramanmaraş testi verisi (ON_KAYIT_3.md).

1) veri/turkiye.npz: KOERI kataloğu, Türkiye kutusu, derinlik <= 70 km, büyüklük xM (0.1'e yuvarlı), M >= Mc.
   bolum: 0 geçmiş (2006-2007), 1 eğitim (2008-2020), 2 doğrulama (2021-01-01 .. 2023-02-06 00:00),
          3 test (2023-02-06 00:00 .. 2024-02-06 00:00). Testten sonrası atılır.
   Mc: 2008-2022 yıllarının en büyük eğrilik + 0.2 değerlerinin en büyüğü (test verisi kullanılmaz).
   Maskeler: dizi kutusu, ana şok sonrası 1/7/30 gün, ana şok öncesi test olayları.
2) veri/ek_tr.npz: California (EarthquakeNPP, tüm olaylar, M - 2.5) + JMA/INGV/GeoNet (veri/ek_kataloglar.npz)
   yalnızca 2023-02-06 öncesi. Hepsi Kahramanmaraş'tan önce.
Mac'te: /usr/bin/python3 kod/hazirla_tr.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
T0 = pd.Timestamp("1971-01-01")
KUTU = dict(enlem=(36.0, 42.2), boylam=(26.0, 45.0))
DIZI = dict(enlem=(35.8, 38.8), boylam=(35.3, 39.7))
DERIN_MAX = 70.0
ANA_SOK = pd.Timestamp("2023-02-06 01:17:32.67")
SINIR = {"egitim": "2008-01-01", "dogrulama": "2021-01-01", "test": "2023-02-06", "son": "2024-02-06"}


def gun(ts):
    return (pd.Timestamp(ts) - T0).total_seconds() / 86400.0


def yuvarla(x):
    return np.floor(np.asarray(x, float) * 10 + 0.5 + 1e-9) / 10


def mc_maxc(m):
    b = np.round(m * 10).astype(int)
    u, n = np.unique(b, return_counts=True)
    return u[np.argmax(n)] / 10 + 0.2


d = pd.read_csv(KOK / "veri/koeri/koeri_2006_2024.csv", dtype={"kod": str})
ham = len(d)
d["zaman"] = pd.to_datetime(d.tarih + " " + d.saat, format="%Y.%m.%d %H:%M:%S.%f")
d = d[d.enlem.between(*KUTU["enlem"]) & d.boylam.between(*KUTU["boylam"]) & (d.derinlik <= DERIN_MAX)].copy()
d["m"] = yuvarla(d.xM)
d = d[d.m > 0].sort_values("zaman")
kutu_n = len(d)

on = d[(d.zaman >= SINIR["egitim"]) & (d.zaman < SINIR["test"])]
yil_mc = on.groupby(on.zaman.dt.year).m.apply(lambda x: mc_maxc(x.values))
mc = float(yil_mc.max())
d = d[(d.m >= mc - 1e-9) & (d.zaman < SINIR["son"])]

t = (d.zaman - T0).dt.total_seconds().values / 86400.0
m = d.m.values
bolum = np.zeros(len(d), np.int8)
bolum[t >= gun(SINIR["egitim"])] = 1
bolum[t >= gun(SINIR["dogrulama"])] = 2
bolum[t >= gun(SINIR["test"])] = 3
ta = gun(ANA_SOK)
ana_i = int(np.argmin(np.abs(t - ta) + 100 * (m < 7.0)))
assert abs(t[ana_i] - ta) < 1e-3 and m[ana_i] >= 7.5, (t[ana_i] - ta, m[ana_i])
dizi = (d.enlem.between(*DIZI["enlem"]) & d.boylam.between(*DIZI["boylam"])).values
sonra = (t > t[ana_i])
maske = {
    "dizi_kutusu": dizi,
    "ana_sok_oncesi": (bolum == 3) & (t < t[ana_i]),
    "sonra_1g": sonra & (t <= t[ana_i] + 1),
    "sonra_7g": sonra & (t <= t[ana_i] + 7),
    "sonra_30g": sonra & (t <= t[ana_i] + 30),
}
maske["dizi_30g"] = maske["sonra_30g"] & dizi
maske["dizi_disi"] = (bolum == 3) & ~dizi
np.savez(KOK / "veri/turkiye.npz", t_gun=t, buyukluk=m, bolum=bolum, mc=mc, ana_sok_idx=ana_i,
         enlem=d.enlem.values, boylam=d.boylam.values, **maske)

dt = np.diff(t)
ozet = {
    "ham_satir": int(ham), "kutu_derinlik_sonrasi": int(kutu_n), "mc_yillik_maxc": {int(k): float(x) for k, x in yil_mc.items()},
    "mc": mc, "olay": int(len(t)), "bolum_sayisi": {int(k): int((bolum == k).sum()) for k in range(4)},
    "ana_sok": {"idx": ana_i, "zaman": str(d.zaman.iloc[ana_i]), "m": float(m[ana_i])},
    "test_M6_ustu": [[str(z), float(x)] for z, x in zip(d.zaman[(bolum == 3) & (m >= 6.0)], m[(bolum == 3) & (m >= 6.0)])],
    "maske_sayisi": {k: int((x & (bolum == 3)).sum()) for k, x in maske.items()},
    "sifir_dt": int((dt <= 0).sum()), "kutu": KUTU, "dizi_kutusu": DIZI, "sinirlar": SINIR,
}

# ---------- ek katalog: California + JMA/INGV/GeoNet, 2023-02-06 öncesi ----------
v = np.load(KOK / "veri/olaylar.npz")
e = np.load(KOK / "veri/ek_kataloglar.npz")
kes = gun(SINIR["test"])
tl, ml, kat, sb, ss, sa = [v["t_gun"]], [np.round(v["buyukluk"] - 2.5, 1)], [np.full(len(v["t_gun"]), 9, np.int8)], [0], [len(v["t_gun"])], ["California_ComCat"]
n = len(v["t_gun"])
ek_say = {"California_ComCat": int(n)}
for a, b, ad in zip(e["seg_bas"], e["seg_son"], e["seg_ad"]):
    tt, mm = e["t_gun"][a:b], e["m_goreli"][a:b]
    k = tt < kes
    tl.append(tt[k])
    ml.append(mm[k])
    kat.append(e["kat"][a:b][k])
    sb.append(n)
    n += int(k.sum())
    ss.append(n)
    sa.append(str(ad))
    ek_say[str(ad)] = int(k.sum())
assert max(x.max() for x in tl) < kes
np.savez(KOK / "veri/ek_tr.npz", t_gun=np.concatenate(tl), m_goreli=np.concatenate(ml), kat=np.concatenate(kat),
         seg_bas=np.array(sb), seg_son=np.array(ss), seg_ad=np.array(sa))
ozet["ek_tr"] = ek_say
ozet["ek_tr_toplam"] = int(n)
with open(KOK / "sonuclar/tr_veri_ozeti.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)
print(json.dumps(ozet, indent=1, ensure_ascii=False))
