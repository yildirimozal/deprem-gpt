"""Hualien testi verisi (ON_KAYIT_11.md).

1) veri/tayvan.npz: ISC'den TAP (CWA) depremleri, TAP ML; olay türü deprem (ISC türünün ikinci harfi 'e'),
   21–26°K, 119–123°D, derinlik <= 70 km, M >= Mc. Mc: 2007–2023 yıllık MAXC + 0.2'nin en büyüğü.
   bolum: 0 geçmiş (2005–2006), 1 eğitim (2007–2020), 2 doğrulama (2021 – 2024-04-02 00:00),
          3 test (2024-04-02 00:00 – 2025-01-01).
2) veri/ek_tw.npz: California + JMA + INGV + GeoNet + Türkiye (KOERI), hepsi 2024-04-02 öncesi.
3) veri/konum.npz'ye Tayvan_TAP anahtarları eklenir (konum girdisi/çıktısı için).
Mac'te: /usr/bin/python3 kod/hazirla_tw.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
V = KOK / "veri"
T0 = pd.Timestamp("1971-01-01")
SINIR = {"egitim": "2007-01-01", "dogrulama": "2021-01-01", "test": "2024-04-02", "son": "2025-01-01"}
DIZI = dict(enlem=(23.0, 25.0), boylam=(121.0, 122.5))


def gun(ts):
    return (pd.Timestamp(ts) - T0).total_seconds() / 86400.0


def mc_maxc(m):
    u, n = np.unique(np.round(m * 10).astype(int), return_counts=True)
    return u[np.argmax(n)] / 10 + 0.2


d = pd.read_csv(V / "tayvan/isc_tap_2005_2024.csv", dtype={"eventid": str, "tur": str})
ham = len(d)
tur_say = d.tur.str.strip().value_counts().to_dict()
d = d[d.tur.str.strip().str[1] == "e"].copy()
d["zaman"] = pd.to_datetime(d.tarih + " " + d.saat, format="ISO8601")
d = d[d.enlem.between(21, 26) & d.boylam.between(119, 123)]
derinlik_yok = int(d.derinlik.isna().sum())
d = d[d.derinlik <= 70.0]
d["m"] = np.floor(d.ml.astype(float) * 10 + 0.5 + 1e-9) / 10
d = d.sort_values("zaman", kind="stable").reset_index(drop=True)
on = d[(d.zaman >= SINIR["egitim"]) & (d.zaman < "2024-01-01")]
yil_mc = on.groupby(on.zaman.dt.year).m.apply(lambda x: mc_maxc(x.values))
mc = float(yil_mc.max())
d = d[(d.m >= mc - 1e-9) & (d.zaman < SINIR["son"])].reset_index(drop=True)
t = (d.zaman - T0).dt.total_seconds().values / 86400.0
m = d.m.values
bolum = np.zeros(len(d), np.int8)
bolum[t >= gun(SINIR["egitim"])] = 1
bolum[t >= gun(SINIR["dogrulama"])] = 2
bolum[t >= gun(SINIR["test"])] = 3
ana_k = ((d.zaman >= "2024-04-02 23:50") & (d.zaman < "2024-04-03 00:10") & (m >= 7.0)).values
assert ana_k.sum() == 1, ana_k.sum()
ana = int(np.where(ana_k)[0][0])
kutu = (d.enlem.between(*DIZI["enlem"]) & d.boylam.between(*DIZI["boylam"])).values
s = t - t[ana]
maske = {"dizi_kutusu": kutu, "sonra_1g": (s > 0) & (s <= 1), "sonra_7g": (s > 0) & (s <= 7), "sonra_30g": (s > 0) & (s <= 30)}
maske["dizi_30g"] = maske["sonra_30g"] & kutu
maske["dizi_disi"] = (bolum == 3) & ~kutu
np.savez(V / "tayvan.npz", t_gun=t, buyukluk=m, bolum=bolum, mc=mc, ana_sok_idx=ana, enlem=d.enlem.values.astype(float),
         boylam=d.boylam.values.astype(float), **maske)

ozet = {"ham_satir": int(ham), "tur_sayilari": {str(k): int(v) for k, v in tur_say.items()}, "derinlik_yok": derinlik_yok,
        "mc_yillik_maxc": {int(k): float(x) for k, x in yil_mc.items()}, "mc": mc, "olay": int(len(t)),
        "bolum_sayisi": {int(k): int((bolum == k).sum()) for k in range(4)},
        "ana_sok": {"idx": ana, "zaman": str(d.zaman[ana]), "m": float(m[ana]), "enlem": float(d.enlem[ana]), "boylam": float(d.boylam[ana])},
        "egitim_en_buyuk": float(m[bolum == 1].max()), "dogrulama_en_buyuk": float(m[bolum == 2].max()),
        "test_M6_ustu": [[str(z), float(x)] for z, x in zip(d.zaman[(bolum == 3) & (m >= 6.0)], m[(bolum == 3) & (m >= 6.0)])],
        "maske_sayisi": {k: int((x & (bolum == 3)).sum()) for k, x in maske.items()}, "sinirlar": SINIR, "dizi_kutusu": DIZI}

# ---------- ek katalog ----------
kes = gun(SINIR["test"])
vc, e, vt = np.load(V / "olaylar.npz"), np.load(V / "ek_kataloglar.npz"), np.load(V / "turkiye.npz")
tl, ml, sb, ss, sa = [vc["t_gun"]], [np.round(vc["buyukluk"] - 2.5, 1)], [0], [len(vc["t_gun"])], ["California_ComCat"]
n = len(vc["t_gun"])
for a, b, ad in zip(e["seg_bas"], e["seg_son"], e["seg_ad"]):
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
np.savez(V / "ek_tw.npz", t_gun=np.concatenate(tl), m_goreli=np.concatenate(ml),
         kat=np.concatenate([np.full(len(x), i, np.int8) for i, x in enumerate(tl)]),
         seg_bas=np.array(sb), seg_son=np.array(ss), seg_ad=np.array(sa))
ozet["ek_tw"] = {a: int(s2 - s1) for a, s1, s2 in zip(sa, sb, ss)}

# ---------- konum.npz'ye Tayvan ----------
kn = dict(np.load(V / "konum.npz"))
kn.update(Tayvan_TAP_t=t, Tayvan_TAP_enlem=d.enlem.values.astype(float), Tayvan_TAP_boylam=d.boylam.values.astype(float))
np.savez(V / "konum.npz", **kn)
with open(KOK / "sonuclar/tw_veri_ozeti.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in ozet.items() if k != "mc_yillik_maxc"}, indent=1, ensure_ascii=False))
print("yıllık MAXC+0.2:", ozet["mc_yillik_maxc"])
