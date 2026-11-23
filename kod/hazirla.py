"""ETAS'ın değerlendirdiği olay kümesinden model girdisini hazırlar.

Girdi : veri/earthquakenpp/Experiments/ETAS/output_data_ComCat_25/augmented_catalog.csv
Çıktı : veri/olaylar.npz  (t_gun, buyukluk, bolum, etas_tll, yogun maskeleri)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parent.parent
A = KOK / "veri/earthquakenpp/Experiments/ETAS/output_data_ComCat_25/augmented_catalog.csv"

df = pd.read_csv(A, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
t0 = np.datetime64("1971-01-01T00:00:00")
t = (df.time.values.astype("datetime64[us]") - t0).astype("timedelta64[us]").astype(np.float64) / 86400e6
m = np.round(df.magnitude.values, 1)

sinirlar = {k: (np.datetime64(v) - t0).astype("timedelta64[us]").astype(np.float64) / 86400e6
            for k, v in {"egitim": "1981-01-01", "dogrulama": "1998-01-01",
                         "test": "2007-01-01", "son": "2020-01-17"}.items()}
bolum = np.zeros(len(t), dtype=np.int8)               # 0 yardımcı
bolum[t >= sinirlar["egitim"]] = 1                     # 1 eğitim
bolum[t >= sinirlar["dogrulama"]] = 2                  # 2 doğrulama
bolum[(t >= sinirlar["test"]) & (t <= sinirlar["son"])] = 3  # 3 test

etas_tll = df.TLL.values.astype(np.float64)
etas_tll[bolum != 3] = np.nan
assert abs(np.nanmean(etas_tll) - 1.4343428344882627) < 1e-9

# Yoğun dönem maskeleri (ON_KAYIT.md)
test = bolum == 3
ana = np.where(test & (m >= 6.0))[0]


def pencere(esik, gun):
    k = np.zeros(len(t), bool)
    for i in ana:
        if m[i] >= esik:
            k |= (t > t[i]) & (t <= t[i] + gun) & test
    return k


maskeler = {
    "yogun_M6_30g": pencere(6.0, 30),
    "yogun_M6_7g": pencere(6.0, 7),
    "yogun_M6_1g": pencere(6.0, 1),
    "yogun_M7_30g": pencere(7.0, 30),
}
diziler = {}
for i in ana:
    ad = f"{df.time[i]:%Y-%m-%d} M{m[i]:.1f}"
    diziler[ad] = (t > t[i]) & (t <= t[i] + 30) & test

np.savez(KOK / "veri/olaylar.npz", t_gun=t, buyukluk=m, bolum=bolum, etas_tll=etas_tll,
         **maskeler, **{"dizi::" + k: v for k, v in diziler.items()})
dt = np.diff(t)
ozet = {
    "olay": int(len(t)), "yardimci": int((bolum == 0).sum()), "egitim": int((bolum == 1).sum()),
    "dogrulama": int((bolum == 2).sum()), "test": int(test.sum()),
    "etas_tll_test_ortalama": float(np.nanmean(etas_tll)),
    "sifir_dt_toplam": int((dt <= 0).sum()),
    "sifir_dt_test": int(((dt <= 0) & test[1:]).sum()),
    "yogun_olay_sayisi": {k: int(v.sum()) for k, v in maskeler.items()},
    "dizi_olay_sayisi": {k: int(v.sum()) for k, v in diziler.items()},
    "etas_tll_yogun": {k: float(etas_tll[v].mean()) for k, v in maskeler.items()},
}
(KOK / "sonuclar").mkdir(exist_ok=True)
with open(KOK / "sonuclar/veri_ozeti.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)
print(json.dumps(ozet, indent=1, ensure_ascii=False))
