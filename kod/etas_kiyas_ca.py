"""İkinci testin modellerini, kendi zamansal ETAS'ımıza (kod/etas_zaman.py, California 1981-2007 uydurması)
karşı yeniden kıyaslar. Çıktı: sonuclar/c2_etas_zaman_kiyas.json.  Mac'te: /usr/bin/python3 kod/etas_kiyas_ca.py"""
import json
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent
v = np.load(KOK / "veri/olaylar.npz")
test = np.where(v["bolum"] == 3)[0]
tt = v["t_gun"][test]
e = np.load(KOK / "sonuclar/etas_ca_dogrulama/etas_test.npz")
assert np.array_equal(e["idx"], test)
eo, ez = v["etas_tll"][test], e["tll"]
rng = np.random.default_rng(0)


def boot(f, mk, bg):
    f2, bb = f[mk], np.floor((tt[mk] - tt[0]) / bg).astype(int)
    ub, ters = np.unique(bb, return_inverse=True)
    top, say = np.bincount(ters, weights=f2), np.bincount(ters)
    s = rng.integers(0, len(ub), (2000, len(ub)))
    return [float(x) for x in np.quantile(top[s].sum(1) / say[s].sum(1), [.025, .975])]


kume = {"tum_test": np.ones(len(test), bool), "yogun_M6_30g": v["yogun_M6_30g"][test], "yogun_M6_1g": v["yogun_M6_1g"][test],
        "sakin": ~v["yogun_M6_30g"][test], "El Mayor 2010 M7.2": v["dizi::2010-04-04 M7.2"][test],
        "Ridgecrest 2019 M7.1": v["dizi::2019-07-06 M7.1"][test]}
kos = {k: [np.load(KOK / f"sonuclar/c2_{k}_t{s}/test.npz")["tll"] for s in (1, 2, 3)] for k in ["M0", "M1", "M2", "M3", "M4", "M5"]}
out = {"etas_zaman": json.loads((KOK / "sonuclar/etas_ca_dogrulama/etas.json").read_text(encoding="utf-8"))["secilen"], "kume": {}}
for ad, mk in kume.items():
    bg = 30 if ad in ("tum_test", "yogun_M6_30g", "sakin") else 1
    r = {"n": int(mk.sum()), "ETAS_EarthquakeNPP": float(eo[mk].mean()), "ETAS_zaman": float(ez[mk].mean()),
         "ETAS_zaman_eksi_NPP": float((ez - eo)[mk].mean()), "ETAS_zaman_eksi_NPP_GA": boot(ez - eo, mk, bg), "blok_gun": bg}
    for k, xs in kos.items():
        f = np.mean(xs, 0) - ez
        r[f"{k}-ETAS_zaman"] = {"ort": float(f[mk].mean()), "GA": boot(f, mk, bg),
                                "tohum": [float((x - ez)[mk].mean()) for x in xs]}
    out["kume"][ad] = r
with open(KOK / "sonuclar/c2_etas_zaman_kiyas.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
for ad, r in out["kume"].items():
    print(f"{ad:22s} ETASz-NPP {r['ETAS_zaman_eksi_NPP']:+.3f}  " + "  ".join(
        f"{k}:{r[f'{k}-ETAS_zaman']['ort']:+.3f}[{r[f'{k}-ETAS_zaman']['GA'][0]:+.3f},{r[f'{k}-ETAS_zaman']['GA'][1]:+.3f}]" for k in kos))
