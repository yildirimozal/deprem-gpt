"""Hualien testi (ON_KAYIT_11.md): sonuclar/c11_ozet.json, c11_tablolar.md.

Mac'te: /usr/bin/python3 kod/analiz11.py
"""
import json
import math
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)
N_BOOT = 2000
KM2 = 2 * math.log(1000)
v = np.load(KOK / "veri/tayvan.npz")
test = np.where(v["bolum"] == 3)[0]
tt = v["t_gun"][test]
gs = tt - v["t_gun"][int(v["ana_sok_idx"])]
KUME = {"tum": (np.ones(len(test), bool), 7), "ilk_gun": ((gs > 0) & (gs <= 1), 1 / 24),
        "ilk_7g": ((gs > 0) & (gs <= 7), 1), "30g_sonrasi": (gs > 30, 7)}


def boot(f, mk, bg):
    f2, bb = f[mk], np.floor((tt[mk] - tt[0]) / bg).astype(int)
    ub, ters = np.unique(bb, return_inverse=True)
    top, say = np.bincount(ters, weights=f2), np.bincount(ters)
    sec = rng.integers(0, len(ub), (N_BOOT, len(ub)))
    return [float(q) for q in np.quantile(top[sec].sum(1) / say[sec].sum(1), [.025, .975])]


etas, etas_bilgi = {}, {}
for k in ["E0", "E1", "E2", "E5", "E7", "E4"]:
    d = KOK / f"sonuclar/etas11_{k}"
    if (d / "etas_test.npz").exists() and (d / "etas.json").exists():
        z = np.load(d / "etas_test.npz")
        assert np.array_equal(z["idx"], test)
        etas[k] = z["tll"]
        etas_bilgi[k] = json.loads((d / "etas.json").read_text(encoding="utf-8"))["secilen"]
eksik = [k for k in ["E0", "E1", "E2", "E5", "E7", "E4"] if k not in etas]
gecerli = [k for k in ["E1", "E2", "E5", "E7"] if k in etas]
en = max(gecerli, key=lambda k: etas[k].mean())
mod, mod_s = {}, {}
for k in ["H0", "H1", "H2", "H3", "HS2"]:
    p = [KOK / f"sonuclar/c11_{k}_t{s}/test.npz" for s in (1, 2, 3)]
    if all(q.exists() for q in p):
        z = [np.load(q) for q in p]
        mod[k] = [q["tll"] for q in z]
        if "sll" in z[0].files:
            mod_s[k] = [q["sll"] - KM2 for q in z]
ozet = {"eksik_etas": eksik, "en_guclu_gecerli": en, "etas": {}, "model": {}, "fark": {}}
for k, x in etas.items():
    ozet["etas"][k] = {"tll": {a: float(x[mk].mean()) for a, (mk, _) in KUME.items()},
                       **{q: etas_bilgi[k].get(q) for q in ["a", "c", "p", "tau", "G", "H", "dallanma", "uydurma_tll"]}}
for k, xs in mod.items():
    ozet["model"][k] = {"tll": {a: float(np.mean(xs, 0)[mk].mean()) for a, (mk, _) in KUME.items()},
                        "tohum_tum": [float(q.mean()) for q in xs]}


def fark(a, b, ad_a, ad_b):
    A, Bm = (np.mean(a, 0) if isinstance(a, list) else a), (np.mean(b, 0) if isinstance(b, list) else b)
    for ad, (mk, bg) in KUME.items():
        r = {"ort": float((A - Bm)[mk].mean()), "GA": boot(A - Bm, mk, bg)}
        if isinstance(a, list):
            r["tohum"] = [float((q - (b[i] if isinstance(b, list) else Bm))[mk].mean()) for i, q in enumerate(a)]
        ozet["fark"][f"{ad_a}-{ad_b}|{ad}"] = r


for k in mod:
    fark(mod[k], etas[en], k, en)
    if "E4" in etas:
        fark(mod[k], etas["E4"], k, "E4")
for a, b in [("H1", "H0"), ("H2", "H0"), ("H3", "H1"), ("HS2", "H1")]:
    if a in mod and b in mod:
        fark(mod[a], mod[b], a, b)
r_y, r_g, r_10 = ozet["fark"][f"H1-{en}|tum"], ozet["fark"][f"H1-{en}|ilk_gun"], ozet["fark"]["H1-H0|tum"]
if r_y["GA"][0] > 0 and r_g["GA"][0] > 0 and r_10["ort"] > 0 and all(x > 0 for x in r_10["tohum"]):
    karar = "TEKRARLANDI"
elif r_y["ort"] <= 0 and r_g["ort"] <= 0:
    karar = "TEKRARLANMADI"
else:
    karar = "KISMEN"
k1 = "YOK"
if "H3" in mod:
    r3 = ozet["fark"]["H3-H1|tum"]
    k1 = "TEKRARLANDI" if r3["ort"] >= 0.01 and r3["GA"][0] > 0 and all(x > 0 for x in r3["tohum"]) else "TEKRARLANMADI"
k2 = "YOK"
uz_p = KOK / "sonuclar/etas11_uzay/etas_uzay_test.npz"
if "HS2" in mod_s and uz_p.exists():
    U = np.load(uz_p)["sll"]
    S = np.mean(mod_s["HS2"], 0)
    T = np.mean(mod["HS2"], 0)
    for ad, (mk, bg) in KUME.items():
        ozet["fark"][f"HS2_SLL-uzayETAS|{ad}"] = {"ort": float((S - U)[mk].mean()), "GA": boot(S - U, mk, bg)}
        top = (S + T) - (U + etas[en])
        ozet["fark"][f"HS2_TOPLAM-uzayETAS|{ad}"] = {"ort": float(top[mk].mean()), "GA": boot(top, mk, bg)}
    ozet["uzay"] = {"HS2_sll": {a: float(S[mk].mean()) for a, (mk, _) in KUME.items()},
                    "etas_sll": {a: float(U[mk].mean()) for a, (mk, _) in KUME.items()},
                    "etas_uzay_param": json.loads((KOK / "sonuclar/etas11_uzay/etas_uzay.json").read_text(encoding="utf-8"))["secilen"]}
    q = ozet["fark"]["HS2_TOPLAM-uzayETAS|ilk_gun"]
    k2 = "TEKRARLANDI" if q["ort"] > 0 and q["GA"][0] > 0 else "TEKRARLANMADI"
ozet["karar"] = {"birincil": karar, "ikincil_sentetik": k1, "ikincil_konum": k2}
with open(KOK / "sonuclar/c11_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)


def ga(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


sat = [f"Eksik ETAS varyantları: {eksik or 'yok'}. En güçlü geçerli ETAS: **{en}**", "",
       "| | " + " | ".join(KUME) + " | a | c | p | τ | G | H | dallanma |", "|---|" + "---:|" * (len(KUME) + 7)]
for k, e in ozet["etas"].items():
    f_ = lambda x, s: format(x, s) if x is not None else "–"  # noqa: E731
    sat.append(f"| ETAS {k} | " + " | ".join(f"{e['tll'][a]:.3f}" for a in KUME) + f" | {f_(e['a'], '.3f')} | {f_(e['c'], '.4f')} | "
               f"{f_(e['p'], '.3f')} | {f_(e['tau'], '.0f')} | {f_(e['G'], '.2f')} | {f_(e['H'], '.2f')} | {f_(e['dallanma'], '.2f')} |")
for k, e in ozet["model"].items():
    sat.append(f"| {k} | " + " | ".join(f"{e['tll'][a]:.3f}" for a in KUME) + " | | | | | | | |")
sat += ["", "| Fark | " + " | ".join(KUME) + " | tohumlar (tüm) |", "|---|" + "---|" * (len(KUME) + 1)]
for an in sorted({k.split("|")[0] for k in ozet["fark"]}):
    sat.append(f"| {an} | " + " | ".join(f"{ozet['fark'][f'{an}|{a}']['ort']:+.3f} {ga(ozet['fark'][f'{an}|{a}']['GA'])}" for a in KUME)
               + " | " + ", ".join(f"{x:+.3f}" for x in ozet["fark"][f"{an}|tum"].get("tohum", [])) + " |")
sat += ["", f"**Karar (ön kayıtlı):** birincil {karar}; sentetik ön eğitim {k1}; konum {k2}"]
(KOK / "sonuclar/c11_tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))
