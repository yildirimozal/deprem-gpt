"""Noto testi (ON_KAYIT_7.md): sonuclar/c7_ozet.json, c7_tablolar.md, sekiller/c7_*.

Mac'te: /usr/bin/python3 kod/analiz7.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)
N_BOOT = 2000
v = np.load(KOK / "veri/japonya.npz")
b, t = v["bolum"], v["t_gun"]
test = np.where(b == 3)[0]
tt = t[test]
ana, hy = int(v["ana_sok_idx"]), int(v["hyuga_idx"])
gun_sonra = tt - t[ana]
KOD = ["J0", "J1", "J2"]
ACIK = {"J0": "Japan only", "J1": "Japan + other catalogs", "J2": "Other catalogs only"}
ETAS_TUM = ["E0", "E1", "E2", "E5", "E6", "E7", "E4", "E8"]
GECERLI_ADAY = ["E1", "E2", "E5", "E6", "E7"]
ETAS_ACIK = {"E0": "Unconstrained", "E1": "Tapered Omori", "E2": "Subcritical", "E5": "Pooled + tapered",
             "E6": "Incompleteness (literature)", "E7": "Incompleteness (fitted)", "E4": "Oracle", "E8": "Oracle + incompleteness"}

etas, bilgi = {}, {}
for k in ETAS_TUM:
    d = KOK / f"sonuclar/etas7_{k}"
    if (d / "etas_test.npz").exists():
        z = np.load(d / "etas_test.npz")
        assert np.array_equal(z["idx"], test)
        etas[k] = z["tll"]
        bilgi[k] = json.loads((d / "etas.json").read_text(encoding="utf-8"))
EKSIK = [k for k in ETAS_TUM if k not in etas]
GECERLI = [k for k in GECERLI_ADAY if k in etas]
print("eksik ETAS:", EKSIK)
mod = {k: [np.load(KOK / f"sonuclar/c7_{k}_t{s}/test.npz")["tll"] for s in (1, 2, 3)] for k in KOD}
kayit = {k: [json.loads((KOK / f"sonuclar/c7_{k}_t{s}/kayit.json").read_text(encoding="utf-8")) for s in (1, 2, 3)] for k in KOD}


def boot(f, mk, bg):
    f2, bb = f[mk], np.floor((tt[mk] - tt[0]) / bg).astype(int)
    ub, te = np.unique(bb, return_inverse=True)
    top, say = np.bincount(te, weights=f2), np.bincount(te)
    s = rng.integers(0, len(ub), (N_BOOT, len(ub)))
    return [float(x) for x in np.quantile(top[s].sum(1) / say[s].sum(1), [.025, .975])] + [int(len(ub))]


KUME = {"tum_test": (np.ones(len(test), bool), 7), "noto_1g": (v["sonra_1g"][test], 1 / 24),
        "noto_7g": (v["sonra_7g"][test], 1), "noto_kutu_30g": (v["dizi_30g"][test], 1),
        "30g_sonrasi": (gun_sonra > 30, 7), "noto_kutu_disi": (v["dizi_disi"][test], 7),
        "hyuga_1g": (v["hyuga_1g"][test], 1 / 24)}
ozet = {"eksik_etas": EKSIK, "etas": {}, "model": {}, "kiyas": {}}
for k in etas:
    s = bilgi[k]["secilen"]
    ozet["etas"][k] = {"test_tll": float(etas[k].mean()), "kume": {ad: float(etas[k][mk].mean()) for ad, (mk, _) in KUME.items()},
                       **{q: s.get(q) for q in ["a", "c", "p", "tau", "G", "H", "dallanma", "uydurma_tll"]},
                       "denemeler": [d["uydurma_tll"] for d in bilgi[k]["denemeler"]]}
for k in KOD:
    ozet["model"][k] = {"tohum_test": [float(x.mean()) for x in mod[k]],
                        "kume": {ad: float(np.mean([x[mk].mean() for x in mod[k]])) for ad, (mk, _) in KUME.items()},
                        "en_iyi_adim": [r["en_iyi_adim"] for r in kayit[k]]}
en_guclu = max(GECERLI, key=lambda k: ozet["etas"][k]["test_tll"])
ozet["en_guclu_gecerli"] = en_guclu
for m_ad in KOD:
    ort = np.mean(mod[m_ad], 0)
    for k in etas:
        for ad, (mk, bg) in KUME.items():
            f = ort - etas[k]
            ozet["kiyas"][f"{m_ad}-{k}|{ad}"] = {"ort": float(f[mk].mean()), "GA": boot(f, mk, bg),
                                                 "tohum": [float((x - etas[k])[mk].mean()) for x in mod[m_ad]]}
for a, c in [("J1", "J0"), ("J2", "J0"), ("J1", "J2")]:
    for ad, (mk, bg) in KUME.items():
        f = np.mean(mod[a], 0) - np.mean(mod[c], 0)
        ozet["kiyas"][f"{a}-{c}|{ad}"] = {"ort": float(f[mk].mean()), "GA": boot(f, mk, bg),
                                          "tohum": [float((x - y)[mk].mean()) for x, y in zip(mod[a], mod[c])]}
r_yil = ozet["kiyas"][f"J1-{en_guclu}|tum_test"]
r_gun = ozet["kiyas"][f"J1-{en_guclu}|noto_1g"]
r_10 = ozet["kiyas"]["J1-J0|tum_test"]
if r_yil["GA"][0] > 0 and r_gun["GA"][0] > 0 and r_10["ort"] > 0 and all(x > 0 for x in r_10["tohum"]):
    karar = "TEKRARLANDI"
elif r_yil["ort"] <= 0 and r_gun["ort"] <= 0:
    karar = "TEKRARLANMADI"
else:
    karar = "KISMEN"
ozet["karar"] = {"karar": karar, "en_guclu": en_guclu, "eksik_varyantlar": EKSIK, "J1-en_guclu_yil": r_yil,
                 "J1-en_guclu_ilk_gun": r_gun, "J1-J0_yil": r_10}
with open(KOK / "sonuclar/c7_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)


def ga(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


ADLAR = list(KUME)
sat = [f"Eksik ETAS varyantları: {EKSIK or 'yok'}", "",
       "| | " + " | ".join(f"{a} (n={int(KUME[a][0].sum())})" for a in ADLAR) + " | a | c | p | τ | G | H | dallanma |",
       "|---|" + "---:|" * (len(ADLAR) + 7)]
for k in etas:
    e = ozet["etas"][k]

    def f(x, fmt):
        return format(x, fmt) if x is not None else "–"
    sat.append(f"| {k} {ETAS_ACIK[k]} | " + " | ".join(f"{e['kume'][a]:.3f}" for a in ADLAR)
               + f" | {f(e['a'], '.3f')} | {f(e['c'], '.4f')} | {f(e['p'], '.3f')} | {f(e['tau'], '.0f')} | {f(e['G'], '.2f')} | {f(e['H'], '.2f')} | {f(e['dallanma'], '.2f')} |")
for k in KOD:
    m = ozet["model"][k]
    sat.append(f"| {k} {ACIK[k]} | " + " | ".join(f"{m['kume'][a]:.3f}" for a in ADLAR) + " | | | | | | | |")
sat += ["", f"En güçlü geçerli ETAS ({', '.join(GECERLI)} içinden, test TLL'sine göre): **{en_guclu}**", "",
        "| Kıyas | " + " | ".join(ADLAR) + " |", "|---|" + "---|" * len(ADLAR)]
for m_ad in ["J1", "J2", "J0"]:
    for k in etas:
        sat.append(f"| {m_ad} − {k} | " + " | ".join(
            f"{ozet['kiyas'][f'{m_ad}-{k}|{a}']['ort']:+.3f} {ga(ozet['kiyas'][f'{m_ad}-{k}|{a}']['GA'])}" for a in ADLAR) + " |")
for a_, c_ in [("J1", "J0"), ("J2", "J0"), ("J1", "J2")]:
    sat.append(f"| {a_} − {c_} | " + " | ".join(
        f"{ozet['kiyas'][f'{a_}-{c_}|{a}']['ort']:+.3f} {ga(ozet['kiyas'][f'{a_}-{c_}|{a}']['GA'])}" for a in ADLAR) + " |")
sat += ["", "Tohumlar (tüm test): " + "; ".join(f"{k}: " + ", ".join(f"{x:.4f}" for x in ozet['model'][k]['tohum_test']) for k in KOD),
        f"J1 − J0 tohum eşleşik (tüm test): " + ", ".join(f"{x:+.3f}" for x in r_10["tohum"]),
        f"J1 − {en_guclu} tohumlar (tüm test): " + ", ".join(f"{x:+.3f}" for x in r_yil["tohum"]),
        f"J1 − {en_guclu} tohumlar (Noto ilk gün): " + ", ".join(f"{x:+.3f}" for x in r_gun["tohum"]), "",
        f"**Karar (ön kayıtlı kural): {karar}**"]
(KOK / "sonuclar/c7_tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))

# ---------- şekiller ----------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "savefig.dpi": 300})
KOYU, ORTA = "#404040", "#8c8c8c"
MAVI, MAVI_ACIK, MACENTA, MACENTA_ACIK, TURUNCU, KIRMIZI = "#1f3fcf", "#b9c6f2", "#d61fd0", "#f7c2f3", "#ef8f3a", "#e0342b"
liste = [k for k in ETAS_TUM if k in etas]
paneller = [("tum_test", "Full test year (2024)"), ("noto_1g", "First day after Noto M7.6"), ("30g_sonrasi", "Later than 30 d")]
fig, eks = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=True)
y = np.arange(len(liste))[::-1]
for j, (ax, (ad, bas)) in enumerate(zip(eks, paneller)):
    for yy, k in zip(y, liste):
        x = ozet["kiyas"][f"J1-{k}|{ad}"]
        lo, hi = x["GA"][:2]
        anl = lo > 0 or hi < 0
        ax.barh(yy, x["ort"], height=0.6, color=MAVI if anl else "white", edgecolor="black" if anl else MAVI, hatch=None if anl else "////")
        ax.errorbar(x["ort"], yy, xerr=[[x["ort"] - lo], [hi - x["ort"]]], color="black", capsize=3, lw=1)
    ax.axvline(0, color=KIRMIZI, lw=1.0, ls="--")
    ax.set_title(f"({'abc'[j]}) {bas}", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k}: {ETAS_ACIK[k]}" for k in liste], fontsize=8)
fig.supxlabel("J1 temporal log-likelihood minus ETAS variant (nats per event)", fontweight="bold", fontsize=10)
fig.tight_layout()
for uz in ("png", "pdf"):
    fig.savefig(KOK / f"sekiller/c7_sekil1_etas_varyantlari.{uz}")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.4))
stil = {"J0": dict(color=TURUNCU, ls="-"), "J1": dict(color=MAVI, ls="-"), "J2": dict(color=KOYU, ls="--")}
for k, st in stil.items():
    ax.plot(gun_sonra, np.cumsum(np.mean(mod[k], 0) - etas[en_guclu]), lw=1.2, label=f"{k}: {ACIK[k]}", **st)
for i, et, yon in [(ana, "Noto M7.6", "left"), (hy, "Hyuganada M7.1", "right")]:
    xg = t[i] - t[ana]
    ax.axvline(xg, color=ORTA, lw=0.6, ls=":")
    ax.text(xg, 0.97, f" {et} " , va="top", ha=yon, fontsize=8, transform=ax.get_xaxis_transform())
ax.axhline(0, color=KIRMIZI, lw=1.0, ls="--")
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlim(gun_sonra[0] - 0.01, gun_sonra[-1] * 1.05)
ax.set_xlabel("Days since the Noto M7.6 mainshock")
ax.set_ylabel(f"Cumulative gain over ETAS {en_guclu} (nats)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)
fig.tight_layout()
for uz in ("png", "pdf"):
    fig.savefig(KOK / f"sekiller/c7_sekil2_kumulatif.{uz}")
plt.close(fig)
