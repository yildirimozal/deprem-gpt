"""Güçlendirilmiş ETAS'a karşı (ON_KAYIT_4.md): sonuclar/c4_ozet.json, c4_tablolar.md, sekiller/c4_sekil1.

Mac'te: /usr/bin/python3 kod/analiz4.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
N_BOOT = 2000
rng = np.random.default_rng(0)
ETAS_TR = [k for k in ["E0", "E1", "E2", "E3", "E5", "E4"] if (KOK / f"sonuclar/etas4_{k}/etas_test.npz").exists()]
GECERLI = [k for k in ["E1", "E2", "E3", "E5"] if k in ETAS_TR]
EKSIK = [k for k in ["E0", "E1", "E2", "E3", "E5", "E4", "C1", "C2"] if not (KOK / f"sonuclar/etas4_{k}/etas_test.npz").exists()]
print("eksik varyantlar:", EKSIK)
ACIK = {"E0": "Baseline (test 3)", "E1": "Tapered Omori", "E2": "Subcritical (n ≤ 0.99)",
        "E3": "Pooled catalogs", "E5": "Pooled + tapered", "E4": "Oracle (fit on test year)",
        "C1": "Tapered Omori", "C2": "Subcritical (n ≤ 0.99)", "C0": "Baseline (etas_zaman)"}


def boot(f, mk, tt, bg):
    f2, bb = f[mk], np.floor((tt[mk] - tt[0]) / bg).astype(int)
    ub, ters = np.unique(bb, return_inverse=True)
    top, say = np.bincount(ters, weights=f2), np.bincount(ters)
    s = rng.integers(0, len(ub), (N_BOOT, len(ub)))
    return [float(x) for x in np.quantile(top[s].sum(1) / say[s].sum(1), [.025, .975])] + [int(len(ub))]


def etas_yukle(ad, test):
    z = np.load(KOK / f"sonuclar/{ad}/etas_test.npz")
    assert np.array_equal(z["idx"], test)
    return z["tll"], json.loads((KOK / f"sonuclar/{ad}/etas.json").read_text(encoding="utf-8"))


# ---------------- Türkiye ----------------
v = np.load(KOK / "veri/turkiye.npz")
test = np.where(v["bolum"] == 3)[0]
tt = v["t_gun"][test]
ana = int(v["ana_sok_idx"])
kume = {"tum_test": (np.ones(len(test), bool), 7), "sonra_1g": (v["sonra_1g"][test], None),
        "dizi_30g": (v["dizi_30g"][test], 1), "30g_sonrasi": (tt > v["t_gun"][ana] + 30, 7),
        "dizi_disi": (v["dizi_disi"][test], 7)}
mod = {k: [np.load(KOK / f"sonuclar/c3_{k}_t{s}/test.npz")["tll"] for s in (1, 2, 3)] for k in ["T0", "T1", "T2"]}
etas, bilgi = {}, {}
for k in ETAS_TR:
    etas[k], bilgi[k] = etas_yukle(f"etas4_{k}", test)
e_eski = np.load(KOK / "sonuclar/etas_tr/etas_test.npz")["tll"]
ozet = {"eksik_varyantlar": EKSIK, "E0_eski_kod_farki_max": float(np.abs(etas["E0"] - e_eski).max()), "etas": {}, "kiyas": {}}
for k in ETAS_TR:
    s = bilgi[k]["secilen"]
    ozet["etas"][k] = {"test_tll": float(etas[k].mean()), "uydurma_tll": s["uydurma_tll"], "a": s["a"], "c": s["c"],
                       "p": s["p"], "tau": s["tau"], "K": s["K"], "mu_turkiye": s["mu"][0], "dallanma": s["dallanma"],
                       "denemeler_uydurma_tll": [d["uydurma_tll"] for d in bilgi[k]["denemeler"]],
                       "kume": {ad: float(etas[k][mk].mean()) for ad, (mk, _) in kume.items()}}
en_guclu = max(GECERLI, key=lambda k: ozet["etas"][k]["test_tll"])
ozet["en_guclu_gecerli_etas"] = en_guclu
for m_ad, xs in mod.items():
    ort = np.mean(xs, 0)
    for k in ETAS_TR:
        for ad, (mk, bg) in kume.items():
            f = ort - etas[k]
            r = {"ort": float(f[mk].mean()), "tohum": [float((x - etas[k])[mk].mean()) for x in xs]}
            if bg:
                r["GA"] = boot(f, mk, tt, bg)
            ozet["kiyas"][f"{m_ad}-{k}|{ad}"] = r
r = ozet["kiyas"][f"T1-{en_guclu}|tum_test"]
karar = "SAGLAM" if r["ort"] > 0 and r["GA"][0] > 0 else ("ZAYIFLADI" if r["ort"] > 0 else "COKTU")
ozet["karar"] = {"karar": karar, "en_guclu": en_guclu, "T1-en_guclu": r}

# ---------------- California (ikincil) ----------------
vc = np.load(KOK / "veri/olaylar.npz")
test_c = np.where(vc["bolum"] == 3)[0]
tt_c = vc["t_gun"][test_c]
kume_c = {"tum_test": (np.ones(len(test_c), bool), 30), "yogun_M6_1g": (vc["yogun_M6_1g"][test_c], 1),
          "sakin": (~vc["yogun_M6_30g"][test_c], 30)}
etas_c = {"NPP": vc["etas_tll"][test_c]}
etas_c["C0"], _ = etas_yukle("etas_ca_dogrulama", test_c)
bilgi_c = {}
for k in ["C1", "C2"]:
    if k not in EKSIK:
        etas_c[k], bilgi_c[k] = etas_yukle(f"etas4_{k}", test_c)
ozet["california"] = {"etas": {k: float(x.mean()) for k, x in etas_c.items()}, "kiyas": {},
                      "parametre": {k: b["secilen"] for k, b in bilgi_c.items()}}
for m_ad in ["M0", "M1", "M5"]:
    xs = [np.load(KOK / f"sonuclar/c2_{m_ad}_t{s}/test.npz")["tll"] for s in (1, 2, 3)]
    ort = np.mean(xs, 0)
    for k in etas_c:
        for ad, (mk, bg) in kume_c.items():
            f = ort - etas_c[k]
            ozet["california"]["kiyas"][f"{m_ad}-{k}|{ad}"] = {"ort": float(f[mk].mean()), "GA": boot(f, mk, tt_c, bg),
                                                               "tohum": [float((x - etas_c[k])[mk].mean()) for x in xs]}
with open(KOK / "sonuclar/c4_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)


# ---------------- tablolar ----------------
def ga(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


sat = [f"Eksik varyantlar: {EKSIK or 'yok'}", f"E0 yeni kod ile eski kod en büyük fark: {ozet['E0_eski_kod_farki_max']:.2e}", "",
       "| ETAS | Test TLL | İlk gün | Dizi 30 g | 30 g sonrası | Dizi dışı | a | c | p | τ (gün) | Dallanma | Uydurma TLL (3 başlangıç) |",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for k in ETAS_TR:
    e = ozet["etas"][k]
    sat.append(f"| {k} {ACIK[k]} | {e['test_tll']:.3f} | {e['kume']['sonra_1g']:.3f} | {e['kume']['dizi_30g']:.3f} | "
               f"{e['kume']['30g_sonrasi']:.3f} | {e['kume']['dizi_disi']:.3f} | {e['a']:.3f} | {e['c']:.4f} | {e['p']:.3f} | "
               + (f"{e['tau']:.0f}" if e["tau"] else "–") + f" | {e['dallanma']:.3f} | "
               + ", ".join(f"{x:.5f}" for x in e["denemeler_uydurma_tll"]) + " |")
sat += ["", f"En güçlü geçerli ETAS (test TLL'sine göre): **{en_guclu}**", "",
        "| Kıyas | Tüm test (7 g GA) | İlk gün (tohumlar) | Dizi 30 g (1 g GA) | 30 g sonrası (7 g GA) | Dizi dışı (7 g GA) |",
        "|---|---|---|---|---|---|"]
for m_ad in ["T1", "T2", "T0"]:
    for k in ETAS_TR:
        q = lambda ad: ozet["kiyas"][f"{m_ad}-{k}|{ad}"]  # noqa: E731
        sat.append(f"| {m_ad} − {k} | {q('tum_test')['ort']:+.3f} {ga(q('tum_test')['GA'])} | "
                   f"{q('sonra_1g')['ort']:+.3f} (" + ", ".join(f"{y:+.3f}" for y in q('sonra_1g')['tohum']) + ") | "
                   f"{q('dizi_30g')['ort']:+.3f} {ga(q('dizi_30g')['GA'])} | {q('30g_sonrasi')['ort']:+.3f} {ga(q('30g_sonrasi')['GA'])} | "
                   f"{q('dizi_disi')['ort']:+.3f} {ga(q('dizi_disi')['GA'])} |")
sat += ["", f"**Karar (ön kayıtlı kural): {karar}** (T1 − {en_guclu} = {r['ort']:+.3f} {ga(r['GA'])}, tohumlar "
        + ", ".join(f"{y:+.3f}" for y in r["tohum"]) + ")", "",
        "California (ikincil): ETAS test TLL " + ", ".join(f"{k} {x:.3f}" for k, x in ozet["california"]["etas"].items()), "",
        "| Kıyas | Tüm test (30 g GA) | M≥6 sonrası 1 g (1 g GA) | Sakin (30 g GA) |", "|---|---|---|---|"]
for m_ad in ["M1", "M0", "M5"]:
    for k in etas_c:
        q = lambda ad: ozet["california"]["kiyas"][f"{m_ad}-{k}|{ad}"]  # noqa: E731
        sat.append(f"| {m_ad} − {k} | " + " | ".join(f"{q(ad)['ort']:+.3f} {ga(q(ad)['GA'])}" for ad in kume_c) + " |")
(KOK / "sonuclar/c4_tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))

# ---------------- şekil ----------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "savefig.dpi": 300})
KOYU = "#404040"
MAVI, MAVI_ACIK, MACENTA, MACENTA_ACIK, TURUNCU, KIRMIZI = "#1f3fcf", "#b9c6f2", "#d61fd0", "#f7c2f3", "#ef8f3a", "#e0342b"
paneller = [("tum_test", "Full test year"), ("dizi_30g", "Sequence box, first 30 d"), ("30g_sonrasi", "Later than 30 d")]
fig, eks = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
y = np.arange(len(ETAS_TR))[::-1]
for j, (ax, (ad, bas)) in enumerate(zip(eks, paneller)):
    for yy, k in zip(y, ETAS_TR):
        x = ozet["kiyas"][f"T1-{k}|{ad}"]
        lo, hi = x["GA"][:2]
        anl = lo > 0 or hi < 0
        ax.barh(yy, x["ort"], height=0.6, color=MAVI if anl else "white", edgecolor="black" if anl else MAVI, hatch=None if anl else "////")
        ax.errorbar(x["ort"], yy, xerr=[[x["ort"] - lo], [hi - x["ort"]]], color="black", capsize=3, lw=1)
    ax.axvline(0, color=KIRMIZI, lw=1.0, ls="--")
    ax.set_title(f"({'abc'[j]}) {bas}", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k}: {ACIK[k]}" for k in ETAS_TR], fontsize=8)
fig.supxlabel("T1 temporal log-likelihood minus ETAS variant (nats per event)", fontweight="bold", fontsize=10)
fig.tight_layout()
for uz in ("png", "pdf"):
    fig.savefig(KOK / f"sekiller/c4_sekil1_etas_varyantlari.{uz}")
plt.close(fig)
