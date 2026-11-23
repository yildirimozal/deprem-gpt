"""Çok katalog testi (ON_KAYIT_2.md) sonuçları: sonuclar/c2_ozet.json, c2_tablolar.md, sekiller/c2_*.

Mac'te: /usr/bin/python3 kod/analiz2.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
KOD = ["M0", "M1", "M2", "M3", "M4", "M5"]
ACIK = {"M0": "California only, small, L=512", "M1": "Multi-catalog, small, L=512",
        "M2": "Multi-catalog, large, L=512", "M3": "California only, large, L=512",
        "M4": "Multi-catalog, small, L=4096", "M5": "Strict multi-catalog (pre-2007), small"}
TOHUM = [1, 2, 3]
N_BOOT = 2000

v = np.load(KOK / "veri/olaylar.npz")
t, m, bolum, etas = v["t_gun"], v["buyukluk"], v["bolum"], v["etas_tll"]
test = np.where(bolum == 3)[0]
tt = t[test]
etas_t = etas[test]
kume = {"tum_test": np.ones(len(test), bool), "yogun_M6_30g": v["yogun_M6_30g"][test],
        "yogun_M6_1g": v["yogun_M6_1g"][test]}
kume["sakin"] = ~kume["yogun_M6_30g"]
kume["El Mayor 2010 M7.2"] = v["dizi::2010-04-04 M7.2"][test]
kume["Ridgecrest 2019 M7.1"] = v["dizi::2019-07-06 M7.1"][test]

kos, eksik = {}, []
for k in KOD:
    for s in TOHUM:
        d = KOK / "sonuclar" / f"c2_{k}_t{s}"
        if not (d / "test.npz").exists():
            eksik.append(d.name)
            continue
        z = np.load(d / "test.npz")
        assert np.array_equal(z["idx"], test)
        kos[(k, s)] = dict(tll=z["tll"], mll=z["mll"], kayit=json.loads((d / "kayit.json").read_text(encoding="utf-8")))
print("eksik:", eksik)
tam = [k for k in KOD if all((k, s) in kos for s in TOHUM)]
rng = np.random.default_rng(0)


def blok_boot(fark, mk, blok_gun=30):
    f, b = fark[mk], np.floor((tt[mk] - tt[0]) / blok_gun).astype(int)
    ub, ters = np.unique(b, return_inverse=True)
    top, say = np.bincount(ters, weights=f), np.bincount(ters)
    sec = rng.integers(0, len(ub), (N_BOOT, len(ub)))
    o = top[sec].sum(1) / say[sec].sum(1)
    return [float(np.quantile(o, .025)), float(np.quantile(o, .975)), int(len(ub))]


def ort(k):
    return np.mean([kos[(k, s)]["tll"] for s in TOHUM], 0)


ozet = {"eksik": eksik, "kume": {}, "fark": {}, "kosular": {}}
for ad, mk in kume.items():
    r = {"n": int(mk.sum()), "ETAS": float(etas_t[mk].mean())}
    for k in tam:
        x = [float(kos[(k, s)]["tll"][mk].mean()) for s in TOHUM]
        f = ort(k) - etas_t
        r[k] = {"ort": float(np.mean(x)), "ss": float(np.std(x, ddof=1)), "tohum": x,
                "eksi_ETAS": float(f[mk].mean()), "eksi_ETAS_GA30": blok_boot(f, mk),
                "eksi_ETAS_GA1": blok_boot(f, mk, 1),
                "mll": float(np.mean([kos[(k, s)]["mll"][mk].mean() for s in TOHUM]))}
    ozet["kume"][ad] = r

KARSI = [("M1", "M0"), ("M2", "M3"), ("M2", "M0"), ("M3", "M0"), ("M4", "M1"), ("M5", "M0"), ("M1", "M5")]
for a, b in KARSI:
    if a not in tam or b not in tam:
        continue
    for ad, mk in kume.items():
        f = ort(a) - ort(b)
        ozet["fark"][f"{a}-{b}|{ad}"] = {
            "ort": float(f[mk].mean()), "GA30": blok_boot(f, mk), "GA1": blok_boot(f, mk, 1),
            "tohum_eslesik": [float((kos[(a, s)]["tll"] - kos[(b, s)]["tll"])[mk].mean()) for s in TOHUM]}
for (k, s), x in kos.items():
    ozet["kosular"][f"{k}_t{s}"] = {q: x["kayit"][q] for q in
                                    ["en_iyi_adim", "en_iyi_dogrulama_tll", "test_tll", "test_mll", "sure_s", "n_param", "egitim_hedef"]}

# ---------- karar (ON_KAYIT_2.md) ----------
karar = "eksik"
if {"M0", "M1", "M2", "M3"} <= set(tam):
    d10 = ozet["fark"]["M1-M0|tum_test"]
    d23 = ozet["fark"]["M2-M3|tum_test"]
    etas_gecer = any(ozet["kume"]["tum_test"][k]["eksi_ETAS_GA30"][0] > 0 for k in ("M1", "M2"))
    devam = d10["ort"] >= 0.03 and d10["GA30"][0] > 0 and all(x > 0 for x in d10["tohum_eslesik"]) and etas_gecer
    dur = (d10["ort"] < 0.01 or d10["GA30"][0] <= 0) and d23["ort"] < 0.01
    karar = "DEVAM" if devam else ("DUR" if dur else "BELIRSIZ")
    ozet["karar"] = {"karar": karar, "M1-M0": d10, "M2-M3": d23, "ETAS_GA_disliyor_M1_veya_M2": etas_gecer}
with open(KOK / "sonuclar/c2_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)


def ga(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


sat = ["| Koşu | " + " | ".join(kume) + " |", "|---|" + "---:|" * len(kume)]
sat.append("| ETAS | " + " | ".join(f"{ozet['kume'][a]['ETAS']:.3f}" for a in kume) + " |")
for k in tam:
    sat.append(f"| {k} | " + " | ".join(f"{ozet['kume'][a][k]['ort']:.3f} ± {ozet['kume'][a][k]['ss']:.3f}" for a in kume) + " |")
sat += ["", "| Koşu − ETAS | " + " | ".join(kume) + " |", "|---|" + "---|" * len(kume)]
for k in tam:
    sat.append(f"| {k} | " + " | ".join(
        f"{ozet['kume'][a][k]['eksi_ETAS']:+.3f} {ga(ozet['kume'][a][k]['eksi_ETAS_GA30' if a in ('tum_test', 'sakin', 'yogun_M6_30g', 'yogun_M6_1g') else 'eksi_ETAS_GA1'])}"
        for a in kume) + " |")
sat += ["", "| Fark | Küme | Ortalama | %95 GA (30 g blok) | %95 GA (1 g blok) | Tohum eşleşik |", "|---|---|---:|---|---|---|"]
for anahtar, x in ozet["fark"].items():
    a, b = anahtar.split("|")
    sat.append(f"| {a} | {b} | {x['ort']:+.4f} | {ga(x['GA30'])} | {ga(x['GA1'])} | "
               + ", ".join(f"{y:+.3f}" for y in x["tohum_eslesik"]) + " |")
sat += ["", "| Koşu | Büyüklük LL (tüm test) |", "|---|---:|"]
for k in tam:
    sat.append(f"| {k} | {ozet['kume']['tum_test'][k]['mll']:.4f} |")
sat += ["", f"**Karar (ön kayıtlı kural): {karar}**"]
(KOK / "sonuclar/c2_tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))

# ---------- şekiller ----------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "savefig.dpi": 300})
KOYU, ORTA = "#404040", "#8c8c8c"
if len(tam) == len(KOD):
    paneller = [("tum_test", "All test events"), ("yogun_M6_30g", "30 d after M≥6"), ("sakin", "Outside sequences")]
    fig, eks = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    y = np.arange(len(KOD))[::-1]
    for j, (ax, (ad, bas)) in enumerate(zip(eks, paneller)):
        for yy, k in zip(y, KOD):
            x = ozet["kume"][ad][k]
            lo, hi = x["eksi_ETAS_GA30"][:2]
            anl = lo > 0 or hi < 0
            ax.barh(yy, x["eksi_ETAS"], height=0.6, color=KOYU if anl else "white", edgecolor="black",
                    hatch=None if anl else "////")
            ax.errorbar(x["eksi_ETAS"], yy, xerr=[[x["eksi_ETAS"] - lo], [hi - x["eksi_ETAS"]]], color="black", capsize=3, lw=1)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"({'abc'[j]}) {bas}", fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{k}: {ACIK[k]}" for k in KOD], fontsize=8)
    fig.supxlabel("Temporal log-likelihood minus ETAS (nats per event)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    for uz in ("png", "pdf"):
        fig.savefig(KOK / f"sekiller/c2_sekil1_etas_farki.{uz}")
    plt.close(fig)

    yil = 1971 + tt / 365.25
    fig, ax = plt.subplots(figsize=(7, 3.4))
    stil = {"M0": dict(color=ORTA, ls="-"), "M1": dict(color="black", ls="-"), "M2": dict(color=KOYU, ls="--")}
    for k, st in stil.items():
        ax.plot(yil, np.cumsum(ort(k) - etas_t), lw=1.2, label=f"{k}: {ACIK[k]}", **st)
    for i in np.where(m[test] >= 7.0)[0]:
        ax.axvline(yil[i], color=ORTA, lw=0.6, ls=":")
        ax.text(yil[i], 0.97, f"M{m[test][i]:.1f} ", va="top", ha="right", fontsize=8, transform=ax.get_xaxis_transform())
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative gain over ETAS (nats)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    for uz in ("png", "pdf"):
        fig.savefig(KOK / f"sekiller/c2_sekil2_kumulatif.{uz}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    stil3 = {"M0": dict(color=ORTA, ls="-"), "M1": dict(color="black", ls="-"), "M2": dict(color=KOYU, ls="--"),
             "M3": dict(color=ORTA, ls="--")}
    for k, st in stil3.items():
        for s in TOHUM:
            d = np.array(kos[(k, s)]["kayit"]["dogrulama"])
            ax.plot(d[:, 0], d[:, 1], lw=1, label=k if s == 1 else None, **st)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation temporal LL")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
    fig.tight_layout()
    for uz in ("png", "pdf"):
        fig.savefig(KOK / f"sekiller/c2_sekil3_dogrulama.{uz}")
    plt.close(fig)
