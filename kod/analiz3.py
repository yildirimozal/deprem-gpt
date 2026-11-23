"""Kahramanmaraş testi (ON_KAYIT_3.md) sonuçları: sonuclar/c3_ozet.json, c3_tablolar.md, sekiller/c3_*.

Mac'te: /usr/bin/python3 kod/analiz3.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
KOD = ["T0", "T1", "T2"]
ACIK = {"T0": "Turkey only", "T1": "Turkey + other catalogs", "T2": "Other catalogs only (zero-shot)"}
TOHUM = [1, 2, 3]
N_BOOT = 2000
BLOK = 7          # gün, ön kayıtlı

v = np.load(KOK / "veri/turkiye.npz")
t, m, bolum = v["t_gun"], v["buyukluk"], v["bolum"]
test = np.where(bolum == 3)[0]
tt = t[test]
ana = int(v["ana_sok_idx"])
e = np.load(KOK / "sonuclar/etas_tr/etas_test.npz")
assert np.array_equal(e["idx"], test)
etas_t = e["tll"]
kume = {"tum_test": np.ones(len(test), bool)}
for k in ["sonra_1g", "sonra_7g", "sonra_30g", "dizi_30g", "dizi_disi"]:
    kume[k] = v[k][test]
kume["30g_sonrasi"] = (tt > t[ana] + 30)

kos, eksik = {}, []
for k in KOD:
    for s in TOHUM:
        d = KOK / "sonuclar" / f"c3_{k}_t{s}"
        if not (d / "test.npz").exists():
            eksik.append(d.name)
            continue
        z = np.load(d / "test.npz")
        assert np.array_equal(z["idx"], test)
        kos[(k, s)] = dict(tll=z["tll"], mll=z["mll"], kayit=json.loads((d / "kayit.json").read_text(encoding="utf-8")))
print("eksik:", eksik)
tam = [k for k in KOD if all((k, s) in kos for s in TOHUM)]
rng = np.random.default_rng(0)


def blok_boot(fark, mk, blok_gun=BLOK):
    f, b = fark[mk], np.floor((tt[mk] - tt[0]) / blok_gun).astype(int)
    ub, ters = np.unique(b, return_inverse=True)
    top, say = np.bincount(ters, weights=f), np.bincount(ters)
    sec = rng.integers(0, len(ub), (N_BOOT, len(ub)))
    o = top[sec].sum(1) / say[sec].sum(1)
    return [float(np.quantile(o, .025)), float(np.quantile(o, .975)), int(len(ub))]


def ort(k):
    return np.mean([kos[(k, s)]["tll"] for s in TOHUM], 0)


ozet = {"eksik": eksik, "blok_gun": BLOK, "kume": {}, "fark": {}, "kosular": {}}
for ad, mk in kume.items():
    r = {"n": int(mk.sum()), "ETAS": float(etas_t[mk].mean())}
    for k in tam:
        x = [float(kos[(k, s)]["tll"][mk].mean()) for s in TOHUM]
        f = ort(k) - etas_t
        r[k] = {"ort": float(np.mean(x)), "ss": float(np.std(x, ddof=1)), "tohum": x,
                "eksi_ETAS": float(f[mk].mean()), "eksi_ETAS_GA": blok_boot(f, mk), "eksi_ETAS_GA1": blok_boot(f, mk, 1),
                "tohum_eksi_ETAS": [float((kos[(k, s)]["tll"] - etas_t)[mk].mean()) for s in TOHUM],
                "mll": float(np.mean([kos[(k, s)]["mll"][mk].mean() for s in TOHUM]))}
    ozet["kume"][ad] = r

for a, b in [("T1", "T0"), ("T2", "T0"), ("T1", "T2")]:
    if a not in tam or b not in tam:
        continue
    for ad, mk in kume.items():
        f = ort(a) - ort(b)
        ozet["fark"][f"{a}-{b}|{ad}"] = {
            "ort": float(f[mk].mean()), "GA": blok_boot(f, mk), "GA1": blok_boot(f, mk, 1),
            "tohum_eslesik": [float((kos[(a, s)]["tll"] - kos[(b, s)]["tll"])[mk].mean()) for s in TOHUM]}
for (k, s), x in kos.items():
    ozet["kosular"][f"{k}_t{s}"] = {q: x["kayit"][q] for q in
                                    ["en_iyi_adim", "en_iyi_dogrulama_tll", "test_tll", "test_mll", "sure_s", "n_param", "egitim_hedef"]}

# ---------- karar (ON_KAYIT_3.md) ----------
karar = "eksik"
if {"T0", "T1"} <= set(tam):
    k1 = ozet["kume"]["tum_test"]["T1"]
    k30 = ozet["kume"]["dizi_30g"]["T1"]
    d10 = ozet["fark"]["T1-T0|tum_test"]
    devam = (k1["eksi_ETAS_GA"][0] > 0 and k30["eksi_ETAS"] > 0
             and d10["ort"] > 0 and all(x > 0 for x in d10["tohum_eslesik"]))
    dur = k1["eksi_ETAS"] <= 0
    karar = "DEVAM" if devam else ("DUR" if dur else "BELIRSIZ")
    ozet["karar"] = {"karar": karar, "T1-ETAS": k1["eksi_ETAS"], "T1-ETAS_GA": k1["eksi_ETAS_GA"],
                     "T1-ETAS_dizi_30g": k30["eksi_ETAS"], "T1-T0": d10}
with open(KOK / "sonuclar/c3_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)


def ga(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


sat = ["| Koşu | " + " | ".join(f"{a} (n={ozet['kume'][a]['n']})" for a in kume) + " |", "|---|" + "---:|" * len(kume)]
sat.append("| ETAS | " + " | ".join(f"{ozet['kume'][a]['ETAS']:.3f}" for a in kume) + " |")
for k in tam:
    sat.append(f"| {k} | " + " | ".join(f"{ozet['kume'][a][k]['ort']:.3f} ± {ozet['kume'][a][k]['ss']:.3f}" for a in kume) + " |")
sat += ["", f"| Koşu − ETAS ({BLOK} g blok GA; 1 g blok GA) | " + " | ".join(kume) + " |", "|---|" + "---|" * len(kume)]
for k in tam:
    sat.append(f"| {k} | " + " | ".join(
        f"{ozet['kume'][a][k]['eksi_ETAS']:+.3f} {ga(ozet['kume'][a][k]['eksi_ETAS_GA'])}; {ga(ozet['kume'][a][k]['eksi_ETAS_GA1'])}"
        for a in kume) + " |")
sat += ["", "| Koşu − ETAS, tohum tohum (tüm test) |", "|---|"]
for k in tam:
    sat.append(f"| {k}: " + ", ".join(f"{y:+.3f}" for y in ozet["kume"]["tum_test"][k]["tohum_eksi_ETAS"]) + " |")
sat += ["", f"| Fark | Küme | Ortalama | %95 GA ({BLOK} g blok) | %95 GA (1 g blok) | Tohum eşleşik |", "|---|---|---:|---|---|---|"]
for anahtar, x in ozet["fark"].items():
    a, b = anahtar.split("|")
    sat.append(f"| {a} | {b} | {x['ort']:+.4f} | {ga(x['GA'])} | {ga(x['GA1'])} | "
               + ", ".join(f"{y:+.3f}" for y in x["tohum_eslesik"]) + " |")
sat += ["", "| Koşu | Büyüklük LL (tüm test) |", "|---|---:|"]
for k in tam:
    sat.append(f"| {k} | {ozet['kume']['tum_test'][k]['mll']:.4f} |")
sat += ["", f"**Karar (ön kayıtlı kural): {karar}**"]
(KOK / "sonuclar/c3_tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))

# ---------- şekiller ----------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "savefig.dpi": 300})
KOYU, ORTA = "#404040", "#8c8c8c"
if len(tam) == len(KOD):
    paneller = [("tum_test", "Full test year"), ("dizi_30g", "Sequence box, first 30 d"),
                ("30g_sonrasi", "Later than 30 d"), ("dizi_disi", "Outside sequence box")]
    fig, eks = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
    y = np.arange(len(KOD))[::-1]
    for j, (ax, (ad, bas)) in enumerate(zip(eks, paneller)):
        for yy, k in zip(y, KOD):
            x = ozet["kume"][ad][k]
            lo, hi = x["eksi_ETAS_GA1" if ad == "dizi_30g" else "eksi_ETAS_GA"][:2]   # 30 günlük pencerede 7 g blok çok az
            anl = lo > 0 or hi < 0
            ax.barh(yy, x["eksi_ETAS"], height=0.6, color=KOYU if anl else "white", edgecolor="black",
                    hatch=None if anl else "////")
            ax.errorbar(x["eksi_ETAS"], yy, xerr=[[x["eksi_ETAS"] - lo], [hi - x["eksi_ETAS"]]], color="black", capsize=3, lw=1)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"({'abcd'[j]}) {bas}", fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{k}: {ACIK[k]}" for k in KOD], fontsize=8)
    fig.supxlabel("Temporal log-likelihood minus ETAS (nats per event)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    for uz in ("png", "pdf"):
        fig.savefig(KOK / f"sekiller/c3_sekil1_etas_farki.{uz}")
    plt.close(fig)

    gun_sonra = tt - t[ana]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    stil = {"T0": dict(color=ORTA, ls="-"), "T1": dict(color="black", ls="-"), "T2": dict(color=KOYU, ls="--")}
    for k, st in stil.items():
        ax.plot(gun_sonra, np.cumsum(ort(k) - etas_t), lw=1.2, label=f"{k}: {ACIK[k]}", **st)
    for i in np.where(m[test] >= 7.0)[0]:
        ax.axvline(gun_sonra[i], color=ORTA, lw=0.6, ls=":")
        ax.text(gun_sonra[i], 0.97, f" M{m[test][i]:.1f}", va="top", ha="left", fontsize=8, transform=ax.get_xaxis_transform())
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(gun_sonra[0] - 0.01, gun_sonra[-1] * 1.05)
    ax.set_xlabel("Days since the M7.8 mainshock")
    ax.set_ylabel("Cumulative gain over ETAS (nats)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    for uz in ("png", "pdf"):
        fig.savefig(KOK / f"sekiller/c3_sekil2_kumulatif.{uz}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for k, st in stil.items():
        for s in TOHUM:
            d = np.array(kos[(k, s)]["kayit"]["dogrulama"])
            ax.plot(d[:, 0], d[:, 1], lw=1, label=k if s == 1 else None, **st)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation temporal LL (Turkey)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout()
    for uz in ("png", "pdf"):
        fig.savefig(KOK / f"sekiller/c3_sekil3_dogrulama.{uz}")
    plt.close(fig)
