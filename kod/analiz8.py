"""Mimari testi (ON_KAYIT_8.md): sonuclar/c8_ozet.json, c8_tablolar.md, sekiller/c8_sekil1_mimari.

Mac'te: /usr/bin/python3 kod/analiz8.py
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
VAR = ["A0", "A1", "A2", "A3", "A4"]
ACIK = {"A0": "Baseline", "A1": "Catalog embedding", "A2": "Location input", "A3": "Synthetic ETAS pretraining",
        "A4": "All three"}
TEST = {"TR": dict(npz="turkiye.npz", eski="c3_T1", ad="Kahramanmaraş", etas="etas6_E7"),
        "JP": dict(npz="japonya.npz", eski="c7_J1", ad="Noto", etas="etas7_E5")}
ozet = {"kosu": {}, "fark": {}, "denetim": {}, "karar": {}}
veri = {}
for te, c in TEST.items():
    v = np.load(KOK / "veri" / c["npz"])
    test = np.where(v["bolum"] == 3)[0]
    tt = v["t_gun"][test]
    ana = int(v["ana_sok_idx"])
    s = tt - v["t_gun"][ana]
    kume = {"tum_yil": (np.ones(len(test), bool), 7), "ilk_gun": ((s > 0) & (s <= 1), 1 / 24)}
    kos = {}
    for var in VAR:
        for sd in (1, 2, 3):
            d = KOK / f"sonuclar/c8_{te}_{var}_t{sd}"
            z = np.load(d / "test.npz")
            assert np.array_equal(z["idx"], test)
            k = json.loads((d / "kayit.json").read_text(encoding="utf-8"))
            kos[(var, sd)] = z["tll"]
            ozet["kosu"][f"{te}_{var}_t{sd}"] = {"test_tll": float(z["tll"].mean()), "dogrulama_tll": k["en_iyi_dogrulama_tll"],
                                                 "en_iyi_adim": k["en_iyi_adim"], "n_param": k["n_param"]}
    eski = [np.load(KOK / f"sonuclar/{c['eski']}_t{sd}/test.npz")["tll"] for sd in (1, 2, 3)]
    ozet["denetim"][te] = [float(np.abs(kos[("A0", sd)] - eski[sd - 1]).max()) for sd in (1, 2, 3)]
    etas = np.load(KOK / f"sonuclar/{c['etas']}/etas_test.npz")["tll"]
    veri[te] = dict(tt=tt, kume=kume, kos=kos, etas=etas)

    def boot(f, mk, bg):
        f2, bb = f[mk], np.floor((tt[mk] - tt[0]) / bg).astype(int)
        ub, ters = np.unique(bb, return_inverse=True)
        top, say = np.bincount(ters, weights=f2), np.bincount(ters)
        sec = rng.integers(0, len(ub), (N_BOOT, len(ub)))
        return [float(x) for x in np.quantile(top[sec].sum(1) / say[sec].sum(1), [.025, .975])]

    for var in VAR:
        ort = np.mean([kos[(var, sd)] for sd in (1, 2, 3)], 0)
        for ad, (mk, bg) in kume.items():
            r = {"tll": float(ort[mk].mean()), "eksi_ETAS": float((ort - etas)[mk].mean())}
            if var != "A0":
                f = ort - np.mean([kos[("A0", sd)] for sd in (1, 2, 3)], 0)
                r.update({"eksi_A0": float(f[mk].mean()), "GA": boot(f, mk, bg),
                          "tohum": [float((kos[(var, sd)] - kos[("A0", sd)])[mk].mean()) for sd in (1, 2, 3)]})
            r["dogrulama_eksi_A0"] = float(np.mean([ozet["kosu"][f"{te}_{var}_t{sd}"]["dogrulama_tll"] - ozet["kosu"][f"{te}_A0_t{sd}"]["dogrulama_tll"] for sd in (1, 2, 3)]))
            ozet["fark"][f"{te}|{var}|{ad}"] = r

for var in VAR[1:]:
    yil = [ozet["fark"][f"{te}|{var}|tum_yil"] for te in TEST]
    tohum = sum((r["tohum"] for r in yil), [])
    if all(r["eksi_A0"] >= 0.01 for r in yil) and any(r["GA"][0] > 0 for r in yil) and sum(x > 0 for x in tohum) >= 5:
        k = "YARARLI"
    elif all(r["eksi_A0"] <= 0 for r in yil):
        k = "YARARSIZ"
    else:
        k = "BELIRSIZ"
    ozet["karar"][var] = k
with open(KOK / "sonuclar/c8_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)


def ga(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


sat = ["Denetim (A0 ile üçüncü/yedinci testteki T1/J1, olay başına en büyük fark): "
       + "; ".join(f"{te}: " + ", ".join(f"{x:.1e}" for x in v) for te, v in ozet["denetim"].items()), ""]
for te, c in TEST.items():
    sat += [f"### {te} ({c['ad']})", "", "| Varyant | Test TLL (yıl) | − A0 (yıl, 7 g GA) | tohumlar | İlk gün TLL | − A0 (ilk gün, 1 sa GA) | tohumlar | − ETAS ({}) yıl | Doğrulama − A0 |".format(c["etas"]),
            "|---|---:|---|---|---:|---|---|---:|---:|"]
    for var in VAR:
        y, g = ozet["fark"][f"{te}|{var}|tum_yil"], ozet["fark"][f"{te}|{var}|ilk_gun"]
        if var == "A0":
            sat.append(f"| A0 {ACIK[var]} | {y['tll']:.4f} | | | {g['tll']:.3f} | | | {y['eksi_ETAS']:+.3f} | |")
        else:
            sat.append(f"| {var} {ACIK[var]} | {y['tll']:.4f} | {y['eksi_A0']:+.4f} {ga(y['GA'])} | " + ", ".join(f"{x:+.3f}" for x in y["tohum"])
                       + f" | {g['tll']:.3f} | {g['eksi_A0']:+.3f} {ga(g['GA'])} | " + ", ".join(f"{x:+.3f}" for x in g["tohum"])
                       + f" | {y['eksi_ETAS']:+.3f} | {y['dogrulama_eksi_A0']:+.4f} |")
    sat.append("")
sat += ["Koşu ayrıntıları (test TLL / doğrulama TLL / en iyi adım):", ""]
for te in TEST:
    for var in VAR:
        sat.append(f"- {te} {var}: " + "; ".join(
            f"{ozet['kosu'][f'{te}_{var}_t{sd}']['test_tll']:.4f} / {ozet['kosu'][f'{te}_{var}_t{sd}']['dogrulama_tll']:.4f} / {ozet['kosu'][f'{te}_{var}_t{sd}']['en_iyi_adim']}" for sd in (1, 2, 3)))
sat += ["", "**Karar (ön kayıtlı kural):** " + ", ".join(f"{v} {ACIK[v]}: {k}" for v, k in ozet["karar"].items())]
(KOK / "sonuclar/c8_tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "savefig.dpi": 300})
KOYU = "#404040"
MAVI, MAVI_ACIK, MACENTA, MACENTA_ACIK, TURUNCU, KIRMIZI = "#1f3fcf", "#b9c6f2", "#d61fd0", "#f7c2f3", "#ef8f3a", "#e0342b"
fig, eks = plt.subplots(2, 2, figsize=(9, 4.6), sharey=True)
y = np.arange(4)[::-1]
for i, te in enumerate(TEST):
    for j, (ad, bas) in enumerate([("tum_yil", "full test year"), ("ilk_gun", "first day after mainshock")]):
        ax = eks[i, j]
        for yy, var in zip(y, VAR[1:]):
            r = ozet["fark"][f"{te}|{var}|{ad}"]
            lo, hi = r["GA"]
            anl = lo > 0 or hi < 0
            ax.barh(yy, r["eksi_A0"], height=0.6, color=MAVI if anl else "white", edgecolor="black" if anl else MAVI, hatch=None if anl else "////")
            ax.errorbar(r["eksi_A0"], yy, xerr=[[r["eksi_A0"] - lo], [hi - r["eksi_A0"]]], color="black", capsize=3, lw=1)
        ax.axvline(0, color=KIRMIZI, lw=1.0, ls="--")
        ax.set_title(f"({'abcd'[2 * i + j]}) {TEST[te]['ad']}, {bas}", fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{v}: {ACIK[v]}" for v in VAR[1:]], fontsize=8)
fig.supxlabel("Temporal log-likelihood minus baseline A0 (nats per event)", fontweight="bold", fontsize=10)
fig.tight_layout()
for uz in ("png", "pdf"):
    fig.savefig(KOK / f"sekiller/c8_sekil1_mimari.{uz}")
plt.close(fig)
