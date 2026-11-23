"""Sonuçları toplar: tablolar (sonuclar/ozet.json, sonuclar/tablolar.md) ve şekiller (sekiller/).

Mac'te: /usr/bin/python3 kod/analiz.py [önek]
önek: "" (ana koşular, L{L}_t{tohum}) ya da "tam_" (eğitim+doğrulama ile yeniden eğitim)
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
ONEK = sys.argv[1] if len(sys.argv) > 1 else ""
LER = [20, 512, 4096]
TOHUM = [1, 2, 3]
BETA = 2.1471442086213064            # ETAS parametre dosyasındaki b-değeri (doğal log tabanı)
POISSON_TLL = 0.5126406686259881      # ll_scores.json
N_BOOT = 2000

v = np.load(KOK / "veri/olaylar.npz")
t, m, bolum, etas = v["t_gun"], v["buyukluk"], v["bolum"], v["etas_tll"]
test = np.where(bolum == 3)[0]
tt = t[test]
maskeler = {"tum_test": np.ones(len(test), bool)}
for k in ["yogun_M6_30g", "yogun_M7_30g", "yogun_M6_7g", "yogun_M6_1g"]:
    maskeler[k] = v[k][test]
diziler = {k.split("::")[1]: v[k][test] for k in v.files if k.startswith("dizi::")}
maskeler["sakin"] = ~maskeler["yogun_M6_30g"]
etas_t = etas[test]
gr_mll = -BETA * (m[test] - 2.5) + np.log1p(-np.exp(-BETA * 0.1))

# ---------- koşuları oku ----------
kos = {}
eksik = []
for L in LER:
    for s in TOHUM:
        d = KOK / "sonuclar" / f"{ONEK}L{L}_t{s}"
        if not (d / "test.npz").exists():
            eksik.append(d.name)
            continue
        z = np.load(d / "test.npz")
        assert np.array_equal(z["idx"], test)
        kos[(L, s)] = dict(tll=z["tll"], mll=z["mll"], kayit=json.loads((d / "kayit.json").read_text(encoding="utf-8")))
print("eksik koşular:", eksik)


def blok_boot(fark, maske, blok_gun, rng):
    """Olay farklarının ortalaması için blok bootstrap %95 GA (bloklar zaman ekseninde)."""
    f, b = fark[maske], np.floor((tt[maske] - tt[0]) / blok_gun).astype(int)
    ub, ters = np.unique(b, return_inverse=True)
    top = np.bincount(ters, weights=f)
    say = np.bincount(ters)
    sec = rng.integers(0, len(ub), (N_BOOT, len(ub)))
    ort = top[sec].sum(1) / say[sec].sum(1)
    return float(np.quantile(ort, 0.025)), float(np.quantile(ort, 0.975)), int(len(ub))


rng = np.random.default_rng(0)
ozet = {"onek": ONEK, "eksik": eksik, "n_test": int(len(test)), "kume": {}}
for ad, mk in {**maskeler, **{"dizi " + k: v_ for k, v_ in diziler.items()}}.items():
    satir = {"n": int(mk.sum()), "ETAS": float(etas_t[mk].mean()),
             "GR_mll": float(gr_mll[mk].mean())}
    if ad == "tum_test":
        satir["Poisson"] = POISSON_TLL
    for L in LER:
        ss = [s for s in TOHUM if (L, s) in kos]
        if not ss:
            continue
        tohum_ort = [float(kos[(L, s)]["tll"][mk].mean()) for s in ss]
        tll_ort = np.mean([kos[(L, s)]["tll"] for s in ss], 0)        # olay başına tohum ortalaması
        mll_ort = [float(kos[(L, s)]["mll"][mk].mean()) for s in ss]
        fark = tll_ort - etas_t
        r = {"tohum": tohum_ort, "ort": float(np.mean(tohum_ort)),
             "ss": float(np.std(tohum_ort, ddof=1)) if len(ss) > 1 else None,
             "fark_ETAS": float(fark[mk].mean()),
             "fark_ETAS_GA30": blok_boot(fark, mk, 30, rng),
             "fark_ETAS_GA1": blok_boot(fark, mk, 1, rng),
             "mll_ort": float(np.mean(mll_ort)), "mll_tohum": mll_ort}
        satir[f"L{L}"] = r
    if all((L, s) in kos for L in (20, 4096) for s in TOHUM):
        f = np.mean([kos[(4096, s)]["tll"] for s in TOHUM], 0) - np.mean([kos[(20, s)]["tll"] for s in TOHUM], 0)
        satir["L4096_eksi_L20"] = {"ort": float(f[mk].mean()), "GA30": blok_boot(f, mk, 30, rng),
                                   "GA1": blok_boot(f, mk, 1, rng),
                                   "tohum_eslesik": [float((kos[(4096, s)]["tll"] - kos[(20, s)]["tll"])[mk].mean())
                                                     for s in TOHUM]}
    ozet["kume"][ad] = satir

ozet["kosular"] = {f"L{L}_t{s}": {k: kos[(L, s)]["kayit"][k] for k in
                                  ["en_iyi_adim", "en_iyi_dogrulama_tll", "test_tll", "test_mll", "sure_s", "n_param"]}
                   for (L, s) in kos}
(KOK / "sonuclar").mkdir(exist_ok=True)
with open(KOK / f"sonuclar/{ONEK}ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)


# ---------- markdown tablo ----------
def ga(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


sat = ["| Küme | n | ETAS | L=20 | L=512 | L=4096 | 4096 − 20 (%95 GA, 30 g blok) |", "|---|---:|---:|---:|---:|---:|---|"]
for ad, r in ozet["kume"].items():
    hucre = []
    for L in LER:
        x = r.get(f"L{L}")
        hucre.append("–" if x is None else f"{x['ort']:.3f} ± {x['ss']:.3f}" if x["ss"] is not None else f"{x['ort']:.3f}")
    d = r.get("L4096_eksi_L20")
    sat.append(f"| {ad} | {r['n']} | {r['ETAS']:.3f} | " + " | ".join(hucre) +
               (f" | {d['ort']:+.3f} {ga(d['GA30'])} |" if d else " | – |"))
sat += ["", "| Küme | 30 g blok sayısı | L=20 − ETAS | L=512 − ETAS | L=4096 − ETAS | L=4096 − ETAS (1 g blok GA) |", "|---|---:|---|---|---|---|"]
for ad, r in ozet["kume"].items():
    if r.get("L4096") is None:
        continue
    sat.append(f"| {ad} | {r['L4096']['fark_ETAS_GA30'][2]} | " + " | ".join(
        "–" if r.get(f"L{L}") is None else f"{r[f'L{L}']['fark_ETAS']:+.3f} {ga(r[f'L{L}']['fark_ETAS_GA30'])}"
        for L in LER) + f" | {ga(r['L4096']['fark_ETAS_GA1'])} |")
sat += ["", "| Küme | GR (β ETAS) | L=20 | L=512 | L=4096 |", "|---|---:|---:|---:|---:|"]
for ad in ["tum_test", "yogun_M6_30g", "sakin"]:
    r = ozet["kume"][ad]
    sat.append(f"| {ad} | {r['GR_mll']:.4f} | " + " | ".join(
        "–" if r.get(f"L{L}") is None else f"{r[f'L{L}']['mll_ort']:.4f}" for L in LER) + " |")
(KOK / f"sonuclar/{ONEK}tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))

# ---------- şekiller (dergi stili) ----------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "savefig.dpi": 300,
})
KOYU, ORTA = "#404040", "#8c8c8c"
(KOK / "sekiller").mkdir(exist_ok=True)

if len(kos) == 9:
    # Şekil 1: ETAS'a göre fark, panelli yatay çubuk + %95 GA
    kumeler = [("tum_test", "All test events"), ("sakin", "Outside sequences"),
               ("yogun_M6_30g", "30 d after M≥6"), ("yogun_M6_1g", "1 d after M≥6")]
    fig, eks = plt.subplots(1, len(kumeler), figsize=(10, 2.6), sharey=True)
    for j, (eks_, (ad, baslik)) in enumerate(zip(eks, kumeler)):
        r = ozet["kume"][ad]
        y = np.arange(len(LER))[::-1]
        for yy, L in zip(y, LER):
            x = r[f"L{L}"]
            lo, hi = x["fark_ETAS_GA30"][:2]
            anlamli = lo > 0 or hi < 0
            eks_.barh(yy, x["fark_ETAS"], color=KOYU if anlamli else "white", edgecolor="black",
                      hatch=None if anlamli else "////", height=0.6)
            eks_.errorbar(x["fark_ETAS"], yy, xerr=[[x["fark_ETAS"] - lo], [hi - x["fark_ETAS"]]],
                          color="black", capsize=3, lw=1)
        eks_.axvline(0, color="black", lw=0.8)
        eks_.set_title(f"({'abcd'[j]}) {baslik}", fontsize=10)
        eks_.set_yticks(y)
        eks_.set_yticklabels([f"L = {L}" for L in LER])
    fig.supxlabel("Temporal log-likelihood minus ETAS (nats per event)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(KOK / f"sekiller/{ONEK}sekil1_etas_farki.png")
    fig.savefig(KOK / f"sekiller/{ONEK}sekil1_etas_farki.pdf")
    plt.close(fig)

    # Şekil 2: ETAS'a göre kümülatif bilgi kazancı (tohum ortalaması)
    yil = 1971 + tt / 365.25
    fig, ax = plt.subplots(figsize=(7, 3.4))
    stiller = {20: dict(color=ORTA, ls="-"), 512: dict(color=KOYU, ls="--"), 4096: dict(color="black", ls="-")}
    for L in LER:
        f = np.mean([kos[(L, s)]["tll"] for s in TOHUM], 0) - etas_t
        ax.plot(yil, np.cumsum(f), lw=1.2, label=f"L = {L}", **stiller[L])
    for i in np.where(m[test] >= 7.0)[0]:
        ax.axvline(yil[i], color=ORTA, lw=0.6, ls=":")
        ax.text(yil[i], 0.97, f"M{m[test][i]:.1f} ", va="top", ha="right", fontsize=8,
                transform=ax.get_xaxis_transform())
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative gain over ETAS (nats)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(KOK / f"sekiller/{ONEK}sekil2_kumulatif.png")
    fig.savefig(KOK / f"sekiller/{ONEK}sekil2_kumulatif.pdf")
    plt.close(fig)

    # Şekil 3: doğrulama eğrileri
    if not ONEK:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for L in LER:
            for s in TOHUM:
                d = np.array(kos[(L, s)]["kayit"]["dogrulama"])
                ax.plot(d[:, 0], d[:, 1], lw=1, label=f"L = {L}" if s == 1 else None, **stiller[L])
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation temporal LL")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
        fig.tight_layout()
        fig.savefig(KOK / "sekiller/sekil3_dogrulama.png")
        fig.savefig(KOK / "sekiller/sekil3_dogrulama.pdf")
        plt.close(fig)
