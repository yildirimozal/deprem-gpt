"""GRL makalesi için ana şekiller ve sayılar.

Tam mod (ham veri ve model çıktıları gerekir):
  python kod/makale_sekiller.py
  Çıktı: makale/sekiller/Figure{1,2,3}.{png,pdf}, makale/sayilar.json, makale/sekil_verisi.npz

Özet modu (yalnızca repodaki sonuclar/ dosyalarından şekilleri yeniden çizer):
  python kod/makale_sekiller.py --ozetten
  Çıktı: sekiller/Figure{1,2,3}.{png,pdf} (veya --cikti ile belirtilen dizin)
"""
import argparse
import json
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent
N_BOOT = 2000
KOYU, ORTA = "#404040", "#8c8c8c"
# Renkli palet: model mavi, ETAS macenta, tek katalog turuncu, gözlem siyah, referans çizgisi kırmızı kesikli
MAVI, MAVI_ACIK = "#1f3fcf", "#b9c6f2"
MACENTA, MACENTA_ACIK = "#d61fd0", "#f7c2f3"
TURUNCU = "#ef8f3a"
KIRMIZI = "#e0342b"

TEST_BASLIK = {
    "TR": dict(ad="Kahramanmaraş 2023", ana="M7.8"),
    "JP": dict(ad="Noto 2024", ana="M7.6"),
    "TW": dict(ad="Hualien 2024", ana="M7.2"),
}


def etas_dosya(te, kod):
    if kod.startswith("etas"):
        return KOK / f"sonuclar/{kod}/etas_test.npz"
    onek = {"TR": {"E6": "etas6", "E7": "etas6", "E8": "etas6"}, "JP": {}, "TW": {}}[te].get(
        kod, {"TR": "etas4", "JP": "etas7", "TW": "etas11"}[te]
    )
    return KOK / f"sonuclar/{onek}_{kod}/etas_test.npz"


def _plt_ayarla():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "axes.grid": False, "savefig.dpi": 300,
    })
    return plt


def tam_hesapla():
    """Ham kataloglar ve koşu çıktılarından tüm sayıları ve çizim dizilerini hesaplar."""
    rng = np.random.default_rng(0)
    with open(KOK / "sonuclar/c6_ozet.json", encoding="utf-8") as f:
        etas_tr = json.load(f)["en_guclu_gecerli_tum"]
    with open(KOK / "sonuclar/c7_ozet.json", encoding="utf-8") as f:
        etas_jp = json.load(f)["en_guclu_gecerli"]

    testler = {
        "TR": dict(ad="Kahramanmaraş 2023", npz="turkiye.npz", cok="c3_T1", tek="c3_T0", sifir="c3_T2",
                   etas=etas_tr, kahin="etas4_E4", ana="M7.8"),
        "JP": dict(ad="Noto 2024", npz="japonya.npz", cok="c7_J1", tek="c7_J0", sifir="c7_J2",
                   etas=etas_jp, kahin="etas7_E4", ana="M7.6"),
    }
    c11_yol = KOK / "sonuclar/c11_ozet.json"
    if c11_yol.exists():
        with open(c11_yol, encoding="utf-8") as f:
            o11 = json.load(f)
        testler["TW"] = dict(ad="Hualien 2024", npz="tayvan.npz", cok="c11_H1", tek="c11_H0", sifir="c11_H2",
                             etas=o11["en_guclu_gecerli"], kahin="etas11_E4", ana="M7.2")

    say = {}
    sv = {}
    for te, c in testler.items():
        v = np.load(KOK / "veri" / c["npz"])
        test = np.where(v["bolum"] == 3)[0]
        tt = v["t_gun"][test]
        ana = int(v["ana_sok_idx"])
        gs = tt - v["t_gun"][ana]
        E = np.load(etas_dosya(te, c["etas"]))["tll"]
        Ek = np.load(etas_dosya(te, c["kahin"]))["tll"]
        M = {k: [np.load(KOK / f"sonuclar/{c[k]}_t{s}/test.npz")["tll"] for s in (1, 2, 3)] for k in ("cok", "tek", "sifir")}
        ort = {k: np.mean(x, 0) for k, x in M.items()}

        def boot(f, mk, bg, _rng=rng, _tt=tt):
            f2, bb = f[mk], np.floor((_tt[mk] - _tt[0]) / bg).astype(int)
            ub, ters = np.unique(bb, return_inverse=True)
            top, n_ = np.bincount(ters, weights=f2), np.bincount(ters)
            sec = _rng.integers(0, len(ub), (N_BOOT, len(ub)))
            return [float(q) for q in np.quantile(top[sec].sum(1) / n_[sec].sum(1), [.025, .975])]

        ilk = (gs > 0) & (gs <= 1)
        tum = np.ones(len(test), bool)
        r = {"ad": c["ad"], "n_test": int(len(test)), "n_ilk_gun": int(ilk.sum()), "etas_kodu": c["etas"],
             "mc": float(v["mc"]) if "mc" in v.files else 2.5, "etas_tll": float(E.mean()), "kahin_tll": float(Ek.mean())}
        for k in ("cok", "tek", "sifir"):
            r[f"{k}_tll"] = float(ort[k].mean())
            r[f"{k}_tohum"] = [float(x.mean()) for x in M[k]]
            for ad, mk, bg in [("tum", tum, 7), ("ilk", ilk, 1 / 24)]:
                for ref, R in [("etas", E), ("kahin", Ek)]:
                    f = ort[k] - R
                    r[f"{k}-{ref}_{ad}"] = float(f[mk].mean())
                    r[f"{k}-{ref}_{ad}_GA"] = boot(f, mk, bg)
        f = ort["cok"] - ort["tek"]
        r["cok-tek_tum"] = float(f.mean())
        r["cok-tek_tum_GA"] = boot(f, tum, 7)
        r["cok-tek_tohum"] = [float((a - b).mean()) for a, b in zip(M["cok"], M["tek"])]
        for k in ("cok", "tek", "sifir"):
            r[f"{k}-etas_tohum_tum"] = [float((x - E).mean()) for x in M[k]]
            r[f"{k}-etas_tohum_ilk"] = [float((x - E)[ilk].mean()) for x in M[k]]
        rng2 = np.random.default_rng(1)

        def boot2(f, mk, bg, _rng=rng2, _tt=tt):
            f2, bb = f[mk], np.floor((_tt[mk] - _tt[0]) / bg).astype(int)
            ub, ters = np.unique(bb, return_inverse=True)
            top, n_ = np.bincount(ters, weights=f2), np.bincount(ters)
            sec = _rng.integers(0, len(ub), (N_BOOT, len(ub)))
            return [float(q) for q in np.quantile(top[sec].sum(1) / n_[sec].sum(1), [.025, .975])]

        pencere = {"once": (gs <= 0, 7), "g0_1": (ilk, 1 / 24), "g1_7": ((gs > 1) & (gs <= 7), 1),
                   "g7_30": ((gs > 7) & (gs <= 30), 1), "g30": (gs > 30, 7)}
        r["pencere"] = {}
        for ad, (mk, bg) in pencere.items():
            r["pencere"][ad] = {"n": int(mk.sum())}
            for ref, R in [("etas", E), ("kahin", Ek)]:
                f = ort["cok"] - R
                r["pencere"][ad][ref] = float(f[mk].mean())
                r["pencere"][ad][ref + "_GA"] = boot2(f, mk, bg)
        ai = int(np.where(test == ana)[0][0])
        r["ana_sok_fark_etas"] = float(ort["cok"][ai] - E[ai])
        r["ana_sok_haric_tum"] = float(np.delete(ort["cok"] - E, ai).mean())
        say[te] = r
        sv[f"F1_{te}_gun"] = gs
        for k_, R_ in [("cok_etas", ort["cok"] - E), ("tek_etas", ort["tek"] - E), ("cok_kahin", ort["cok"] - Ek)]:
            sv[f"F1_{te}_{k_}_kumulatif"] = np.cumsum(R_)

    # Figure 2a dizileri
    v = np.load(KOK / "veri/turkiye.npz")
    test = np.where(v["bolum"] == 3)[0]
    saat = (v["t_gun"][test] - v["t_gun"][int(v["ana_sok_idx"])]) * 24
    E5 = np.load(KOK / "sonuclar/c5_etas_ayristirma.npz")
    M5 = np.load(KOK / "sonuclar/c5_model_ayristirma.npz")
    H = {k: np.mean([M5[f"c3_{k}_t{i}_H"] for i in (1, 2, 3)], 0) for k in ("T0", "T1")}
    mk = (saat > 0) & (saat <= 48)
    o = np.argsort(saat[mk])
    x = saat[mk][o]
    sv["F2a_saat"] = x
    sv["F2a_gozlenen"] = np.arange(1, len(x) + 1)
    for k_sv, L in [("etas_altkritik", E5["E2_Lam"]), ("etas_kahin", E5["E4_Lam"]),
                    ("tek", H["T0"]), ("cok", H["T1"])]:
        sv[f"F2a_{k_sv}_beklenen"] = np.cumsum(L[mk][o])

    with open(KOK / "sonuclar/c5_ozet.json", encoding="utf-8") as f:
        c5 = json.load(f)
    with open(KOK / "sonuclar/c9_ozet.json", encoding="utf-8") as f:
        c9 = json.load(f)

    say["mekanizma_TR"] = {"ilk48_gozlenen": int(((saat > 0) & (saat <= 48)).sum()),
                           "ilk48_E2": float(E5["E2_Lam"][mk].sum()), "ilk48_E4": float(E5["E4_Lam"][mk].sum()),
                           "ilk48_T0": float(H["T0"][mk].sum()), "ilk48_T1": float(H["T1"][mk].sum()),
                           "aralik": c5["aralik"]}
    say["olasilik"] = {"skor": c9["skor"], "fark": c9["fark"], "karar": c9["karar"],
                       "kapsama": {te: {"model": r["model_kapsama"], "etas": r["etas_kapsama"]}
                                   for te, r in c9["ana_sok_sonrasi"].items()}}
    return sv, c5, c9, say, list(testler)


def ozetten_yukle():
    """Repoda bulunan özet dosyalarından çizim verilerini yükler."""
    with np.load(KOK / "sonuclar/sekil_verisi.npz") as z:
        sv = {k: z[k] for k in z.files}
    with open(KOK / "sonuclar/c5_ozet.json", encoding="utf-8") as f:
        c5 = json.load(f)
    with open(KOK / "sonuclar/c9_ozet.json", encoding="utf-8") as f:
        c9 = json.load(f)
    aktif = [te for te in ("TR", "JP", "TW") if f"F1_{te}_gun" in sv]
    return sv, c5, c9, aktif


def ciz_figure1(plt, sv, aktif, cik):
    fig, eks = plt.subplots(1, len(aktif), figsize=(3.6 * len(aktif), 3.3), sharey=False)
    eks = np.atleast_1d(eks)
    for j, te in enumerate(aktif):
        c = TEST_BASLIK[te]
        gs = sv[f"F1_{te}_gun"]
        ax = eks[j]
        ax.plot(gs, sv[f"F1_{te}_cok_etas_kumulatif"], color=MAVI, lw=1.5, label="Multi-catalog model")
        ax.plot(gs, sv[f"F1_{te}_tek_etas_kumulatif"], color=TURUNCU, lw=1.4, ls="-.", label="Single-catalog model")
        ax.plot(gs, sv[f"F1_{te}_cok_kahin_kumulatif"], color=MACENTA, lw=1.3, ls="--", label="Multi-catalog model vs. oracle ETAS")
        ax.axhline(0, color=KIRMIZI, lw=1.0, ls="--")
        ax.axvline(0, color="black", lw=0.6, ls=":")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(gs[0] - 0.01, gs[-1] * 1.05)
        ax.set_xlabel(f"Days since the {c['ana']} mainshock")
        if j == 0:
            ax.set_ylabel("Cumulative log-likelihood gain (nats)")
        ax.set_title(f"({'abc'[j]}) {c['ad']}", fontsize=10)
    h, l_ = eks[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for uz in ("png", "pdf"):
        fig.savefig(cik / f"Figure1.{uz}")
    plt.close(fig)


def ciz_figure2(plt, sv, c5, cik):
    fig, eks = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax = eks[0]
    x = sv["F2a_saat"]
    ax.step(x, sv["F2a_gozlenen"], where="post", color="black", lw=1.6, label="Observed")
    for lab, k_sv, st in [
        ("ETAS (subcritical, fitted before 2023)", "etas_altkritik", dict(color=MACENTA, ls="-")),
        ("ETAS (oracle, fitted on test year)", "etas_kahin", dict(color=MACENTA, ls="--")),
        ("Single-catalog model", "tek", dict(color=TURUNCU, ls="-.")),
        ("Multi-catalog model", "cok", dict(color=MAVI, ls="-")),
    ]:
        ax.plot(x, sv[f"F2a_{k_sv}_beklenen"], lw=1.4, label=lab, **st)
    ax.set_xlim(0, 48)
    ax.set_xlabel("Hours since the M7.8 mainshock")
    ax.set_ylabel("Cumulative number of M ≥ 3.0 events")
    ax.set_title("(a) Observed and expected counts", fontsize=10)
    ax = eks[1]
    A = c5["aralik"]
    et = [r["aralik"] for r in A]
    xs = np.arange(len(A))
    for k, (kod, de) in enumerate([("E2", dict(facecolor=MACENTA, hatch="////")), ("E4", dict(facecolor=MACENTA_ACIK, hatch="....")),
                                   ("T0", dict(facecolor=TURUNCU)), ("T1", dict(facecolor=MAVI))]):
        ax.bar(xs + (k - 1.5) * 0.2, [r[f"{kod}_ort_Lambda"] for r in A], width=0.2, edgecolor="black", lw=0.5,
               label={"E2": "ETAS subcritical", "E4": "ETAS oracle", "T0": "Single-catalog", "T1": "Multi-catalog"}[kod], **de)
    ax.axhline(1, color=KIRMIZI, lw=1.0, ls="--")
    ax.set_ylim(0, 1.95)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(et, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean rescaled residual")
    ax.set_title("(b) Rate calibration (ideal = 1)", fontsize=10)
    h1, l1 = eks[0].get_legend_handles_labels()
    fig.legend(h1, l1, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for uz in ("png", "pdf"):
        fig.savefig(cik / f"Figure2.{uz}")
    plt.close(fig)


def ciz_figure3(plt, c9, cik):
    fig = plt.figure(figsize=(10, 6.2))
    gsx = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    for j, (te, ad) in enumerate([(te, ad) for te in ("TR", "JP") for ad in ("M4", "M5")]):
        ax = fig.add_subplot(gsx[0, j])
        ax.plot([0, 1], [0, 1], color=KIRMIZI, lw=1.0, ls="--")
        for kim, st in [("model", dict(color=MAVI, marker="o", ls="-")), ("etas", dict(color=MACENTA, marker="s", ls="--", mfc="white"))]:
            rr = [q for q in c9["kalibrasyon"][f"{te}|{ad}"] if q["kim"] == kim]
            ax.plot([q["ort_p"] for q in rr], [q["gozlenen"] for q in rr], ms=4, lw=1,
                    label="Multi-catalog model" if kim == "model" else "ETAS", **st)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"({'abcd'[j]}) {'Kahramanmaraş' if te == 'TR' else 'Noto'}, M ≥ {ad[1]}", fontsize=9)
        ax.set_xlabel("Forecast probability")
        if j == 0:
            ax.set_ylabel("Observed frequency")
    for j, te in enumerate(("TR", "JP")):
        ax = fig.add_subplot(gsx[1, 2 * j:2 * j + 2])
        r = c9["ana_sok_sonrasi"][te]
        x = np.array(r["saat"])
        ax.fill_between(x, [q[0] for q in r["model_olay_P05_P95"]], [q[1] for q in r["model_olay_P05_P95"]],
                        color=MAVI_ACIK, alpha=0.7, lw=0, label="Model 5–95% range")
        ax.plot(x, r["model_beklenen_olay"], color=MAVI, marker="o", ms=3, lw=1.2, label="Model expected count")
        ax.plot(x, r["etas_beklenen_olay"], color=MACENTA, marker="s", ms=3, lw=1.2, ls="--", mfc="white", label="ETAS expected count")
        ax.plot(x, r["gozlenen_olay"], color="black", ls="none", marker="x", ms=5, label="Observed count")
        ax.set_yscale("log")
        ax.set_xlabel(f"Issue time, hours after the {'M7.8' if te == 'TR' else 'M7.6'} mainshock")
        ax.set_ylabel("M ≥ 3.0 events in next 24 h")
        ax.set_title(f"({'ef'[j]}) {'Kahramanmaraş' if te == 'TR' else 'Noto'}, 24 h count forecasts", fontsize=9)
    ax0 = fig.axes[0]
    h1, l1 = ax0.get_legend_handles_labels()
    h2, l2 = fig.axes[4].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for uz in ("png", "pdf"):
        fig.savefig(cik / f"Figure3.{uz}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="GRL makale şekillerini ve özet sayılarını üretir.")
    ap.add_argument("--ozetten", action="store_true", help="Ham veri yerine repodaki sonuclar/ dosyalarından şekilleri çizer")
    ap.add_argument("--cikti", default="", help="Şekillerin yazılacağı dizin (varsayılan: --ozetten için sekiller, tam modda makale/sekiller)")
    arg = ap.parse_args()

    cik = Path(arg.cikti) if arg.cikti else (KOK / ("sekiller" if arg.ozetten else "makale/sekiller"))
    if not cik.is_absolute():
        cik = KOK / cik
    cik.mkdir(parents=True, exist_ok=True)

    plt = _plt_ayarla()
    if arg.ozetten:
        sv, c5, c9, aktif = ozetten_yukle()
        say = None
    else:
        sv, c5, c9, say, aktif = tam_hesapla()

    ciz_figure1(plt, sv, aktif, cik)
    ciz_figure2(plt, sv, c5, cik)
    ciz_figure3(plt, c9, cik)

    if say is not None:
        (KOK / "makale").mkdir(parents=True, exist_ok=True)
        with open(KOK / "makale/sayilar.json", "w", encoding="utf-8") as f:
            json.dump(say, f, indent=1, ensure_ascii=False)
        np.savez_compressed(KOK / "makale/sekil_verisi.npz", **sv)
        print(json.dumps({k: {q: w for q, w in v_.items() if not isinstance(w, list)} for k, v_ in say.items() if k in aktif},
                         indent=1, ensure_ascii=False))
    else:
        print(f"Figure1, Figure2, Figure3 oluşturuldu: {cik}")


if __name__ == "__main__":
    main()
