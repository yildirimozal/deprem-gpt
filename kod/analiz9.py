"""Olasılık çıktısı ve kalibrasyonu (ON_KAYIT_9.md): sonuclar/c9_ozet.json, c9_tablolar.md, sekiller/c9_*.

Mac'te: /usr/bin/python3 kod/analiz9.py
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
TEST = {"TR": dict(npz="turkiye.npz", ad="Turkey (Kahramanmaraş)", ana="M7.8"),
        "JP": dict(npz="japonya.npz", ad="Japan (Noto)", ana="M7.6")}
ESIK = {"M4": 1.0, "M5": 2.0}
KIRP = 1e-4
BINS = [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0001]
ozet = {"skor": {}, "fark": {}, "kalibrasyon": {}, "ana_sok_sonrasi": {}}
seri = {}
for te, c in TEST.items():
    v = np.load(KOK / "veri" / c["npz"])
    t, m = v["t_gun"], v["buyukluk"] - float(v["mc"])
    E = np.load(KOK / f"sonuclar/c9_{te}_etas/senaryo.npz")
    Ms = [np.load(KOK / f"sonuclar/c9_{te}_model/senaryo_t{s}.npz") for s in (1, 2, 3)]
    for z in Ms:
        assert np.allclose(z["yayin_t"], E["yayin_t"])
    yayin, tur = E["yayin_t"], E["tur"]
    model_maxm = np.concatenate([z["maxm"] for z in Ms], 1)
    model_olay = np.concatenate([z["olay"] for z in Ms], 1)
    tasma = int(sum(z["tasma"].sum() for z in Ms))
    gun = tur == 0
    gozlem = {}
    for ad, e in ESIK.items():
        gozlem[ad] = np.array([((t > t0) & (t <= t0 + 1) & (m >= e - 1e-9)).any() for t0 in yayin])
    gozlem_say = np.array([((t > t0) & (t <= t0 + 1)).sum() for t0 in yayin])
    ozet["kalibrasyon"][f"{te}|tasan_model_senaryo"] = tasma
    for ad, e in ESIK.items():
        pm = np.clip((model_maxm >= e - 1e-9).mean(1), KIRP, 1 - KIRP)
        pe = np.clip((E["maxm"] >= e - 1e-9).mean(1), KIRP, 1 - KIRP)
        o = gozlem[ad]
        ls_m = np.where(o, np.log(pm), np.log(1 - pm))
        ls_e = np.where(o, np.log(pe), np.log(1 - pe))
        br_m, br_e = (pm - o) ** 2, (pe - o) ** 2
        g = gun
        gun_no = np.floor((yayin[g] - yayin[g][0]) / 7).astype(int)
        ub, ters = np.unique(gun_no, return_inverse=True)

        def boot(f):
            top, say = np.bincount(ters, weights=f), np.bincount(ters)
            s = rng.integers(0, len(ub), (N_BOOT, len(ub)))
            return [float(x) for x in np.quantile(top[s].sum(1) / say[s].sum(1), [.025, .975])]

        ozet["skor"][f"{te}|{ad}"] = {"n_gun": int(g.sum()), "gozlenen_oran": float(o[g].mean()),
                                      "model_ort_olasilik": float(pm[g].mean()), "etas_ort_olasilik": float(pe[g].mean()),
                                      "model_log": float(ls_m[g].mean()), "etas_log": float(ls_e[g].mean()),
                                      "model_brier": float(br_m[g].mean()), "etas_brier": float(br_e[g].mean()),
                                      "iklim_log": float(np.where(o[g], np.log(o[g].mean()), np.log(1 - o[g].mean())).mean()),
                                      "iklim_brier": float(((o[g].mean() - o[g]) ** 2).mean())}
        ozet["fark"][f"{te}|{ad}"] = {"log": float((ls_m - ls_e)[g].mean()), "log_GA": boot((ls_m - ls_e)[g]),
                                      "brier": float((br_m - br_e)[g].mean()), "brier_GA": boot((br_m - br_e)[g])}
        kal = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            for nm, p in [("model", pm), ("etas", pe)]:
                k = g & (p >= lo) & (p < hi)
                if k.sum():
                    kal.append({"dilim": [lo, min(hi, 1)], "kim": nm, "n": int(k.sum()), "ort_p": float(p[k].mean()), "gozlenen": float(o[k].mean())})
        ozet["kalibrasyon"][f"{te}|{ad}"] = kal
        seri[(te, ad)] = dict(pm=pm, pe=pe, o=o)
    a = ~gun
    ozet["ana_sok_sonrasi"][te] = {"saat": [float(x) for x in (yayin[a] - t[int(v["ana_sok_idx"])]) * 24],
                                   "model_P_M5": [float(x) for x in seri[(te, "M5")]["pm"][a]],
                                   "etas_P_M5": [float(x) for x in seri[(te, "M5")]["pe"][a]],
                                   "gozlenen_M5": [bool(x) for x in seri[(te, "M5")]["o"][a]],
                                   "model_beklenen_olay": [float(x) for x in model_olay[a].mean(1)],
                                   "etas_beklenen_olay": [float(x) for x in E["olay"][a].mean(1)],
                                   "gozlenen_olay": [int(x) for x in gozlem_say[a]],
                                   "model_olay_P05_P95": [[float(x) for x in np.percentile(r, [5, 95])] for r in model_olay[a]],
                                   "etas_olay_P05_P95": [[float(x) for x in np.percentile(r, [5, 95])] for r in E["olay"][a]]}
    seri[(te, "gun")] = dict(yayin=yayin, gun=gun, ana_t=t[int(v["ana_sok_idx"])])

for te, r in ozet["ana_sok_sonrasi"].items():                 # ikincil: gözlenen sayının %5–95 aralığına düşme oranı
    g_ = np.array(r["gozlenen_olay"])
    for kim in ("model", "etas"):
        a_ = np.array(r[f"{kim}_olay_P05_P95"])
        r[f"{kim}_kapsama"] = float(((g_ >= a_[:, 0]) & (g_ <= a_[:, 1])).mean())
iyi = sum(ozet["fark"][k]["log"] > 0 and ozet["fark"][k]["log_GA"][0] > 0 for k in ozet["fark"])
kotu_tam = sum(ozet["fark"][k]["log_GA"][1] < 0 for k in ozet["fark"])
karar = "DAHA_IYI" if iyi >= 3 and kotu_tam == 0 else ("DAHA_KOTU" if kotu_tam >= 2 else "KARISIK")
kalibre = {}
for k, s_ in ozet["skor"].items():
    r = s_["model_ort_olasilik"] / max(s_["gozlenen_oran"], 1e-9)
    kalibre[k] = {"oran": r, "kalibre_degil": bool(r < 0.5 or r > 2)}
ozet["karar"] = {"karar": karar, "GA_sifir_ustu_hedef": iyi, "GA_sifir_alti_hedef": kotu_tam, "model_yil_ortalamasi_kalibrasyon": kalibre}
with open(KOK / "sonuclar/c9_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)


def ga(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


sat = ["| Hedef | Gün | Gözlenen oran | Model ort. olasılık | ETAS ort. olasılık | Log skor model | ETAS | iklim | Model − ETAS (7 g GA) | Brier model | ETAS | iklim | Brier farkı (GA) |",
       "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|"]
for k, s_ in ozet["skor"].items():
    f = ozet["fark"][k]
    sat.append(f"| {k} | {s_['n_gun']} | {s_['gozlenen_oran']:.3f} | {s_['model_ort_olasilik']:.3f} | {s_['etas_ort_olasilik']:.3f} | "
               f"{s_['model_log']:.4f} | {s_['etas_log']:.4f} | {s_['iklim_log']:.4f} | {f['log']:+.4f} {ga(f['log_GA'])} | "
               f"{s_['model_brier']:.4f} | {s_['etas_brier']:.4f} | {s_['iklim_brier']:.4f} | {f['brier']:+.4f} {ga(f['brier_GA'])} |")
sat += ["", "Kalibrasyon (günlük tahminler, olasılık dilimine göre):", "", "| Hedef | Kim | Dilim | n | Ort. olasılık | Gözlenen sıklık |", "|---|---|---|---:|---:|---:|"]
for k in ozet["skor"]:
    for r in ozet["kalibrasyon"][k]:
        sat.append(f"| {k} | {r['kim']} | {r['dilim'][0]:.2f}–{r['dilim'][1]:.2f} | {r['n']} | {r['ort_p']:.3f} | {r['gozlenen']:.3f} |")
sat += ["", "Ana şoktan sonra 3 saatte bir yayın (önümüzdeki 24 saat):", ""]
for te, r in ozet["ana_sok_sonrasi"].items():
    sat += [f"**{te}**", "", "| Saat | P(M≥5) model | P(M≥5) ETAS | M≥5 oldu mu | Beklenen olay model [P5–P95] | ETAS [P5–P95] | Gözlenen olay |", "|---:|---:|---:|---|---|---|---:|"]
    for i in range(len(r["saat"])):
        sat.append(f"| {r['saat'][i]:.1f} | {r['model_P_M5'][i]:.2f} | {r['etas_P_M5'][i]:.2f} | {'evet' if r['gozlenen_M5'][i] else 'hayır'} | "
                   f"{r['model_beklenen_olay'][i]:.0f} [{r['model_olay_P05_P95'][i][0]:.0f}–{r['model_olay_P05_P95'][i][1]:.0f}] | "
                   f"{r['etas_beklenen_olay'][i]:.0f} [{r['etas_olay_P05_P95'][i][0]:.0f}–{r['etas_olay_P05_P95'][i][1]:.0f}] | {r['gozlenen_olay'][i]} |")
    sat.append("")
sat += ["%5–95 aralığının gözlenen sayıyı kapsama oranı (ana şok sonrası 25 yayın): " + ", ".join(
    f"{te} model {r['model_kapsama']:.2f}, ETAS {r['etas_kapsama']:.2f}" for te, r in ozet["ana_sok_sonrasi"].items()), ""]
sat += [f"Taşan model senaryosu (adım sınırına ulaşan): " + ", ".join(f"{te}: {ozet['kalibrasyon'][f'{te}|tasan_model_senaryo']}" for te in TEST),
        "", f"**Karar (ön kayıtlı kural): {karar}** (GA'sı sıfırın üstünde hedef: {iyi}/4, tamamen altında: {kotu_tam}/4)",
        "Yıl ortalamasında model öngörülen/gözlenen oranı: " + ", ".join(f"{k} {x['oran']:.2f}" + (" (KALİBRE DEĞİL)" if x["kalibre_degil"] else "") for k, x in kalibre.items())]
(KOK / "sonuclar/c9_tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))

# ---------- şekiller ----------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "savefig.dpi": 300})
KOYU, ORTA = "#404040", "#8c8c8c"
fig, eks = plt.subplots(1, 4, figsize=(11, 3.0))
for j, (te, ad) in enumerate([(te, ad) for te in TEST for ad in ESIK]):
    ax = eks[j]
    ax.plot([0, 1], [0, 1], color=ORTA, lw=0.8, ls=":")
    for kim, st in [("model", dict(color="black", marker="o", ls="-")), ("etas", dict(color=KOYU, marker="s", ls="--", mfc="white"))]:
        r = [x for x in ozet["kalibrasyon"][f"{te}|{ad}"] if x["kim"] == kim]
        ax.plot([x["ort_p"] for x in r], [x["gozlenen"] for x in r], ms=4, lw=1, label="Model (T1/J1)" if kim == "model" else "ETAS", **st)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"({'abcd'[j]}) {te}, {ad[0]} ≥ {ad[1]}", fontsize=10)
    ax.set_xlabel("Forecast probability")
    if j == 0:
        ax.set_ylabel("Observed frequency")
h, l = eks[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
for uz in ("png", "pdf"):
    fig.savefig(KOK / f"sekiller/c9_sekil1_kalibrasyon.{uz}")
plt.close(fig)

fig, eks = plt.subplots(1, 2, figsize=(10, 3.3))
for j, te in enumerate(TEST):
    ax = eks[j]
    r = ozet["ana_sok_sonrasi"][te]
    x = np.array(r["saat"])
    ax.plot(x, r["model_beklenen_olay"], color="black", marker="o", ms=3, lw=1.2, label="Model: expected count")
    ax.fill_between(x, [a_[0] for a_ in r["model_olay_P05_P95"]], [a_[1] for a_ in r["model_olay_P05_P95"]], color=ORTA, alpha=0.35, lw=0, label="Model: 5–95%")
    ax.plot(x, r["etas_beklenen_olay"], color=KOYU, marker="s", ms=3, lw=1.2, ls="--", mfc="white", label="ETAS: expected count")
    ax.plot(x, r["gozlenen_olay"], color="black", ls="none", marker="x", ms=5, label="Observed count")
    ax.set_yscale("log")
    ax.set_xlabel(f"Hours after the {TEST[te]['ana']} mainshock (issue time)")
    ax.set_ylabel("Events M ≥ 3.0 in next 24 h")
    ax.set_title(f"({'ab'[j]}) {TEST[te]['ad']}", fontsize=10)
h, l = eks[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1))
for uz in ("png", "pdf"):
    fig.savefig(KOK / f"sekiller/c9_sekil2_artci_tahmini.{uz}")
plt.close(fig)
