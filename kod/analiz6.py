"""Eksik kaydı modelleyen ETAS'a karşı (ON_KAYIT_6.md): sonuclar/c6_ozet.json, c6_tablolar.md, sekiller/c6_*.

E9 ön kayıt dışı bir keşif varyantı: kâhinin (E8) algılama parametreleri sabit, ETAS 2023 öncesinde uydurulmuş.
Mac'te: /usr/bin/python3 kod/analiz6.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import log_ndtr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)
N_BOOT = 2000
v = np.load(KOK / "veri/turkiye.npz")
b, t, mc = v["bolum"], v["t_gun"], float(v["mc"])
test = np.where(b == 3)[0]
tt = t[test]
ana = int(v["ana_sok_idx"])
saat = (tt - t[ana]) * 24
x_m = np.round(v["buyukluk"][test] - mc, 1)

ETAS = {}
for k in ["E6", "E7", "E8", "E9", "kapali"]:
    z = np.load(KOK / f"sonuclar/etas6_{k}/etas_test.npz")
    assert np.array_equal(z["idx"], test)
    ETAS[k] = dict(z)
    ETAS[k]["bilgi"] = json.loads((KOK / f"sonuclar/etas6_{k}/etas.json").read_text(encoding="utf-8"))
for k in ["E1", "E2", "E4", "E5"]:
    ETAS[k] = {"tll": np.load(KOK / f"sonuclar/etas4_{k}/etas_test.npz")["tll"]}
denetim = float(np.abs(ETAS["kapali"]["tll"] - ETAS["E2"]["tll"]).max())
mod = {k: [np.load(KOK / f"sonuclar/c3_{k}_t{s}/test.npz") for s in (1, 2, 3)] for k in ["T0", "T1", "T2"]}
TLL = {k: [z["tll"] for z in x] for k, x in mod.items()}
MLL = {k: [z["mll"] for z in x] for k, x in mod.items()}


def boot(f, mk, blok_gun):
    f2, bb = f[mk], np.floor((tt[mk] - tt[0]) / blok_gun).astype(int)
    ub, te = np.unique(bb, return_inverse=True)
    top, say = np.bincount(te, weights=f2), np.bincount(te)
    s = rng.integers(0, len(ub), (N_BOOT, len(ub)))
    return [float(x) for x in np.quantile(top[s].sum(1) / say[s].sum(1), [.025, .975])] + [int(len(ub))]


KUME = {"ilk_gun": ((saat > 0) & (saat <= 24), 1 / 24), "0-1 saat": ((saat > 0) & (saat <= 1), None),
        "1-9 saat": ((saat > 1) & (saat <= 9.12), 1 / 24), "9-24 saat": ((saat > 9.12) & (saat <= 24), 1 / 24),
        "1-3 gun": ((saat > 24) & (saat <= 72), 1 / 24), "tum_test": (np.ones(len(test), bool), 7)}

ozet = {"denetim_kapali_eksi_E2_max": denetim, "etas": {}, "kiyas": {}, "lambda": {}, "buyukluk": {}}
for k in ["E2", "E6", "E7", "E8", "E9"]:
    e = ETAS[k]
    r = {"test_tll": float(e["tll"].mean())}
    if "bilgi" in e:
        s = e["bilgi"]["secilen"]
        r.update({q: s[q] for q in ["a", "c", "p", "G", "H", "dallanma", "uydurma_tll", "mu", "K"]})
        r["denemeler_uydurma_tll"] = [d["uydurma_tll"] for d in e["bilgi"]["denemeler"]]
    r["kume"] = {ad: float(e["tll"][mk].mean()) for ad, (mk, _) in KUME.items()}
    ozet["etas"][k] = r
en_guclu_eksik = max(["E6", "E7"], key=lambda k: ozet["etas"][k]["test_tll"])
en_guclu_gecerli = max(["E1", "E2", "E5", "E6", "E7"], key=lambda k: float(ETAS[k]["tll"].mean()))
ozet["en_guclu_eksik_kayitli"] = en_guclu_eksik
ozet["en_guclu_gecerli_tum"] = en_guclu_gecerli
for m_ad in ["T1", "T2", "T0"]:
    ort = np.mean(TLL[m_ad], 0)
    for k in ["E2", "E6", "E7", "E8", "E9"]:
        for ad, (mk, bg) in KUME.items():
            f = ort - ETAS[k]["tll"]
            r = {"ort": float(f[mk].mean()), "tohum": [float((x - ETAS[k]["tll"])[mk].mean()) for x in TLL[m_ad]]}
            if bg:
                r["GA"] = boot(f, mk, bg)
            ozet["kiyas"][f"{m_ad}-{k}|{ad}"] = r

r1 = ozet["kiyas"][f"T1-{en_guclu_eksik}|ilk_gun"]
if r1["ort"] >= 0.05 and all(x > 0 for x in r1["tohum"]):
    karar = "KALIYOR"
elif r1["ort"] <= 0.02:
    karar = "KAPANIYOR"
else:
    karar = "KISMEN"
ozet["karar"] = {"karar": karar, "en_guclu_eksik_kayitli": en_guclu_eksik, "T1-ilk_gun": r1,
                 "T1-tum_test_en_guclu_gecerli": ozet["kiyas"].get(f"T1-{en_guclu_gecerli}|tum_test")}

# Λ kalibrasyonu
ARALIK = [(0, 1), (1, 3), (3, 9.12), (9.12, 12), (12, 24), (24, 72), (72, 720), (720, 9000)]
ET = ["0–1 h", "1–3 h", "3–9 h", "9–12 h", "12–24 h", "1–3 d", "3–30 d", ">30 d"]
M = np.load(KOK / "sonuclar/c5_model_ayristirma.npz")
H_T1 = np.mean([M[f"c3_T1_t{i}_H"] for i in (1, 2, 3)], 0)
for (a, c), et in zip(ARALIK, ET):
    mk = (saat > a) & (saat <= c)
    ozet["lambda"][et] = {k: float(ETAS[k]["Lam"][mk].mean()) for k in ["E6", "E7", "E8", "E9"]}
    ozet["lambda"][et]["T1"] = float(H_T1[mk].mean())

# büyüklük: 0.1'lik bölme olasılıkları. Sürekli GR Mc - 0.05'ten başlar (yuvarlanmış M = Mc bölmesi
# [Mc-0.05, Mc+0.05)); zamana bağlı sürümde yoğunluk Φ((y - μ_d)/σ) ile çarpılır ve yeniden normalize edilir.
beta = ETAS["E6"]["bilgi"]["beta"]
SIG = 0.2
q = np.exp(-beta * 0.1)
k_m = np.clip(np.round(x_m / 0.1), 0, 59).astype(int)
gr_sabit = np.log(1 - q) + k_m * np.log(q)                 # analiz5.py ile aynı ayrık GR
gr_sabit[k_m == 59] = 59 * np.log(q)


def gr_log(mud):
    a0 = -0.05 - mud                                       # normalizasyon: ∫_{-0.05}^∞ β e^{-β(y+0.05)} Φ((y-μ)/σ) dy
    z = a0 / SIG
    logP = np.logaddexp(log_ndtr(z), beta * a0 + 0.5 * beta ** 2 * SIG ** 2 + log_ndtr(-(z + beta * SIG)))
    y = x_m[:, None] - 0.05 + (np.arange(40)[None, :] + 0.5) * 0.1 / 40      # bölme içi orta nokta kuralı
    pay = np.log((beta * np.exp(-beta * (y + 0.05)) * np.exp(log_ndtr((y - mud[:, None]) / SIG))).mean(1) * 0.1)
    return pay - logP
GR = {"sabit": gr_sabit, "E6": gr_log(ETAS["E6"]["mu_d_goreli"]), "E7": gr_log(ETAS["E7"]["mu_d_goreli"]),
      "E8": gr_log(ETAS["E8"]["mu_d_goreli"])}
mll_T1 = np.mean(MLL["T1"], 0)
for (a, c), et in zip(ARALIK, ET):
    mk = (saat > a) & (saat <= c)
    ozet["buyukluk"][et] = {f"T1-GR_{k}": float((mll_T1 - g)[mk].mean()) for k, g in GR.items()}
    ozet["buyukluk"][et]["T1_ilk_gun_tohum_GR_E8"] = [float((x - GR["E8"])[mk].mean()) for x in MLL["T1"]]
mk = (saat > 0) & (saat <= 24)
ozet["buyukluk"]["ilk_gun"] = {f"T1-GR_{k}": float((mll_T1 - g)[mk].mean()) for k, g in GR.items()}
ozet["buyukluk"]["ilk_gun"]["GA_1saat_GR_E8"] = boot(mll_T1 - GR["E8"], mk, 1 / 24)
ozet["buyukluk"]["tum_test"] = {f"T1-GR_{k}": float((mll_T1 - g).mean()) for k, g in GR.items()}
with open(KOK / "sonuclar/c6_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)


def ga(x):
    return f"[{x[0]:+.3f}, {x[1]:+.3f}]"


sat = [f"Denetim: algılama kapalıyken E2 ile en büyük fark {denetim:.1e}", "",
       "| ETAS | Test TLL | İlk gün | 0–1 saat | 1–9 saat | 9–24 saat | 1–3 gün | a | c | p | G | H | Dallanma | Uydurma TLL (3 başlangıç) |",
       "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for k in ["E2", "E6", "E7", "E8", "E9"]:
    e = ozet["etas"][k]
    kk = e["kume"]
    ek = (f"{e['a']:.3f} | {e['c']:.4f} | {e['p']:.3f} | {e['G']:.2f} | {e['H']:.2f} | {e['dallanma']:.2f} | "
          + ", ".join(f"{y:.5f}" for y in e["denemeler_uydurma_tll"])) if "a" in e else "– | – | – | – | – | – | –"
    sat.append(f"| {k} | {e['test_tll']:.4f} | {kk['ilk_gun']:.3f} | {kk['0-1 saat']:.3f} | {kk['1-9 saat']:.3f} | "
               f"{kk['9-24 saat']:.3f} | {kk['1-3 gun']:.3f} | {ek} |")
sat += ["", f"Eksik kayıtlı en güçlü (E6, E7): **{en_guclu_eksik}**. Tüm geçerliler içinde en güçlü (E1, E2, E5, E6, E7): **{en_guclu_gecerli}**", "",
        "| Kıyas | İlk gün (1 saat blok GA) | 0–1 saat (tohumlar) | 1–9 saat | 9–24 saat | 1–3 gün | Tüm test (7 g GA) |",
        "|---|---|---|---|---|---|---|"]
for m_ad in ["T1", "T2", "T0"]:
    for k in ["E2", "E6", "E7", "E8", "E9"]:
        q = lambda ad: ozet["kiyas"][f"{m_ad}-{k}|{ad}"]  # noqa: E731
        sat.append(f"| {m_ad} − {k} | {q('ilk_gun')['ort']:+.3f} {ga(q('ilk_gun')['GA'])} | {q('0-1 saat')['ort']:+.3f} ("
                   + ", ".join(f"{y:+.2f}" for y in q('0-1 saat')['tohum']) + f") | {q('1-9 saat')['ort']:+.3f} | "
                   f"{q('9-24 saat')['ort']:+.3f} | {q('1-3 gun')['ort']:+.3f} | {q('tum_test')['ort']:+.3f} {ga(q('tum_test')['GA'])} |")
sat += ["", f"**Karar (ön kayıtlı kural): {karar}** (T1 − {en_guclu_eksik}, ilk gün: {r1['ort']:+.3f} {ga(r1['GA'])}, tohumlar "
        + ", ".join(f"{y:+.3f}" for y in r1["tohum"]) + ")", "",
        "Ortalama Λ (doğru modelde 1):", "", "| Aralık | E6 | E7 | E8 kâhin | E9 | T1 |", "|---|---:|---:|---:|---:|---:|"]
for et in ET:
    x = ozet["lambda"][et]
    sat.append(f"| {et} | {x['E6']:.2f} | {x['E7']:.2f} | {x['E8']:.2f} | {x['E9']:.2f} | {x['T1']:.2f} |")
sat += ["", "Büyüklük: T1 − GR (nat/olay; 0.1 bölme olasılıkları):", "", "| Aralık | sabit GR | GR + E6 algılama | GR + E7 algılama | GR + E8 algılama (kâhin) |",
        "|---|---:|---:|---:|---:|"]
for et in ET + ["ilk_gun", "tum_test"]:
    x = ozet["buyukluk"][et]
    sat.append(f"| {et} | {x['T1-GR_sabit']:+.3f} | {x['T1-GR_E6']:+.3f} | {x['T1-GR_E7']:+.3f} | {x['T1-GR_E8']:+.3f} |")
sat.append(f"\nİlk gün T1 − (GR + E8 algılama), 1 saat blok GA: {ga(ozet['buyukluk']['ilk_gun']['GA_1saat_GR_E8'])}")
(KOK / "sonuclar/c6_tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))

# şekil: ilk 72 saatte birikimli kazanç T1 − ETAS
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "savefig.dpi": 300})
KOYU, ORTA = "#404040", "#8c8c8c"
fig, eks = plt.subplots(1, 2, figsize=(9.5, 3.4))
mk = (saat > 0) & (saat <= 72)
o = np.argsort(saat[mk])
ort = np.mean(TLL["T1"], 0)
stil = {"E2": dict(color=ORTA, ls="-", label="E2: subcritical ETAS"), "E7": dict(color="black", ls="--", label="E7: + incompleteness, fitted pre-2023"),
        "E8": dict(color="black", ls=":", label="E8: oracle")}          # E9 ölçeği ezdiği için şekilde yok (tabloda var)
ax = eks[0]
for k, st in stil.items():
    ax.plot(saat[mk][o], np.cumsum((ort - ETAS[k]["tll"])[mk][o]), lw=1.2, **st)
ax.axhline(0, color="black", lw=0.6)
ax.axvline(9.12, color=ORTA, lw=0.6, ls=":")
ax.text(9.12, 0.97, " M7.6", va="top", ha="left", fontsize=8, transform=ax.get_xaxis_transform())
ax.set_xlim(0, 72)
ax.set_xlabel("Hours since the M7.8 mainshock")
ax.set_ylabel("Cumulative T1 gain over ETAS (nats)")
ax.set_title("(a) Temporal log-likelihood", fontsize=10)
ax = eks[1]
xs = np.arange(len(ET))
for j, (k, de) in enumerate([("sabit", dict(facecolor="white", hatch="////")), ("E8", dict(facecolor=KOYU))]):
    ax.bar(xs + (j - 0.5) * 0.35, [ozet["buyukluk"][e][f"T1-GR_{k}"] for e in ET], width=0.35, edgecolor="black", lw=0.6,
           label="vs. constant GR" if k == "sabit" else "vs. GR with oracle detection (E8)", **de)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(ET, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("T1 magnitude log-likelihood gain (nats)")
ax.set_title("(b) Magnitude forecast", fontsize=10)
h1, l1 = eks[0].get_legend_handles_labels()
h2, l2 = eks[1].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1))
for uz in ("png", "pdf"):
    fig.savefig(KOK / f"sekiller/c6_sekil1_eksik_kayit.{uz}")
plt.close(fig)
