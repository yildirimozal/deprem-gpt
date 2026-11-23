"""İlk saatler analizi (keşif amaçlı, ön kayıtlı karar kuralı yok). Kahramanmaraş testinde modelin
ETAS'a karşı ilk gün kazancı nereden geliyor?

Girdiler: sonuclar/c5_etas_ayristirma.npz (kod/etas_ayristir.py), sonuclar/c5_model_ayristirma.npz
(kod/model_ayristir.py, sunucuda), sonuclar/c3_*/test.npz, veri/koeri/koeri_2006_2024.csv
Çıktılar: sonuclar/c5_ozet.json, sonuclar/c5_tablolar.md, sekiller/c5_sekil1_ilk_saatler.{png,pdf}
Mac'te: /usr/bin/python3 kod/analiz5.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
v = np.load(KOK / "veri/turkiye.npz")
b, t, mc = v["bolum"], v["t_gun"], float(v["mc"])
test = np.where(b == 3)[0]
ana = int(v["ana_sok_idx"])
saat = (t[test] - t[ana]) * 24
m_test = v["buyukluk"][test]
SAAT_M76 = float((t[test][m_test >= 7.5].max() - t[ana]) * 24)
ARALIK = [(0, 1), (1, 3), (3, 9.12), (9.12, 12), (12, 24), (24, 72), (72, 720), (720, 9000)]
ETIKET = ["0–1 h", "1–3 h", "3–9 h", "9–12 h", "12–24 h", "1–3 d", "3–30 d", ">30 d"]

E = np.load(KOK / "sonuclar/c5_etas_ayristirma.npz")
M = np.load(KOK / "sonuclar/c5_model_ayristirma.npz")
assert np.array_equal(E["idx"], test) and np.array_equal(M["idx"], test)


def mod(k, q):
    return np.mean([M[f"c3_{k}_t{i}_{q}"] for i in (1, 2, 3)], 0)


kaynak = {"E2": (E["E2_log_lam"], E["E2_Lam"]), "E4": (E["E4_log_lam"], E["E4_Lam"]),
          "T0": (mod("T0", "log_h"), mod("T0", "H")), "T1": (mod("T1", "log_h"), mod("T1", "H")),
          "T2": (mod("T2", "log_h"), mod("T2", "H"))}

# ---------- büyüklük: sabit Gutenberg-Richter tabanı (Türkiye eğitim dönemi, Aki-Utsu, 0.1 bölme) ----------
me = v["buyukluk"][b == 1] - mc
beta = 1 / (me.mean() + 0.05)
q = np.exp(-beta * 0.1)
k = np.clip(np.round((m_test - mc) / 0.1), 0, 59).astype(int)
gr = np.log(1 - q) + k * np.log(q)
gr[k == 59] = 59 * np.log(q)
mll = {x: np.mean([np.load(KOK / f"sonuclar/c3_{x}_t{i}/test.npz")["mll"] for i in (1, 2, 3)], 0) for x in ["T0", "T1", "T2"]}

# ---------- tamlık: ham KOERI (M >= 2.0), dizi kutusu, en büyük eğrilik ----------
d = pd.read_csv(KOK / "veri/koeri/koeri_2006_2024.csv", dtype=str, usecols=["tarih", "saat", "enlem", "boylam", "derinlik", "xM"])
d["z"] = pd.to_datetime(d.tarih + " " + d.saat, format="%Y.%m.%d %H:%M:%S.%f")
for c in ["enlem", "boylam", "derinlik", "xM"]:
    d[c] = d[c].astype(float)
d = d[d.enlem.between(35.8, 38.8) & d.boylam.between(35.3, 39.7) & (d.derinlik <= 70)]
d["m"] = np.floor(d.xM * 10 + 0.5 + 1e-9) / 10
d_saat = (d.z - pd.Timestamp("2023-02-06 01:17:32.67")).dt.total_seconds().values / 3600


def maxc(x):
    u, n = np.unique(np.round(x * 10).astype(int), return_counts=True)
    return u[np.argmax(n)] / 10


ozet = {"b_degeri_egitim": float(beta / np.log(10)), "M7.6_saat": SAAT_M76, "aralik": []}
for (a, c), et in zip(ARALIK, ETIKET):
    mk = (saat > a) & (saat <= c)
    ham = d.m.values[(d_saat > a) & (d_saat <= c)]
    m3 = ham[ham >= 3.0]
    r = {"aralik": et, "n": int(mk.sum()), "ort_M": float(m_test[mk].mean()),
         "ham_M2_sayisi": int(len(ham)), "MAXC": float(maxc(ham)), "M3_icinde_3.0_3.4_orani": float(np.mean(m3 < 3.5))}
    for x, (ll, L) in kaynak.items():
        r[f"{x}_ort_Lambda"] = float(L[mk].mean())
        r[f"{x}_tll"] = float((ll - L)[mk].mean())
    for ref in ["E2", "E4"]:
        r[f"T1-{ref}_log_lam"] = float((kaynak["T1"][0] - kaynak[ref][0])[mk].mean())
        r[f"T1-{ref}_eksi_Lambda"] = float((-(kaynak["T1"][1] - kaynak[ref][1]))[mk].mean())
        r[f"T1-{ref}_tll"] = r[f"T1-{ref}_log_lam"] + r[f"T1-{ref}_eksi_Lambda"]
    r["GR_mll"] = float(gr[mk].mean())
    for x in mll:
        r[f"{x}-GR_mll"] = float((mll[x] - gr)[mk].mean())
    ozet["aralik"].append(r)
with open(KOK / "sonuclar/c5_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)

A = ozet["aralik"]
sat = [f"Gutenberg-Richter b (Türkiye eğitim): {ozet['b_degeri_egitim']:.3f}. M7.6 ana şoktan {SAAT_M76:.2f} saat sonra.", "",
       "| Aralık | n | Ort. M | Ham M≥2 (dizi kutusu) | MAXC | M≥3 içinde 3.0–3.4 |", "|---|---:|---:|---:|---:|---:|"]
sat += [f"| {r['aralik']} | {r['n']} | {r['ort_M']:.2f} | {r['ham_M2_sayisi']} | {r['MAXC']:.1f} | {r['M3_icinde_3.0_3.4_orani']:.2f} |" for r in A]
sat += ["", "Ortalama birikimli tehlike Λ (doğru modelde 1; > 1 fazla, < 1 az olay bekliyor):", "",
        "| Aralık | E2 | E4 kâhin | T0 | T1 | T2 |", "|---|---:|---:|---:|---:|---:|"]
sat += [f"| {r['aralik']} | {r['E2_ort_Lambda']:.2f} | {r['E4_ort_Lambda']:.2f} | {r['T0_ort_Lambda']:.2f} | {r['T1_ort_Lambda']:.2f} | {r['T2_ort_Lambda']:.2f} |" for r in A]
sat += ["", "T1 − ETAS farkının ayrışması (nat/olay): log yoğunluk + (−Λ) = TLL", "",
        "| Aralık | T1−E2 log λ | T1−E2 −Λ | T1−E2 TLL | T1−E4 log λ | T1−E4 −Λ | T1−E4 TLL |", "|---|---:|---:|---:|---:|---:|---:|"]
sat += [f"| {r['aralik']} | {r['T1-E2_log_lam']:+.3f} | {r['T1-E2_eksi_Lambda']:+.3f} | {r['T1-E2_tll']:+.3f} | "
        f"{r['T1-E4_log_lam']:+.3f} | {r['T1-E4_eksi_Lambda']:+.3f} | {r['T1-E4_tll']:+.3f} |" for r in A]
sat += ["", "Büyüklük log-olabilirliği, sabit Gutenberg-Richter'e göre (nat/olay):", "",
        "| Aralık | GR | T0 − GR | T1 − GR | T2 − GR |", "|---|---:|---:|---:|---:|"]
sat += [f"| {r['aralik']} | {r['GR_mll']:.3f} | {r['T0-GR_mll']:+.3f} | {r['T1-GR_mll']:+.3f} | {r['T2-GR_mll']:+.3f} |" for r in A]
(KOK / "sonuclar/c5_tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))

# ---------- şekil ----------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "savefig.dpi": 300})
KOYU, ORTA = "#404040", "#8c8c8c"
MAVI, MAVI_ACIK, MACENTA, MACENTA_ACIK, TURUNCU, KIRMIZI = "#1f3fcf", "#b9c6f2", "#d61fd0", "#f7c2f3", "#ef8f3a", "#e0342b"
fig, eks = plt.subplots(1, 3, figsize=(11, 3.4))

ax = eks[0]
sinir = 48
mk = (saat > 0) & (saat <= sinir)
o = np.argsort(saat[mk])
x = saat[mk][o]
ax.step(x, np.arange(1, len(x) + 1), where="post", color="black", lw=1.6, label="Observed (M ≥ 3.0)")
stil = {"E2": dict(color=MACENTA, ls="-", label="ETAS E2 (subcritical)"), "E4": dict(color=MACENTA, ls="--", label="ETAS E4 (oracle)"),
        "T0": dict(color=TURUNCU, ls="-.", label="T0 (Turkey only)"), "T1": dict(color=MAVI, ls="-", label="T1 (Turkey + others)")}
for kk, st in stil.items():
    ax.plot(x, np.cumsum(kaynak[kk][1][mk][o]), lw=1.4, **st)
ax.axvline(SAAT_M76, color=ORTA, lw=0.6, ls=":")
ax.text(SAAT_M76, 0.97, " M7.6", va="top", ha="left", fontsize=8, transform=ax.get_xaxis_transform())
ax.set_xlim(0, sinir)
ax.set_xlabel("Hours since the M7.8 mainshock")
ax.set_ylabel("Cumulative number of events")
ax.set_title("(a) Observed vs. expected count", fontsize=10)

xs = np.arange(len(A))
w = 0.2
desen = {"E2": dict(facecolor=MACENTA, hatch="////"), "E4": dict(facecolor=MACENTA_ACIK, hatch="...."),
         "T0": dict(facecolor=TURUNCU, hatch=None), "T1": dict(facecolor=MAVI, hatch=None)}
ax = eks[1]
for j, kk in enumerate(["E2", "E4", "T0", "T1"]):
    ax.bar(xs + (j - 1.5) * w, [r[f"{kk}_ort_Lambda"] for r in A], width=w, edgecolor="black", lw=0.6, **desen[kk])
ax.axhline(1, color=KIRMIZI, lw=1.0, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels(ETIKET, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean rescaled residual Λ")
ax.set_title("(b) Rate calibration (ideal = 1)", fontsize=10)

ax = eks[2]
for j, kk in enumerate(["T0", "T1"]):
    ax.bar(xs + (j - 0.5) * 0.3, [r[f"{kk}-GR_mll"] for r in A], width=0.3, edgecolor="black", lw=0.6, **desen[kk])
ax.axhline(0, color=KIRMIZI, lw=1.0, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels(ETIKET, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Magnitude log-likelihood minus GR (nats)")
ax.set_title("(c) Magnitude forecast vs. Gutenberg–Richter", fontsize=10)

h1, l1 = eks[0].get_legend_handles_labels()
from matplotlib.patches import Patch  # noqa: E402
h2 = [Patch(edgecolor="black", **desen[kk]) for kk in ["E2", "E4", "T0", "T1"]]
fig.legend(h1 + h2, l1 + ["ETAS E2 (bars)", "ETAS E4 (bars)", "T0 (bars)", "T1 (bars)"], loc="lower center", ncol=4,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1))
for uz in ("png", "pdf"):
    fig.savefig(KOK / f"sekiller/c5_sekil1_ilk_saatler.{uz}")
plt.close(fig)
