"""Haritalı artçı tahmini şekli (ON_KAYIT_10.md, ikincil): ana şoktan 12 saat sonra yayınlanan, önümüzdeki 24
saat için beklenen olay sayısı (10 km hücre), model S2 ve uzaysal ETAS; üstünde gerçekleşen olaylar.
Ayrıca hücre bazında Poisson log-olabilirliği (beklenen sayıya 0.001 taban eklenir, iki yönteme de aynı).

Mac'te: /usr/bin/python3 kod/harita_ciz.py   →  sekiller/c10_sekil1_harita.{png,pdf}, sonuclar/c10_harita_ozet.json
"""
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
kn = np.load(KOK / "veri/konum.npz")
AYAR = {"TR": dict(npz="turkiye.npz", ad="Turkiye_KOERI", kutu=(35.6, 39.4, 35.0, 40.0), baslik="Kahramanmaraş, 12 h after M7.8"),
        "JP": dict(npz="japonya.npz", ad="Japonya_JMA", kutu=(36.3, 38.7, 135.6, 139.0), baslik="Noto, 12 h after M7.6")}
HUCRE = 10.0
ozet = {}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.labelweight": "bold", "axes.titleweight": "bold",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "savefig.dpi": 300})
fig, eks = plt.subplots(2, 2, figsize=(9.5, 8.2))
for i, (te, c) in enumerate(AYAR.items()):
    v = np.load(KOK / "veri" / c["npz"])
    M = np.load(KOK / f"sonuclar/c10_{te}_harita/senaryo_konum.npz")
    E = np.load(KOK / f"sonuclar/c10_{te}_harita/senaryo_konum_etas.npz")
    t0 = float(M["t0"])
    assert abs(t0 - float(E["t0"])) < 1e-9
    la_m, lo_m = np.median(kn[f"{c['ad']}_enlem"]), np.median(kn[f"{c['ad']}_boylam"])
    m_lat = la_m + M["y"] * 1000 / 111.2
    m_lon = lo_m + M["x"] * 1000 / (111.2 * math.cos(math.radians(la_m)))
    la_e, lo_e = E["merkez"]
    e_lat = la_e + E["y_km"] / 111.2
    e_lon = lo_e + E["x_km"] / (111.2 * math.cos(math.radians(la_e)))
    gk = (v["t_gun"] > t0) & (v["t_gun"] <= t0 + 1)
    o_lat, o_lon, o_m = v["enlem"][gk], v["boylam"][gk], v["buyukluk"][gk]
    la1, la2, lo1, lo2 = c["kutu"]
    lac = (la1 + la2) / 2
    nb_la = int(round((la2 - la1) * 111.2 / HUCRE))
    nb_lo = int(round((lo2 - lo1) * 111.2 * math.cos(math.radians(lac)) / HUCRE))
    kenar = (np.linspace(la1, la2, nb_la + 1), np.linspace(lo1, lo2, nb_lo + 1))
    Hm = np.histogram2d(m_lat, m_lon, bins=kenar)[0] / float(M["n_senaryo"])
    He = np.histogram2d(e_lat, e_lon, bins=kenar)[0] / float(E["n_senaryo"])
    Ho = np.histogram2d(o_lat, o_lon, bins=kenar)[0]
    lg = lambda H: float((Ho * np.log(H + 1e-3) - (H + 1e-3)).sum())          # noqa: E731
    ozet[te] = {"gozlenen_kutu_ici": int(Ho.sum()), "gozlenen_toplam": int(gk.sum()),
                "model_beklenen_kutu_ici": float(Hm.sum()), "etas_beklenen_kutu_ici": float(He.sum()),
                "model_hucre_poisson_ll": lg(Hm), "etas_hucre_poisson_ll": lg(He), "hucre_km": HUCRE,
                "model_senaryo": int(M["n_senaryo"]), "etas_senaryo": int(E["n_senaryo"])}
    vmax = max(Hm.max(), He.max())
    for j, (H, ad) in enumerate([(Hm, "Model S2"), (He, "Spatial ETAS")]):
        ax = eks[i, j]
        im = ax.pcolormesh(kenar[1], kenar[0], np.where(H > 0, H, np.nan), cmap="RdPu", norm=LogNorm(vmin=1e-2, vmax=vmax), shading="flat")
        ax.scatter(o_lon, o_lat, s=4, facecolors="none", edgecolors="black", linewidths=0.4)
        ax.set_xlim(lo1, lo2)
        ax.set_ylim(la1, la2)
        ax.set_aspect(1 / math.cos(math.radians(lac)))
        ax.set_title(f"({'abcd'[2 * i + j]}) {ad}: {c['baslik']}", fontsize=9)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.text(0.02, 0.02, f"expected {H.sum():.0f}, observed {int(Ho.sum())}", transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    cb = fig.colorbar(im, ax=eks[i, :].tolist(), shrink=0.8, pad=0.02)
    cb.set_label("Expected events per 10 km cell in next 24 h", fontweight="bold")
for uz in ("png", "pdf"):
    fig.savefig(KOK / f"sekiller/c10_sekil1_harita.{uz}", bbox_inches="tight")
plt.close(fig)
with open(KOK / "sonuclar/c10_harita_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)
print(json.dumps(ozet, indent=1, ensure_ascii=False))
