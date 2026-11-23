"""Konum çıktısı (ON_KAYIT_10.md): sonuclar/c10_ozet.json, c10_tablolar.md, sekiller/c10_sekil1.

Birincil: model S (ön kayıtlı) − uzaysal ETAS, olay başına SLL (km²). Keşif (ön kayıt dışı): S2 (işaretçi
başlık, 12 000 adım) ve S_uzun (S, 12 000 adım, tek tohum).
Yalnız arka plan tabanı burada logaritma uzayında yeniden hesaplanır (sunucuda float32 taşması vardı).
Mac'te: /usr/bin/python3 kod/analiz10.py
"""
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)
N_BOOT = 2000
KM2 = 2 * math.log(1000)
TEST = {"TR": dict(npz="turkiye.npz", ad="Kahramanmaraş", etas_t="etas4_E2", etas_s="etas10_TR"),
        "JP": dict(npz="japonya.npz", ad="Noto", etas_t="etas7_E5", etas_s="etas10_JP")}
ozet = {"test": {}, "fark": {}, "etas_uzay": {}}


def lse(a, ax):
    m_ = a.max(ax, keepdims=True)
    return (m_ + np.log(np.exp(a - m_).sum(ax, keepdims=True))).squeeze(ax)


for te, c in TEST.items():
    v = np.load(KOK / "veri" / c["npz"])
    test = np.where(v["bolum"] == 3)[0]
    tt = v["t_gun"][test]
    sa = (tt - v["t_gun"][int(v["ana_sok_idx"])])
    kume = {"tum_yil": (np.ones(len(test), bool), 7), "ilk_gun": ((sa > 0) & (sa <= 1), 1 / 24)}
    ej = json.loads((KOK / f"sonuclar/{c['etas_s']}/etas_uzay.json").read_text(encoding="utf-8"))
    es = np.load(KOK / f"sonuclar/{c['etas_s']}/etas_uzay_test.npz")
    assert np.array_equal(es["idx"], test)
    ozet["etas_uzay"][te] = ej["secilen"]
    # arka plan: log u(x) logaritma uzayında
    la0, lo0 = ej["merkez"]
    x = (v["boylam"] - lo0) * np.cos(np.radians(la0)) * 111.2
    y = (v["enlem"] - la0) * 111.2
    eg = np.where(v["bolum"] == 1)[0]
    h = ej["secilen"]["h_km"]
    lu = np.empty(len(test))
    for k in range(0, len(test), 200):
        i = test[k:k + 200]
        d2 = (x[i][:, None] - x[eg][None]) ** 2 + (y[i][:, None] - y[eg][None]) ** 2
        lu[k:k + 200] = lse(-d2 / (2 * h * h), 1) - math.log(len(eg)) - math.log(2 * math.pi * h * h)
    etas_t = np.load(KOK / f"sonuclar/{c['etas_t']}/etas_test.npz")["tll"]
    seriler = {"ETAS": dict(sll=es["sll"], tll=etas_t), "Arka plan": dict(sll=lu, tll=None)}
    a0 = [np.load(KOK / f"sonuclar/c8_{te}_A0_t{s}/test.npz")["tll"] for s in (1, 2, 3)]
    for kod, tohum in [("S", (1, 2, 3)), ("S2", (1, 2, 3)), ("Suzun", (1,))]:
        dz = [KOK / f"sonuclar/c10_{te}_{kod}_t{s}/test.npz" for s in tohum]
        if not all(p.exists() for p in dz):
            continue
        z = [np.load(p) for p in dz]
        seriler[kod] = dict(sll=np.mean([q["sll"] - KM2 for q in z], 0), tll=np.mean([q["tll"] for q in z], 0),
                            sll_tohum=[q["sll"] - KM2 for q in z], tll_tohum=[q["tll"] for q in z],
                            adim=[json.loads((p.parent / "kayit.json").read_text(encoding="utf-8"))["en_iyi_adim"] for p in dz])

    def boot(f, mk, bg):
        f2, bb = f[mk], np.floor((tt[mk] - tt[0]) / bg).astype(int)
        ub, ters = np.unique(bb, return_inverse=True)
        top, say = np.bincount(ters, weights=f2), np.bincount(ters)
        sec = rng.integers(0, len(ub), (N_BOOT, len(ub)))
        return [float(q) for q in np.quantile(top[sec].sum(1) / say[sec].sum(1), [.025, .975])]

    r = {}
    for ad, sr in seriler.items():
        r[ad] = {k: float(sr["sll"][mk].mean()) for k, (mk, _) in kume.items()}
        if sr.get("tll") is not None:
            r[ad].update({f"tll_{k}": float(sr["tll"][mk].mean()) for k, (mk, _) in kume.items()})
        if "adim" in sr:
            r[ad]["en_iyi_adim"] = sr["adim"]
            r[ad]["sll_tohum"] = [float(q.mean()) for q in sr["sll_tohum"]]
    r["A0_tll"] = {k: float(np.mean(a0, 0)[mk].mean()) for k, (mk, _) in kume.items()}
    ozet["test"][te] = r
    for kod in [k for k in ("S", "S2", "Suzun") if k in seriler]:
        for ad, (mk, bg) in kume.items():
            f = seriler[kod]["sll"] - seriler["ETAS"]["sll"]
            g = seriler[kod]["sll"] - seriler["Arka plan"]["sll"]
            top = (seriler[kod]["sll"] + seriler[kod]["tll"]) - (seriler["ETAS"]["sll"] + seriler["ETAS"]["tll"])
            ozet["fark"][f"{te}|{kod}|{ad}"] = {
                "sll_eksi_etas": float(f[mk].mean()), "GA": boot(f, mk, bg),
                "tohum": [float((q - seriler["ETAS"]["sll"])[mk].mean()) for q in seriler[kod]["sll_tohum"]],
                "sll_eksi_arka_plan": float(g[mk].mean()), "toplam_eksi_etas": float(top[mk].mean()), "toplam_GA": boot(top, mk, bg),
                "tll_eksi_A0": float((seriler[kod]["tll"] - np.mean(a0, 0))[mk].mean())}

y = [ozet["fark"][f"{te}|S|tum_yil"] for te in TEST]
if all(q["sll_eksi_etas"] > 0 and q["GA"][0] > 0 for q in y):
    karar = "DAHA_IYI"
elif all(q["sll_eksi_etas"] < 0 and q["GA"][1] < 0 for q in y):
    karar = "DAHA_KOTU"
else:
    karar = "KARISIK"
ozet["karar"] = {"karar": karar, "zaman_bozuluyor_mu": {te: ozet["fark"][f"{te}|S|tum_yil"]["tll_eksi_A0"] < -0.01 for te in TEST}}
with open(KOK / "sonuclar/c10_ozet.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, indent=1, ensure_ascii=False)


def ga(q):
    return f"[{q[0]:+.3f}, {q[1]:+.3f}]"


sat = ["Uzaysal ETAS parametreleri: " + "; ".join(f"{te}: d={p['d_km2']:.2f} km², γ={p['gamma']:.3f}, ρ={p['rho']:.3f}, h={p['h_km']:.1f} km, uydurma SLL {p['uydurma_sll']:.4f}" for te, p in ozet["etas_uzay"].items()), ""]
for te in TEST:
    r = ozet["test"][te]
    sat += [f"### {te}", "", "| | SLL yıl (km²) | SLL ilk gün | TLL yıl | TLL ilk gün | en iyi adım |", "|---|---:|---:|---:|---:|---|"]
    for ad in ["Arka plan", "ETAS", "S", "S2", "Suzun"]:
        if ad not in r:
            continue
        q = r[ad]
        sat.append(f"| {ad} | {q['tum_yil']:.3f} | {q['ilk_gun']:.3f} | {q.get('tll_tum_yil', float('nan')):.3f} | {q.get('tll_ilk_gun', float('nan')):.3f} | {q.get('en_iyi_adim', '')} |")
    sat.append(f"| A0 (yalnız zaman) | | | {r['A0_tll']['tum_yil']:.3f} | {r['A0_tll']['ilk_gun']:.3f} | |")
    sat += ["", "| Kıyas | Küme | SLL − ETAS (GA) | tohumlar | SLL − arka plan | (TLL+SLL) − ETAS (GA) | TLL − A0 |", "|---|---|---|---|---:|---|---:|"]
    for k, q in ozet["fark"].items():
        if not k.startswith(te + "|"):
            continue
        _, kod, ad = k.split("|")
        sat.append(f"| {kod} | {ad} | {q['sll_eksi_etas']:+.3f} {ga(q['GA'])} | " + ", ".join(f"{z:+.3f}" for z in q["tohum"])
                   + f" | {q['sll_eksi_arka_plan']:+.3f} | {q['toplam_eksi_etas']:+.3f} {ga(q['toplam_GA'])} | {q['tll_eksi_A0']:+.4f} |")
    sat.append("")
sat += [f"**Karar (ön kayıtlı kural, model S): {karar}**; zaman tahmini bozuluyor mu (TLL − A0 < −0.01): {ozet['karar']['zaman_bozuluyor_mu']}"]
(KOK / "sonuclar/c10_tablolar.md").write_text("\n".join(sat) + "\n", encoding="utf-8")
print("\n".join(sat))
