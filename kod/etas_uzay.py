"""Uzaysal ETAS (ON_KAYIT_10.md). Zaman bileşeni sabit (Türkiye E2, Japonya E5); uzay çekirdeği ve arka plan
bant genişliği uydurulur. GPU, yoğun matrislerle.

    p(x_i | t_i) = [mu u(x_i) + Σ_j g_j(t_i) f(x_i - x_j | m_j)] / [mu + Σ_j g_j(t_i)]
    f(r | m) = (ρ/π) D^ρ / (r² + D)^(1+ρ),   D = d e^(γ m)          (r km)
    u(x) = eğitim olaylarının izotropik Gauss çekirdek yoğunluğu (bant h km, uydurmada kendisi hariç)

Kullanım: python etas_uzay.py --ana turkiye.npz --zaman etas4_E2 --ad etas10_TR
Çıktı: sonuclar/<ad>/etas_uzay.json, etas_uzay_test.npz (sll km² başına, sll_arka_plan, idx)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--ana", required=True)
ap.add_argument("--zaman", required=True)
ap.add_argument("--ad", required=True)
ap.add_argument("--n_uydur", type=int, default=5000)
ap.add_argument("--nm_adim", type=int, default=300)
ap.add_argument("--cihaz", default="auto", choices=["auto", "cuda", "mps", "cpu"])
arg = ap.parse_args()

if arg.cihaz == "auto":
    if torch.cuda.is_available():
        cihaz = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        cihaz = torch.device("mps")
    else:
        cihaz = torch.device("cpu")
else:
    cihaz = torch.device(arg.cihaz)

v = np.load(KOK / "veri" / arg.ana)
t = v["t_gun"].astype(np.float64)
m = v["buyukluk"] - float(v["mc"])
bolum = v["bolum"]
la, lo = v["enlem"], v["boylam"]
la0, lo0 = np.median(la), np.median(lo)
x = (lo - lo0) * np.cos(np.radians(la0)) * 111.2
y = (la - la0) * 111.2
s = json.load(open(KOK / f"sonuclar/{arg.zaman}/etas.json"))["secilen"]
a, c, p, K = s["a"], s["c"], s["p"], s["K"]
tau = s.get("tau")
mu = s["mu"][0] if isinstance(s["mu"], list) else s["mu"]

rng = np.random.default_rng(0)
egit = np.where(bolum == 1)[0]
uydur_idx = np.sort(rng.choice(egit[egit > 0], arg.n_uydur, replace=False))
test_idx = np.where(bolum == 3)[0]
bg_idx = egit
T = torch.tensor(t, device=cihaz)
XY = torch.tensor(np.stack([x, y], 1), device=cihaz)
Mg = torch.tensor(m, device=cihaz)
XYb = XY[torch.tensor(bg_idx, device=cihaz)]
bg_t = torch.tensor(bg_idx, device=cihaz)


def hazirla(idx, parca=500):
    """Her parça için: G (tetikleme ağırlıkları, j < i), R2 (uzaklık kare), B2 (arka plan uzaklık kare), öz maske."""
    out = []
    for k in range(0, len(idx), parca):
        ii = torch.tensor(idx[k:k + parca], device=cihaz)
        jmax = int(ii.max())
        d = (T[ii][:, None] - T[None, :jmax])
        maske = torch.arange(jmax, device=cihaz)[None, :] < ii[:, None]
        if tau is not None:
            maske = maske & (d < tau)
        G = torch.where(maske, K * torch.exp(a * Mg[:jmax])[None, :] * (d.clamp_min(0) + c) ** (-p), 0.0).float()
        R2 = ((XY[ii][:, None, :] - XY[None, :jmax, :]) ** 2).sum(-1).float()
        B2 = ((XY[ii][:, None, :] - XYb[None]) ** 2).sum(-1).float()
        oz = (bg_t[None, :] == ii[:, None])
        out.append((G, R2, B2, oz, jmax))
    return out


def sll(theta, veri, ayri=False):
    d_, g_, rho, h = math.exp(theta[0]), theta[1], math.exp(theta[2]), math.exp(theta[3])
    sonuc, arka = [], []
    for G, R2, B2, oz, jmax in veri:
        D = (d_ * torch.exp(g_ * Mg[:jmax])).float()[None, :]
        f = (rho / math.pi) * D ** rho / (R2 + D) ** (1 + rho)
        ker = torch.where(oz, 0.0, torch.exp(-B2 / (2 * h * h)))
        n_bg = (~oz).sum(1).double()
        u = ker.double().sum(1) / n_bg / (2 * math.pi * h * h)
        pay = mu * u + (G.double() * f.double()).sum(1)
        payda = mu + G.double().sum(1)
        sonuc.append(torch.log(pay) - torch.log(payda))
        arka.append(torch.log(u))
    if ayri:
        return torch.cat(sonuc), torch.cat(arka)
    return float(torch.cat(sonuc).mean())


def nelder_mead(f, x0, adim, n_adim):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[k] * adim[k] for k in range(n)]
    fv = [f(z) for z in pts]
    it = 0
    for it in range(n_adim):
        o = np.argsort(fv)
        pts, fv = [pts[k] for k in o], [fv[k] for k in o]
        if abs(fv[-1] - fv[0]) < 1e-7 and it > 20:
            break
        merkez = np.mean(pts[:-1], 0)
        xr = merkez + (merkez - pts[-1])
        fr = f(xr)
        if fr < fv[0]:
            xe = merkez + 2 * (merkez - pts[-1])
            fe = f(xe)
            pts[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fv[-2]:
            pts[-1], fv[-1] = xr, fr
        else:
            xc = merkez + 0.5 * (pts[-1] - merkez)
            fc = f(xc)
            if fc < fv[-1]:
                pts[-1], fv[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (z - pts[0]) for z in pts[1:]]
                fv = [fv[0]] + [f(z) for z in pts[1:]]
    k = int(np.argmin(fv))
    return pts[k], fv[k], it


bas = time.time()
with torch.no_grad():
    veri_u = hazirla(uydur_idx)

    def amac(th):
        if not (-8 < th[0] < 8 and -1 < th[1] < 4 and -4 < th[2] < 2 and 0 < th[3] < 6.5):
            return 1e9
        val = sll(th, veri_u)
        return -val if np.isfinite(val) else 1e9

    sonuclar = []
    for b in [[0.0, 1.2, math.log(0.6), math.log(20)], [math.log(4), 1.8, 0.0, math.log(50)], [math.log(0.25), 0.8, math.log(0.4), math.log(10)]]:
        th, fx, it = nelder_mead(amac, b, [1.0, 0.3, 0.4, 0.5], arg.nm_adim)
        sonuclar.append({"baslangic": b, "uydurma_sll": -fx, "d_km2": math.exp(th[0]), "gamma": th[1], "rho": math.exp(th[2]),
                         "h_km": math.exp(th[3]), "theta": list(map(float, th)), "nm_adim": it})
        print(json.dumps(sonuclar[-1]), f"{time.time() - bas:.0f}s", flush=True)
    en = max(sonuclar, key=lambda r: r["uydurma_sll"])
    del veri_u
    if cihaz.type == "cuda":
        torch.cuda.empty_cache()
    veri_t = hazirla(test_idx)
    s_t, a_t = sll(en["theta"], veri_t, ayri=True)
cikti = KOK / "sonuclar" / arg.ad
cikti.mkdir(parents=True, exist_ok=True)
np.savez(cikti / "etas_uzay_test.npz", idx=test_idx, sll=s_t.cpu().numpy(), sll_arka_plan=a_t.cpu().numpy())
json.dump({"arg": vars(arg), "zaman_param": {"a": a, "c": c, "p": p, "K": K, "tau": tau, "mu": mu}, "merkez": [float(la0), float(lo0)],
           "denemeler": sonuclar, "secilen": en, "test_sll_km2": float(s_t.mean()), "test_sll_arka_plan_km2": float(a_t.mean()),
           "sure_s": time.time() - bas}, open(cikti / "etas_uzay.json", "w"), indent=1)
print(f"TEST SLL (km²) {float(s_t.mean()):.4f}  arka plan {float(a_t.mean()):.4f}", flush=True)
print("IS_BITTI", flush=True)
