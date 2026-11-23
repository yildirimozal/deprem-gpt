"""Zamansal ETAS (Ogata 1988), en büyük olabilirlik kestirimi, GPU üzerinde float64.

    lambda(t) = mu + K * sum_{j: t_j < t} exp(a (m_j - Mc)) (t - t_j + c)^(-p)

Olay başına zamansal log-olabilirlik, EarthquakeNPP'deki tanımla aynı:
    TLL_i = log lambda(t_i) - integral_{t_(i-1)}^{t_i} lambda(s) ds      (gün birimi)
Uydurma, uydurma hedeflerindeki TLL toplamını en büyükler. O hedeflerden önceki tüm olaylar geçmiş
olarak lambda'ya girer. Test hedefleri aynı formülle, tüm geçmişle değerlendirilir.

Yöntem (profil olabilirlik): (a, c, p) sabitken her hedef için
    S_i = sum_j e^{a m_j} (d_ij + c)^(-p),   I_i = sum_j e^{a m_j} [F(d_ij) - F(d'_ij)]
bir kez hesaplanır. TLL_i = log(mu + K S_i) - mu Δ_i - K I_i, (mu, K) içinde içbükeydir ve
EM güncellemesiyle (mu <- sum(mu/λ_i)/sum Δ_i, K <- sum(K S_i/λ_i)/sum I_i) en büyüklenir.
Dış üç parametre (a, log c, log(p-1)) Nelder-Mead ile aranır. Otomatik türev kullanılmaz.

Kullanım:
  python etas_zaman.py --veri olaylar.npz --mc 2.5 --uydur 1,2 --test 3 --ad etas_ca
Girdi npz: t_gun, buyukluk (mutlak), bolum. Çıktı: sonuclar/<ad>/etas.json, etas_test.npz
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--veri", required=True)
ap.add_argument("--mc", type=float, default=None, help="verilmezse npz içindeki mc")
ap.add_argument("--uydur", default="1,2")
ap.add_argument("--test", default="3")
ap.add_argument("--ad", required=True)
ap.add_argument("--parca", type=int, default=1024)
ap.add_argument("--nm_adim", type=int, default=300)
arg = ap.parse_args()

cihaz = torch.device("cuda")
v = np.load(KOK / "veri" / arg.veri)
t = torch.tensor(v["t_gun"], dtype=torch.float64, device=cihaz)
arg.mc = float(v["mc"]) if arg.mc is None else arg.mc
mg = torch.tensor(v["buyukluk"] - arg.mc, dtype=torch.float64, device=cihaz)
bolum = v["bolum"]
assert np.all(np.diff(v["t_gun"]) >= 0)
uydur_idx = np.where(np.isin(bolum, [int(x) for x in arg.uydur.split(",")]))[0]
test_idx = np.where(np.isin(bolum, [int(x) for x in arg.test.split(",")]))[0]
assert uydur_idx[0] > 0 and test_idx[0] > 0


@torch.no_grad()
def toplamlar(a, c, p, idx):
    """Her hedef için S_i, I_i, Δ_i (K ve mu hariç)."""
    S, I, D = [], [], []
    q = 1.0 - p
    cq = c ** q
    for s in range(0, len(idx), arg.parca):
        i = idx[s:s + arg.parca]
        jmax = int(i[-1])
        ii = torch.as_tensor(i, device=cihaz)
        ti, onc = t[ii], t[ii - 1]
        tj = t[:jmax]
        g = torch.exp(a * mg[:jmax])
        maske = torch.arange(jmax, device=cihaz)[None, :] < ii[:, None]
        d = (ti[:, None] - tj[None, :]).clamp_min(0)
        dp = (onc[:, None] - tj[None, :]).clamp_min(0)
        S.append(torch.where(maske, (d + c) ** (-p), 0.0) @ g)
        Fark = ((dp + c) ** q - (d + c) ** q) / (p - 1.0)          # F(d) - F(d')
        I.append(torch.where(maske, Fark, 0.0) @ g)
        D.append(ti - onc)
    return torch.cat(S), torch.cat(I), torch.cat(D)


@torch.no_grad()
def em(S, I, D, mu=None, K=None, n=500):
    mu = mu or 0.5 * len(D) / float(D.sum())
    K = K or 0.5 * len(D) / float(I.sum())
    sD, sI = float(D.sum()), float(I.sum())
    for _ in range(n):
        lam = mu + K * S
        mu_y = float((mu / lam).sum()) / sD
        K_y = float((K * S / lam).sum()) / sI
        if abs(mu_y - mu) < 1e-12 * mu and abs(K_y - K) < 1e-12 * K:
            mu, K = mu_y, K_y
            break
        mu, K = mu_y, K_y
    return mu, K


def tll(mu, K, S, I, D):
    return torch.log(mu + K * S) - mu * D - K * I


def ac(x):
    return float(x[0]), float(np.exp(x[1])), 1.0 + float(np.exp(x[2]))


cache = {}


def amac(x):
    """Dış parametreler için -ortalama TLL (mu, K profilden)."""
    key = tuple(np.round(x, 10))
    if key in cache:
        return cache[key][0]
    a, c, p = ac(x)
    if not (-1.0 < a < 6.0 and 1e-7 < c < 10.0 and 1.0 + 1e-5 < p < 4.0):
        cache[key] = (1e9, None)
        return 1e9
    S, I, D = toplamlar(a, c, p, uydur_idx)
    mu, K = em(S, I, D)
    deger = -float(tll(mu, K, S, I, D).mean())
    if not np.isfinite(deger):
        deger = 1e9
    cache[key] = (deger, (mu, K))
    return deger


def nelder_mead(f, x0, adim, n_adim):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[k] * adim[k] for k in range(n)]
    fv = [f(x) for x in pts]
    for it in range(n_adim):
        o = np.argsort(fv)
        pts, fv = [pts[k] for k in o], [fv[k] for k in o]
        if abs(fv[-1] - fv[0]) < 1e-9 and it > 20:
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
                pts = [pts[0]] + [pts[0] + 0.5 * (x - pts[0]) for x in pts[1:]]
                fv = [fv[0]] + [f(x) for x in pts[1:]]
    k = int(np.argmin(fv))
    return pts[k], fv[k], it


bas = time.time()
baslangiclar = [[1.5, np.log(0.01), np.log(0.1)], [2.0, np.log(0.003), np.log(0.2)], [1.0, np.log(0.03), np.log(0.05)]]
sonuc = []
for b in baslangiclar:
    x, fx, it = nelder_mead(amac, b, [0.3, 0.7, 0.5], arg.nm_adim)
    a, c, p = ac(x)
    mu, K = cache[tuple(np.round(x, 10))][1]
    sonuc.append({"baslangic": b, "uydurma_tll": -fx, "mu": mu, "K": K, "a": a, "c": c, "p": p,
                  "x": list(map(float, x)), "nm_adim": it})
    print(json.dumps(sonuc[-1]), f"{time.time() - bas:.0f}s", flush=True)
en = max(sonuc, key=lambda r: r["uydurma_tll"])
S, I, D = toplamlar(en["a"], en["c"], en["p"], test_idx)
tt = tll(en["mu"], en["K"], S, I, D).cpu().numpy()
# dallanma oranı (Mc üstü, b-değeri katalogdan): n = K * E[e^{a m}] * c^{1-p}/(p-1)
m_uy = v["buyukluk"][uydur_idx] - arg.mc
dallanma = en["K"] * float(np.mean(np.exp(en["a"] * m_uy))) * en["c"] ** (1 - en["p"]) / (en["p"] - 1)
cikti = KOK / "sonuclar" / arg.ad
cikti.mkdir(parents=True, exist_ok=True)
np.savez(cikti / "etas_test.npz", idx=test_idx, tll=tt)
with open(cikti / "etas.json", "w", encoding="utf-8") as f:
    json.dump({"arg": vars(arg), "n_uydurma": int(len(uydur_idx)), "n_test": int(len(test_idx)), "denemeler": sonuc,
               "secilen": en, "dallanma_orani_yaklasik": dallanma, "test_tll": float(tt.mean()), "sure_s": time.time() - bas},
              f, indent=1)
print(f"TEST TLL {tt.mean():.4f}  n={len(tt)}  dallanma~{dallanma:.3f}", flush=True)
print("IS_BITTI", flush=True)
