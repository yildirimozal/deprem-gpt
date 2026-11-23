"""Eksik kaydı modelleyen zamansal ETAS (ON_KAYIT_6.md). GPU, float64.

    λ(t)       = mu + K Σ_{j<i} e^{a m_j} (t - t_j + c)^(-p)                (tam süreç, Mc üstü)
    λ_kayıt(t) = P(t) λ(t)
    P(t)       = ∫_{Mc}^∞ GR(m) Φ((m - μ_d(t)) / σ) dm                      (Ogata-Katsura algılama)
               = Φ(A/σ) + exp(βA + β²σ²/2) (1 - Φ(A/σ + βσ)),   A = Mc - μ_d(t)
    μ_d(t)     = max( Mc - 2, max_{j: t_j < t, M_j >= 5} [M_j - G - H log10(t - t_j)] )   (Helmstetter 2006)

TLL_i = log P(t_i) + log λ(t_i) - ∫_{t_(i-1)}^{t_i} P λ ds. İntegral, P(t_(i-1)+) < 0.9999 olan aralıklarda
log-zamanda 32 düğümlü Gauss-Legendre ile, diğerlerinde kapalı biçimle (P = 1) alınır. Profil olabilirlik:
mu ve K EM ile (λ_kayıt mu ve K'da doğrusal), diğerleri Nelder-Mead ile.

Kullanım:
  python etas_eksik.py --ad etas6_E6 --n_max 0.99                      (G = 4.5, H = 0.75 sabit)
  python etas_eksik.py --ad etas6_E7 --n_max 0.99 --GH_uydur
  python etas_eksik.py --ad etas6_E8 --uydur 3 --GH_uydur               (kâhin)
  python etas_eksik.py --ad etas6_kapali --n_max 0.99 --G 100           (denetim: E2'yi üretmeli)
Çıktı: sonuclar/<ad>/etas.json, etas_test.npz (tll, log_P, log_lam, Lam, mu_d_goreli)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--ana", default="turkiye.npz")
ap.add_argument("--uydur", default="1,2")
ap.add_argument("--test", default="3")
ap.add_argument("--G", type=float, default=4.5)
ap.add_argument("--H", type=float, default=0.75)
ap.add_argument("--GH_uydur", action="store_true")
ap.add_argument("--sigma", type=float, default=0.2)
ap.add_argument("--n_max", type=float, default=0.0)
ap.add_argument("--buyuk_esik", type=float, default=5.0)
ap.add_argument("--ad", required=True)
ap.add_argument("--nm_adim", type=int, default=400)
ap.add_argument("--sure_dk", type=float, default=0.0, help=">0 ise bu kadar dakikadan sonra sonuçsuz çık (ON_KAYIT_11)")
arg = ap.parse_args()
cihaz = torch.device("cuda")
dd = torch.float64

v = np.load(KOK / "veri" / arg.ana)
mc = float(v["mc"])
t_np = v["t_gun"].astype(np.float64)
m_np = v["buyukluk"] - mc
bolum = v["bolum"]
me = m_np[bolum == 1]
BETA = 1.0 / (me.mean() + 0.05)                      # Aki-Utsu, 0.1 bölme (analiz5.py ile aynı)
t = torch.tensor(t_np, dtype=dd, device=cihaz)
mg = torch.tensor(m_np, dtype=dd, device=cihaz)
buyuk = np.where(v["buyukluk"] >= arg.buyuk_esik)[0]
tb = torch.tensor(t_np[buyuk], dtype=dd, device=cihaz)
mb = torch.tensor(m_np[buyuk], dtype=dd, device=cihaz)
uy_idx = np.where(np.isin(bolum, [int(x) for x in arg.uydur.split(",")]))[0]
te_idx = np.where(np.isin(bolum, [int(x) for x in arg.test.split(",")]))[0]
SIG = arg.sigma
DELTA = 1e-5
gl_x, gl_w = np.polynomial.legendre.leggauss(32)
GLX = torch.tensor(gl_x, dtype=dd, device=cihaz)
GLW = torch.tensor(gl_w, dtype=dd, device=cihaz)
LN10 = np.log(10.0)


def mu_d(s, G, H):
    """s: (n,) zamanlar. Göreli algılama eşiği (Mc'ye göre)."""
    fark = s[:, None] - tb[None, :]
    once = fark > 0
    deger = mb[None, :] - G - H * torch.log(fark.clamp_min(1e-12)) / LN10
    deger = torch.where(once, deger, torch.tensor(-1e9, dtype=dd, device=cihaz))
    return torch.clamp(deger.max(1).values, min=-2.0)


def log_P(mud):
    A = -mud
    z = A / SIG
    t1 = torch.special.log_ndtr(z)
    t2 = BETA * A + 0.5 * BETA ** 2 * SIG ** 2 + torch.special.log_ndtr(-(z + BETA * SIG))
    return torch.logaddexp(t1, t2)


def F(x, c, p):
    q = 1.0 - p
    return c ** q * torch.expm1(q * torch.log1p(x / c)) / q


@torch.no_grad()
def cekirdek_toplam(s, jcut, a, c, p, parca=512):     # Japonya'da 2048 ile GPU belleği yetmedi
    """s: (n,), jcut: (n,) -> Σ_{j < jcut} e^{a m_j} (s - t_j + c)^(-p)."""
    out = []
    for k in range(0, len(s), parca):
        ss, jj = s[k:k + parca], jcut[k:k + parca]
        jmax = int(jj.max())
        g = torch.exp(a * mg[:jmax])
        d = (ss[:, None] - t[None, :jmax]).clamp_min(0)
        maske = torch.arange(jmax, device=cihaz)[None, :] < jj[:, None]
        out.append(torch.where(maske, (d + c) ** (-p), 0.0) @ g)
    return torch.cat(out)


@torch.no_grad()
def toplamlar(a, c, p, G, H, idx):
    """Her hedef için S_i, D'_i = ∫P ds, I'_i = ∫P Σ g h ds, log P(t_i), μ_d(t_i)."""
    S, D, I, LP, MD = [], [], [], [], []
    for k in range(0, len(idx), 1024):
        i = idx[k:k + 1024]
        jmax = int(i[-1])
        ii = torch.as_tensor(i, device=cihaz)
        ti, onc = t[ii], t[ii - 1]
        g = torch.exp(a * mg[:jmax])
        maske = torch.arange(jmax, device=cihaz)[None, :] < ii[:, None]
        d = (ti[:, None] - t[None, :jmax]).clamp_min(0)
        dp = (onc[:, None] - t[None, :jmax]).clamp_min(0)
        S.append(torch.where(maske, (d + c) ** (-p), 0.0) @ g)
        Ian = torch.where(maske, F(d, c, p) - F(dp, c, p), 0.0) @ g
        Dan = ti - onc
        md = mu_d(ti, G, H)
        LP.append(log_P(md))
        MD.append(md)
        # eksik aralıklar: P(t_(i-1)+) < 0.9999
        p0 = torch.exp(log_P(mu_d(onc + DELTA, G, H)))
        eks = torch.where(p0 < 0.9999)[0]
        if len(eks):
            de = (ti - onc)[eks]
            ulo, uhi = np.log(DELTA), torch.log(de + DELTA)
            u = (uhi[:, None] - ulo) * (GLX[None, :] + 1) / 2 + ulo          # (n_eks, 32)
            w = GLW[None, :] * (uhi[:, None] - ulo) / 2 * torch.exp(u)
            s = onc[eks][:, None] + torch.exp(u) - DELTA
            ps = torch.exp(log_P(mu_d(s.flatten(), G, H))).view_as(s)
            jc = ii[eks][:, None].expand_as(s).flatten()
            Ss = cekirdek_toplam(s.flatten(), jc, a, c, p).view_as(s)
            Dan = Dan.clone()
            Ian = Ian.clone()
            Dan[eks] = (w * ps).sum(1)
            Ian[eks] = (w * ps * Ss).sum(1)
        D.append(Dan)
        I.append(Ian)
    return torch.cat(S), torch.cat(D), torch.cat(I), torch.cat(LP), torch.cat(MD)


SON = {}


@torch.no_grad()
def em(S, I, D, K_max=None, n_it=3000):
    sD, sI = D.sum(), I.sum()
    if "mu" in SON:
        mu, K = SON["mu"], SON["K"]
    else:
        mu, K = 0.5 * len(D) / float(sD), 0.5 * len(D) / float(sI)
    K = min(K, K_max) if K_max is not None else K
    onceki = None
    for it in range(n_it):
        lam = mu + K * S
        mu = float((mu / lam).sum() / sD)
        K = float((K * S / lam).sum() / sI)
        if K_max is not None:
            K = min(K, K_max)
        if it % 20 == 19:
            ll = float((torch.log(mu + K * S) - mu * D - K * I).mean())
            if onceki is not None and ll - onceki < 1e-10:
                break
            onceki = ll
    SON["mu"], SON["K"] = mu, K
    return mu, K


m_uy = m_np[uy_idx]


def ac(x):
    a, c, p = float(x[0]), float(np.exp(x[1])), 1.0 + float(np.exp(x[2]))
    G, H = (float(x[3]), float(x[4])) if arg.GH_uydur else (arg.G, arg.H)
    return a, c, p, G, H


cache = {}


def amac(x):
    if arg.sure_dk > 0 and time.time() - bas > 60 * arg.sure_dk:
        print("SURE_ASIMI: tamamlanmadi", flush=True)
        raise SystemExit(3)
    key = tuple(np.round(x, 10))
    if key in cache:
        return cache[key][0]
    a, c, p, G, H = ac(x)
    gh_gecerli = (1.0 < G < 9.0 and 0.05 < H < 3.0) if arg.GH_uydur else True
    if not (-1.0 < a < 6.0 and 1e-7 < c < 10.0 and 1.0 + 1e-5 < p < 4.0 and gh_gecerli):
        cache[key] = (1e9, None)
        return 1e9
    S, D, I, LP, _ = toplamlar(a, c, p, G, H, uy_idx)
    K_max = arg.n_max / (float(np.mean(np.exp(a * m_uy))) * c ** (1 - p) / (p - 1)) if arg.n_max > 0 else None
    mu, K = em(S, I, D, K_max)
    tll = LP + torch.log(mu + K * S) - mu * D - K * I
    deger = -float(tll.mean())
    if not np.isfinite(deger):
        deger = 1e9
    cache[key] = (deger, (mu, K))
    return deger


def nelder_mead(f, x0, adim, n_adim):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[k] * adim[k] for k in range(n)]
    fv = [f(x) for x in pts]
    it = 0
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
taban = [[0.986, np.log(0.036), np.log(0.189)], [1.5, np.log(0.01), np.log(0.1)], [1.0, np.log(0.03), np.log(0.05)]]
adim = [0.3, 0.7, 0.5]
if arg.GH_uydur:
    taban = [b + gh for b, gh in zip(taban, [[4.5, 0.75], [4.0, 1.0], [5.0, 0.5]])]
    adim = adim + [0.5, 0.25]
sonuc = []
for b in taban:
    SON.clear()
    x, fx, it = nelder_mead(amac, b, adim, arg.nm_adim)
    a, c, p, G, H = ac(x)
    mu, K = cache[tuple(np.round(x, 10))][1]
    n_dal = K * float(np.mean(np.exp(a * m_uy))) * c ** (1 - p) / (p - 1)
    sonuc.append({"baslangic": [float(y) for y in b], "uydurma_tll": -fx, "mu": mu, "K": K, "a": a, "c": c, "p": p,
                  "G": G, "H": H, "dallanma": n_dal, "x": [float(y) for y in x], "nm_adim": int(it)})
    print(json.dumps(sonuc[-1]), f"{time.time() - bas:.0f}s", flush=True)
en = max(sonuc, key=lambda r: r["uydurma_tll"])
S, D, I, LP, MD = toplamlar(en["a"], en["c"], en["p"], en["G"], en["H"], te_idx)
log_lam = torch.log(en["mu"] + en["K"] * S)
Lam = en["mu"] * D + en["K"] * I
tll = LP + log_lam - Lam
cikti = KOK / "sonuclar" / arg.ad
cikti.mkdir(parents=True, exist_ok=True)
np.savez(cikti / "etas_test.npz", idx=te_idx, tll=tll.cpu().numpy(), log_P=LP.cpu().numpy(),
         log_lam=log_lam.cpu().numpy(), Lam=Lam.cpu().numpy(), mu_d_goreli=MD.cpu().numpy())
with open(cikti / "etas.json", "w", encoding="utf-8") as f:
    json.dump({"arg": vars(arg), "mc": mc, "beta": BETA, "b": BETA / LN10, "n_buyuk_olay": int(len(buyuk)),
               "n_uydurma": int(len(uy_idx)), "n_test": int(len(te_idx)), "denemeler": sonuc, "secilen": en,
               "test_tll": float(tll.mean()), "sure_s": time.time() - bas}, f, indent=1)
print(f"TEST TLL {float(tll.mean()):.4f}  n={len(te_idx)}  dallanma={en['dallanma']:.3f}  G={en['G']:.2f} H={en['H']:.2f}", flush=True)
print("IS_BITTI", flush=True)
