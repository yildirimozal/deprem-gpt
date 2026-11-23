"""Güçlendirilmiş zamansal ETAS varyantları (ON_KAYIT_4.md). Yöntem kod/etas_zaman.py ile aynı
(profil olabilirlik: mu ve K EM ile, diğerleri Nelder-Mead), şu seçeneklerle:

  --kesik       Omori çekirdeği t < tau için (t + c)^(-p), sonra 0. tau ve p > 0 serbest.
  --n_max X     dallanma oranı n = K E[e^{a m}] int(çekirdek) <= X kısıtı (K sınırda tutulur).
  --ek npz      havuz: ek kataloglar da uydurma hedefi; K, a, c, p ortak, mu her katalog için ayrı.
                Tetikleme yalnızca aynı katalog içinde.

  lambda_k(t) = mu_k + K * sum_{j in k, t_j < t} exp(a m_j) h(t - t_j)
  TLL_i = log lambda(t_i) - integral_{t_(i-1)}^{t_i} lambda      (gün birimi)

Kullanım:
  python etas_guclu.py --ana turkiye.npz --uydur 1,2 --test 3 --ad etas4_E1 --kesik
Çıktı: sonuclar/<ad>/etas.json, etas_test.npz (ana katalogdaki test hedefleri)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--ana", required=True)
ap.add_argument("--mc", type=float, default=None)
ap.add_argument("--uydur", default="1,2")
ap.add_argument("--test", default="3")
ap.add_argument("--ek", default="")
ap.add_argument("--kesik", action="store_true")
ap.add_argument("--n_max", type=float, default=0.0)
ap.add_argument("--ad", required=True)
ap.add_argument("--parca", type=int, default=1024)
ap.add_argument("--nm_adim", type=int, default=400)
ap.add_argument("--sure_dk", type=float, default=0.0, help=">0 ise bu kadar dakikadan sonra sonuçsuz çık (ON_KAYIT_11)")
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

# ---------- veri: ana katalog + (isteğe bağlı) ek kataloglar, her biri bir segment ----------
v = np.load(KOK / "veri" / arg.ana)
mc = float(v["mc"]) if arg.mc is None else arg.mc
t_l, m_l, segler = [v["t_gun"].astype(np.float64)], [v["buyukluk"] - mc], [(0, len(v["t_gun"]))]
bolum = v["bolum"]
uy_l = [np.where(np.isin(bolum, [int(x) for x in arg.uydur.split(",")]))[0]]
test_idx = np.where(np.isin(bolum, [int(x) for x in arg.test.split(",")]))[0]
if arg.ek:
    e = np.load(KOK / "veri" / arg.ek)
    n = len(t_l[0])
    for a, b in zip(e["seg_bas"], e["seg_son"]):
        t_l.append(e["t_gun"][a:b].astype(np.float64))
        m_l.append(e["m_goreli"][a:b].astype(np.float64))
        segler.append((n, n + b - a))
        uy_l.append(np.arange(n + 1, n + b - a))          # segmentin ilk olayı hedef olamaz
        n += b - a
t = torch.tensor(np.concatenate(t_l), device=cihaz)
mg = torch.tensor(np.concatenate(m_l), device=cihaz)
for a, b in segler:
    assert np.all(np.diff(np.concatenate(t_l)[a:b]) >= 0)
assert uy_l[0][0] > 0 and test_idx[0] > 0
n_kat = len(segler)


def parcalar(hedef_l):
    """[(segment no, segment başı, hedef dizisi)] ; parça segment sınırını aşmaz."""
    out = []
    for k, idx in enumerate(hedef_l):
        for s in range(0, len(idx), arg.parca):
            out.append((k, segler[k][0], idx[s:s + arg.parca]))
    return out


UY = parcalar(uy_l)
TE = parcalar([test_idx] + [np.array([], int)] * (n_kat - 1))
uy_kat = torch.tensor(np.concatenate([np.full(len(x), k) for k, x in enumerate(uy_l)]), device=cihaz)
m_uy_np = np.concatenate([np.concatenate(m_l)[x] for x in uy_l])


def F(x, c, p):
    """int_0^x (s + c)^(-p) ds, p = 1 yakınında kararlı."""
    q = 1.0 - p
    if abs(q) < 1e-9:
        q = 1e-9
    return c ** q * torch.expm1(q * torch.log1p(x / c)) / q


@torch.no_grad()
def toplamlar(a, c, p, tau, plist):
    S, I, D = [], [], []
    for k, j0, i in plist:
        if len(i) == 0:
            continue
        jmax = int(i[-1])
        ii = torch.as_tensor(i, device=cihaz)
        ti, onc = t[ii], t[ii - 1]
        tj = t[j0:jmax]
        g = torch.exp(a * mg[j0:jmax])
        maske = torch.arange(j0, jmax, device=cihaz)[None, :] < ii[:, None]
        d = (ti[:, None] - tj[None, :]).clamp_min(0)
        dp = (onc[:, None] - tj[None, :]).clamp_min(0)
        if tau is None:
            cek = torch.where(maske, (d + c) ** (-p), 0.0)
            Fark = F(d, c, p) - F(dp, c, p)
        else:
            cek = torch.where(maske & (d < tau), (d + c) ** (-p), 0.0)
            Fark = F(d.clamp_max(tau), c, p) - F(dp.clamp_max(tau), c, p)
        S.append(cek @ g)
        I.append(torch.where(maske, Fark, 0.0) @ g)
        D.append(ti - onc)
    return torch.cat(S), torch.cat(I), torch.cat(D)


def cekirdek_integrali(c, p, tau):
    if tau is None:
        return c ** (1 - p) / (p - 1)
    return float(F(torch.tensor(tau, dtype=torch.float64), c, p))


SON = {}      # sıcak başlangıç: bir önceki değerlendirmenin (mu, K) çözümü


@torch.no_grad()
def em(S, I, D, kat, K_max=None, n_it=3000):
    """EM (monoton). Ortalama TLL artışı 20 yinelemede 1e-10'dan küçükse durur."""
    sD = torch.bincount(kat, weights=D, minlength=n_kat).clamp_min(1e-12)
    sI = I.sum()
    if "mu" in SON:
        mu, K = SON["mu"].clone(), torch.tensor(SON["K"], dtype=torch.float64, device=cihaz)
    else:
        mu = 0.5 * torch.bincount(kat, minlength=n_kat).double() / sD
        K = 0.5 * len(D) / sI
    K_ust = torch.tensor(K_max if K_max is not None else float("inf"), dtype=torch.float64, device=cihaz)
    K = torch.minimum(K, K_ust)
    onceki = None
    for it in range(n_it):
        lam = mu[kat] + K * S
        mu = torch.bincount(kat, weights=mu[kat] / lam, minlength=n_kat) / sD
        K = torch.minimum((K * S / lam).sum() / sI, K_ust)
        if it % 20 == 19:
            ll = float((torch.log(mu[kat] + K * S) - mu[kat] * D - K * I).mean())
            if onceki is not None and ll - onceki < 1e-10:
                break
            onceki = ll
    SON["mu"], SON["K"] = mu, float(K)
    return mu, float(K)


def ac(x):
    a, c = float(x[0]), float(np.exp(x[1]))
    p = float(np.exp(x[2])) if arg.kesik else 1.0 + float(np.exp(x[2]))
    tau = float(np.exp(x[3])) if arg.kesik else None
    return a, c, p, tau


cache = {}


def amac(x):
    if arg.sure_dk > 0 and time.time() - bas > 60 * arg.sure_dk:
        print("SURE_ASIMI: tamamlanmadi", flush=True)
        raise SystemExit(3)
    key = tuple(np.round(x, 10))
    if key in cache:
        return cache[key][0]
    a, c, p, tau = ac(x)
    gecerli = -1.0 < a < 6.0 and 1e-7 < c < 10.0 and (0.2 < p < 4.0 if arg.kesik else 1.0 + 1e-5 < p < 4.0)
    if tau is not None:
        gecerli = gecerli and 0.01 < tau < 1e5
    if not gecerli:
        cache[key] = (1e9, None)
        return 1e9
    S, I, D = toplamlar(a, c, p, tau, UY)
    K_max = None
    if arg.n_max > 0:
        K_max = arg.n_max / (float(np.mean(np.exp(a * m_uy_np))) * cekirdek_integrali(c, p, tau))
    mu, K = em(S, I, D, uy_kat, K_max)
    tll = torch.log(mu[uy_kat] + K * S) - mu[uy_kat] * D - K * I
    deger = -float(tll.mean())
    if not np.isfinite(deger):
        deger = 1e9
    cache[key] = (deger, (mu.cpu().tolist(), K))
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
if arg.kesik:
    baslangiclar = [[1.5, np.log(0.01), np.log(1.1), np.log(365.0)], [2.0, np.log(0.003), np.log(1.0), np.log(3650.0)],
                    [1.0, np.log(0.03), np.log(0.9), np.log(30.0)]]
    adim = [0.3, 0.7, 0.2, 1.0]
else:
    baslangiclar = [[1.5, np.log(0.01), np.log(0.1)], [2.0, np.log(0.003), np.log(0.2)], [1.0, np.log(0.03), np.log(0.05)]]
    adim = [0.3, 0.7, 0.5]
sonuc = []
for b in baslangiclar:
    SON.clear()
    x, fx, it = nelder_mead(amac, b, adim, arg.nm_adim)
    a, c, p, tau = ac(x)
    mu, K = cache[tuple(np.round(x, 10))][1]
    n_dal = K * float(np.mean(np.exp(a * m_uy_np))) * cekirdek_integrali(c, p, tau)
    sonuc.append({"baslangic": [float(y) for y in b], "uydurma_tll": -fx, "mu": mu, "K": K, "a": a, "c": c, "p": p,
                  "tau": tau, "dallanma": n_dal, "x": list(map(float, x)), "nm_adim": it})
    print(json.dumps(sonuc[-1]), f"{time.time() - bas:.0f}s", flush=True)
en = max(sonuc, key=lambda r: r["uydurma_tll"])
S, I, D = toplamlar(en["a"], en["c"], en["p"], en["tau"], TE)
mu0 = en["mu"][0]
tt = (torch.log(mu0 + en["K"] * S) - mu0 * D - en["K"] * I).cpu().numpy()
cikti = KOK / "sonuclar" / arg.ad
cikti.mkdir(parents=True, exist_ok=True)
np.savez(cikti / "etas_test.npz", idx=test_idx, tll=tt)
json.dump({"arg": vars(arg), "mc": mc, "segment": [[int(x) for x in sg] for sg in segler], "n_uydurma": int(sum(len(x) for x in uy_l)),
           "n_test": int(len(test_idx)), "denemeler": sonuc, "secilen": en, "test_tll": float(tt.mean()),
           "sure_s": time.time() - bas}, open(cikti / "etas.json", "w"), indent=1)
print(f"TEST TLL {tt.mean():.4f}  n={len(tt)}  dallanma={en['dallanma']:.3f}", flush=True)
print("IS_BITTI", flush=True)
