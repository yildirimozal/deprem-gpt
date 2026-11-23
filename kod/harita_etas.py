"""Haritalı artçı tahmini için uzaysal ETAS ileri benzetimi (ON_KAYIT_10.md, ikincil).

Ana şoktan 12 saat sonra, 24 saatlik ufuk. Zaman: sabit zamansal ETAS (Türkiye E2, Japonya E5); uzay:
etas_uzay.py'nin uydurduğu çekirdek f(r|m) ve arka plan (eğitim olaylarından örnek + Gauss h).
Çocuk konumu = ebeveyn + r (cos θ, sin θ), r çekirdeğin ters CDF'inden: r = sqrt(D ((1-u)^(-1/ρ) - 1)).

Mac'te: /usr/bin/python3 kod/harita_etas.py TR | JP
Çıktı: sonuclar/c10_<TR|JP>_harita/senaryo_konum_etas.npz (x, y km, katalog merkezine göre; m; senaryo)
"""
import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent
TE = sys.argv[1]
NPZ, ZAMAN, UZAY = {"TR": ("turkiye.npz", "etas4_E2", "etas10_TR"), "JP": ("japonya.npz", "etas7_E5", "etas10_JP")}[TE]
N = 900
SAAT = 12.0
rng = np.random.default_rng(10)
v = np.load(KOK / "veri" / NPZ)
t, mg, bolum = v["t_gun"], v["buyukluk"] - float(v["mc"]), v["bolum"]
uj = json.loads((KOK / f"sonuclar/{UZAY}/etas_uzay.json").read_text(encoding="utf-8"))
la0, lo0 = uj["merkez"]
x = (v["boylam"] - lo0) * np.cos(np.radians(la0)) * 111.2
y = (v["enlem"] - la0) * 111.2
su = uj["secilen"]
d_, gam, rho, h = su["d_km2"], su["gamma"], su["rho"], su["h_km"]
s = json.loads((KOK / f"sonuclar/{ZAMAN}/etas.json").read_text(encoding="utf-8"))["secilen"]
a, c, p, K = s["a"], s["c"], s["p"], s["K"]
tau = s.get("tau")
mu = s["mu"][0] if isinstance(s["mu"], list) else s["mu"]
beta = 1.0 / ((mg[bolum == 1]).mean() + 0.05)
q = 1.0 - p
eg = np.where(bolum == 1)[0]


def F(z):
    z = np.asarray(z, float)
    if tau is not None:
        z = np.minimum(z, tau)
    return c ** q * np.expm1(q * np.log1p(z / c)) / q


def F_ters(u):
    return c * np.expm1(np.log1p(u * q / c ** q) / q)


def gr(n):
    u = rng.random(n)
    m = -0.05 - np.log(1 - u * (1 - np.exp(-beta * 6.05))) / beta
    return np.floor(m * 10 + 0.5 + 1e-9) / 10


def yer(px, py, pm):
    D = d_ * np.exp(gam * pm)
    r = np.sqrt(D * ((1 - rng.random(len(px))) ** (-1 / rho) - 1))
    th = rng.random(len(px)) * 2 * np.pi
    return px + r * np.cos(th), py + r * np.sin(th)


t0 = t[int(v["ana_sok_idx"])] + SAAT / 24
son = t0 + 1.0
j = np.searchsorted(t, t0, side="left")
Fa, Fb = F(t0 - t[:j]), F(son - t[:j])
lam = K * np.exp(a * mg[:j]) * (Fb - Fa)
n_h = rng.poisson(lam.sum(), N)
eb = rng.choice(j, size=n_h.sum(), p=lam / lam.sum())
ct = t[:j][eb] + F_ters(Fa[eb] + rng.random(len(eb)) * (Fb[eb] - Fa[eb]))
cx, cy = yer(x[eb], y[eb], mg[eb])
cs = np.repeat(np.arange(N), n_h)
n_b = rng.poisson(mu, N)
bk = rng.choice(eg, n_b.sum())
bt = t0 + rng.random(n_b.sum())
bx, by = x[bk] + h * rng.standard_normal(len(bk)), y[bk] + h * rng.standard_normal(len(bk))
nt, nx, ny, ns = np.concatenate([ct, bt]), np.concatenate([cx, bx]), np.concatenate([cy, by]), np.concatenate([cs, np.repeat(np.arange(N), n_b)])
nm = gr(len(nt))
kayit = []
while len(nt):
    k = nt <= son
    nt, nx, ny, ns, nm = nt[k], nx[k], ny[k], ns[k], nm[k]
    kayit.append(np.stack([ns, nt, nx, ny, nm], 1))
    Fu = F(son - nt)
    n_c = rng.poisson(K * np.exp(a * nm) * Fu)
    if n_c.sum() == 0:
        break
    e = np.repeat(np.arange(len(nt)), n_c)
    ct = nt[e] + F_ters(rng.random(len(e)) * Fu[e])
    cx, cy = yer(nx[e], ny[e], nm[e])
    nt, nx, ny, ns, nm = ct, cx, cy, ns[e], gr(len(e))
r = np.concatenate(kayit)
cikti = KOK / f"sonuclar/c10_{TE}_harita"
cikti.mkdir(parents=True, exist_ok=True)
np.savez(cikti / "senaryo_konum_etas.npz", t0=t0, senaryo=r[:, 0], t=r[:, 1], x_km=r[:, 2], y_km=r[:, 3], m=r[:, 4], n_senaryo=N,
         merkez=np.array([la0, lo0]))
print(TE, "olay", len(r), "senaryo başına", len(r) / N)
