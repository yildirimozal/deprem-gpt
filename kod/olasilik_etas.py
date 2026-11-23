"""ETAS'tan operasyonel olasılık tahmini (ON_KAYIT_9.md): ileri benzetim.

Her yayın zamanı t0 için [t0, t0 + 24 saat] aralığında senaryolar:
  - geçmiş olayların (t_j < t0) bu aralıkta doğuracağı artçılar (Poisson, Omori ters CDF),
  - arka plan (mu, düzgün),
  - aralık içinde doğan her olayın kendi artçıları (kademeli, aralık sonuna kadar).
Büyüklükler kesik Gutenberg-Richter'den (b eğitim döneminden, Mc - 0.05'ten M 9.0'a), 0.1'e yuvarlı.
Parametreler sabit: Türkiye E2 (sonuclar/etas4_E2), Japonya E5 (sonuclar/etas7_E5, Japonya arka planı).

Mac'te: /usr/bin/python3 kod/olasilik_etas.py TR   |   ... JP
Çıktı: sonuclar/c9_<TR|JP>_etas/senaryo.npz
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent
TE = sys.argv[1]
AYAR = {"TR": ("turkiye.npz", "etas4_E2"), "JP": ("japonya.npz", "etas7_E5")}[TE]
N = 2000
UFUK = 1.0
M_UST = 6.0                      # göreli (Mc = 3.0 → M 9.0)
SINIR = 20000
rng = np.random.default_rng(9)

v = np.load(KOK / "veri" / AYAR[0])
t, mg, bolum = v["t_gun"], v["buyukluk"] - float(v["mc"]), v["bolum"]
beta = 1.0 / ((mg[bolum == 1]).mean() + 0.05)
s = json.loads((KOK / f"sonuclar/{AYAR[1]}/etas.json").read_text(encoding="utf-8"))["secilen"]
a, c, p, K = s["a"], s["c"], s["p"], s["K"]
tau = s.get("tau")
mu = s["mu"][0] if isinstance(s["mu"], list) else s["mu"]
q = 1.0 - p


def F(x):
    x = np.asarray(x, float)
    if tau is not None:
        x = np.minimum(x, tau)
    return c ** q * np.expm1(q * np.log1p(x / c)) / q


def F_ters(u):
    return c * np.expm1(np.log1p(u * q / c ** q) / q)


def gr(n):
    u = rng.random(n)
    lo, hi = -0.05, M_UST
    m = lo - np.log(1 - u * (1 - np.exp(-beta * (hi - lo)))) / beta
    return np.floor(m * 10 + 0.5 + 1e-9) / 10


def yayin_zamanlari():                       # olasilik_model.py ile aynı
    test = np.where(bolum == 3)[0]
    ilk = np.floor(t[test[0]] + 1e-9)
    son = t[test[-1]]
    gunluk = np.arange(ilk, son - UFUK + 1e-9, 1.0)
    ana_t = t[int(v["ana_sok_idx"])]
    ek = ana_t + np.concatenate([[1.0 / 1440], np.arange(1, 25) * 3.0 / 24])
    return np.concatenate([gunluk, ek]), np.concatenate([np.zeros(len(gunluk), int), np.ones(len(ek), int)])


def senaryo(t0):
    son = t0 + UFUK
    j = np.searchsorted(t, t0, side="left")
    tj, mj = t[:j], mg[:j]
    Fa, Fb = F(t0 - tj), F(son - tj)
    lam = K * np.exp(a * mj) * (Fb - Fa)
    L_top = lam.sum()
    n_h = rng.poisson(L_top, N)
    ebeveyn = rng.choice(j, size=n_h.sum(), p=lam / L_top) if n_h.sum() else np.array([], int)
    u = Fa[ebeveyn] + rng.random(len(ebeveyn)) * (Fb[ebeveyn] - Fa[ebeveyn])
    t_h = tj[ebeveyn] + F_ters(u)
    s_h = np.repeat(np.arange(N), n_h)
    n_b = rng.poisson(mu * UFUK, N)
    t_b = t0 + rng.random(n_b.sum()) * UFUK
    s_b = np.repeat(np.arange(N), n_b)
    nesil_t = np.concatenate([t_h, t_b])
    nesil_s = np.concatenate([s_h, s_b])
    nesil_m = gr(len(nesil_t))
    maxm = np.full(N, -1.0)
    say4 = np.zeros(N, int)
    say5 = np.zeros(N, int)
    olay = np.zeros(N, int)
    tasma = np.zeros(N, bool)
    while len(nesil_t):
        k = nesil_t <= son
        nesil_t, nesil_s, nesil_m = nesil_t[k], nesil_s[k], nesil_m[k]
        np.maximum.at(maxm, nesil_s, nesil_m)
        np.add.at(say4, nesil_s, (nesil_m >= 1.0 - 1e-9).astype(int))
        np.add.at(say5, nesil_s, (nesil_m >= 2.0 - 1e-9).astype(int))
        np.add.at(olay, nesil_s, 1)
        tasma |= olay > SINIR
        k = ~tasma[nesil_s]
        nesil_t, nesil_s, nesil_m = nesil_t[k], nesil_s[k], nesil_m[k]
        Fu = F(son - nesil_t)
        n_c = rng.poisson(K * np.exp(a * nesil_m) * Fu)
        if n_c.sum() == 0:
            break
        e = np.repeat(np.arange(len(nesil_t)), n_c)
        ct = nesil_t[e] + F_ters(rng.random(len(e)) * Fu[e])
        nesil_t, nesil_s, nesil_m = ct, nesil_s[e], gr(len(e))
    return maxm, say4, say5, olay, tasma


yayin, tur = yayin_zamanlari()
bas = time.time()
sonuc = [senaryo(t0) for t0 in yayin]
cikti = KOK / f"sonuclar/c9_{TE}_etas"
cikti.mkdir(parents=True, exist_ok=True)
np.savez(cikti / "senaryo.npz", yayin_t=yayin, tur=tur, maxm=np.stack([r[0] for r in sonuc]),
         say4=np.stack([r[1] for r in sonuc]), say5=np.stack([r[2] for r in sonuc]),
         olay=np.stack([r[3] for r in sonuc]), tasma=np.stack([r[4] for r in sonuc]))
with open(cikti / "bilgi.json", "w", encoding="utf-8") as f:
    json.dump({"ayar": AYAR, "N": N, "beta": beta, "b": beta / np.log(10), "a": a, "c": c, "p": p, "K": K, "tau": tau, "mu": mu,
               "sure_s": time.time() - bas, "tasan_senaryo": int(sum(r[4].sum() for r in sonuc))}, f, indent=1)
print(TE, len(yayin), "yayın", f"{time.time() - bas:.0f}s", "taşan", int(sum(r[4].sum() for r in sonuc)))
