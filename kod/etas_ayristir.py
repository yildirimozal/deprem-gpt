"""ETAS varyantlarının test olaylarındaki TLL'sini iki parçaya ayırır (Mac, numpy):
    TLL_i = log lambda(t_i) - Lambda_i,   Lambda_i = integral_{t_(i-1)}^{t_i} lambda
Lambda_i zaman yeniden ölçekleme kalıntısıdır. Doğru bir modelde Lambda_i ~ Üstel(1), ortalaması 1.
Ortalama 1'den büyükse model o dönemde gözlenenden çok olay bekliyor demektir.

Çıktı: sonuclar/c5_etas_ayristirma.npz  (her varyant için log_lam, Lam; saklanan tll ile denetim)
Mac'te: /usr/bin/python3 kod/etas_ayristir.py
"""
import json
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent
v = np.load(KOK / "veri/turkiye.npz")
t, mg, bolum = v["t_gun"], v["buyukluk"] - float(v["mc"]), v["bolum"]
test = np.where(bolum == 3)[0]


def F(x, c, p):
    q = 1.0 - p
    return c ** q * np.expm1(q * np.log1p(x / c)) / q


def ayristir(prm):
    a, c, p, tau, K, mu = prm["a"], prm["c"], prm["p"], prm.get("tau"), prm["K"], prm["mu"]
    mu = mu[0] if isinstance(mu, list) else mu
    log_lam, Lam = np.empty(len(test)), np.empty(len(test))
    g_all = np.exp(a * mg)
    for s in range(0, len(test), 256):
        ii = test[s:s + 256]
        jmax = ii[-1]
        tj, g = t[:jmax], g_all[:jmax]
        d = np.clip(t[ii][:, None] - tj[None, :], 0, None)
        dp = np.clip(t[ii - 1][:, None] - tj[None, :], 0, None)
        maske = np.arange(jmax)[None, :] < ii[:, None]
        if tau is None:
            cek = np.where(maske, (d + c) ** (-p), 0.0)
            fark = F(d, c, p) - F(dp, c, p)
        else:
            cek = np.where(maske & (d < tau), (d + c) ** (-p), 0.0)
            fark = F(np.minimum(d, tau), c, p) - F(np.minimum(dp, tau), c, p)
        log_lam[s:s + 256] = np.log(mu + K * (cek @ g))
        Lam[s:s + 256] = mu * (t[ii] - t[ii - 1]) + K * (np.where(maske, fark, 0.0) @ g)
    return log_lam, Lam


out = {}
for k in ["E0", "E1", "E2", "E4", "E5"]:
    prm = json.loads((KOK / f"sonuclar/etas4_{k}/etas.json").read_text(encoding="utf-8"))["secilen"]
    ll, L = ayristir(prm)
    kayitli = np.load(KOK / f"sonuclar/etas4_{k}/etas_test.npz")["tll"]
    fark = float(np.abs(ll - L - kayitli).max())
    print(k, "en büyük fark (saklanan tll ile):", f"{fark:.2e}", "ort Lambda:", round(float(L.mean()), 3))
    assert fark < 1e-6, k
    out[f"{k}_log_lam"], out[f"{k}_Lam"] = ll, L
np.savez(KOK / "sonuclar/c5_etas_ayristirma.npz", idx=test, **out)
