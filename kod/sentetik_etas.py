"""Ön eğitim için sentetik zamansal ETAS katalogları (ON_KAYIT_8.md, varyant A3/A4).

Her katalogda parametreler ayrı çekilir; büyüklükler göreli (M - Mc), Mc'nin 0.05 altından başlayan
kesik Gutenberg-Richter (üst sınır 6.0) ve 0.1'e yuvarlanır (gerçek kataloglarla aynı biçim).
Yarısında Helmstetter tipi algılama kaybı uygulanır: algılanmayan olaylar tetiklemeye katılır, ama
kataloğa girmez. Hiçbir gerçek veri kullanılmaz.

Çıktı: veri/sentetik.npz (t_gun, m_goreli, seg_bas, seg_son, seg_ad), sonuclar/sentetik_ozeti.json
Mac'te: /usr/bin/python3 kod/sentetik_etas.py
"""
import json
from math import erf
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(20261121)
N_KAT = 400
M_UST = 6.0
SINIR_OLAY = 40000
SIG = 0.2


def gr(n, beta):
    """[-0.05, M_UST] aralığında kesik üstel."""
    u = rng.random(n)
    lo, hi = -0.05, M_UST
    return lo - np.log(1 - u * (1 - np.exp(-beta * (hi - lo)))) / beta


def omori_ornek(n, c, p, ust):
    """(x + c)^(-p) yoğunluğundan [0, ust] aralığında örnek (ust: her biri için)."""
    q = 1 - p
    Fu = (c ** q - (ust + c) ** q) / (p - 1)
    u = rng.random(n) * Fu
    return (c ** q - u * (p - 1)) ** (1 / q) - c


def omori_integral(c, p, ust):
    q = 1 - p
    return (c ** q - (ust + c) ** q) / (p - 1)


def katalog():
    while True:
        b = rng.uniform(0.85, 1.2)
        beta = b * np.log(10)
        a = rng.uniform(0.9, 2.5)
        p = rng.uniform(1.02, 1.35)
        c = np.exp(rng.uniform(np.log(0.002), np.log(0.1)))
        mu = np.exp(rng.uniform(np.log(0.05), np.log(3.0)))
        n_dal = rng.uniform(0.6, 1.0)
        T = rng.uniform(1000, 4000)
        mm = gr(200000, beta)
        E = np.mean(np.exp(a * mm))
        K = n_dal / (E * c ** (1 - p) / (p - 1))
        n0 = rng.poisson(mu * T)
        t = list(rng.uniform(0, T, n0))
        m = list(gr(n0, beta))
        if rng.random() < 0.7:
            t.append(rng.uniform(0.05 * T, 0.8 * T))
            m.append(rng.uniform(3.5, 5.5))
        t, m = np.array(t), np.array(m)
        ebeveyn_t, ebeveyn_m = t, m
        tum_t, tum_m = [t], [m]
        toplam = len(t)
        tasma = False
        while len(ebeveyn_t):
            beklenen = K * np.exp(a * ebeveyn_m) * omori_integral(c, p, T - ebeveyn_t)
            say = rng.poisson(beklenen)
            if say.sum() == 0:
                break
            et = np.repeat(ebeveyn_t, say)
            ct = et + omori_ornek(len(et), c, p, T - et)
            cm = gr(len(et), beta)
            tum_t.append(ct)
            tum_m.append(cm)
            toplam += len(ct)
            if toplam > SINIR_OLAY:
                tasma = True
                break
            ebeveyn_t, ebeveyn_m = ct, cm
        if tasma or toplam < 300:
            continue
        t = np.concatenate(tum_t)
        m = np.concatenate(tum_m)
        o = np.argsort(t, kind="stable")
        t, m = t[o], m[o]
        eksik = rng.random() < 0.5
        G = H = None
        if eksik:
            G, H = rng.uniform(4.0, 6.0), rng.uniform(0.3, 1.0)
            buyuk = np.where(m >= 2.0)[0]
            mud = np.full(len(t), -2.0)
            for j in buyuk:                                   # her büyük olayın etkisi: sonraki olaylar
                k0 = j + 1
                k1 = np.searchsorted(t, t[j] + 365.0)
                if k1 <= k0:
                    continue
                tau = np.maximum(t[k0:k1] - t[j], 1e-9)
                mud[k0:k1] = np.maximum(mud[k0:k1], m[j] - G - H * np.log10(tau))
            p_alg = 0.5 * (1 + np.vectorize(erf)((m - mud) / (SIG * np.sqrt(2))))
            tut = rng.random(len(t)) < p_alg
            t, m = t[tut], m[tut]
        m = np.floor(m * 10 + 0.5 + 1e-9) / 10
        k = m >= 0
        t, m = t[k], m[k]
        if len(t) < 300:
            continue
        return t, m, {"b": b, "a": a, "p": p, "c": c, "mu": mu, "dallanma": n_dal, "T": T, "eksik": eksik,
                      "G": G, "H": H, "olay": int(len(t)), "M_max": float(m.max())}


tl, ml, sb, ss, bilgi = [], [], [], [], []
n = 0
for i in range(N_KAT):
    t, m, b = katalog()
    tl.append(t + 0.0)
    ml.append(m)
    sb.append(n)
    n += len(t)
    ss.append(n)
    bilgi.append(b)
    if i % 50 == 0:
        print(i, n, flush=True)
np.savez(KOK / "veri/sentetik.npz", t_gun=np.concatenate(tl), m_goreli=np.concatenate(ml),
         seg_bas=np.array(sb), seg_son=np.array(ss), seg_ad=np.array(["Sentetik"] * N_KAT))
olay = np.array([b["olay"] for b in bilgi])
with open(KOK / "sonuclar/sentetik_ozeti.json", "w", encoding="utf-8") as f:
    json.dump({"n_katalog": N_KAT, "toplam_olay": int(n), "olay_medyan": float(np.median(olay)), "olay_max": int(olay.max()),
               "eksik_kayitli_katalog": int(sum(b["eksik"] for b in bilgi)),
               "M_max_4_ustu_katalog": int(sum(b["M_max"] >= 4.0 for b in bilgi)), "kataloglar": bilgi}, f, indent=1)
print("toplam", n, "medyan", np.median(olay))
