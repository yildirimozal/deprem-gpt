"""Modelden operasyonel olasılık tahmini (ON_KAYIT_9.md): otoregresif Monte Carlo.

Her yayın zamanı t0 için: bağlam = t0'dan önceki son 512 olay. Model, her senaryoda bir sonraki olayın
zamanını (karışım dağılımı) ve büyüklüğünü (sınıf dağılımı) çekip olayı bağlama ekleyerek ilerler;
t0 + 24 saat aşılınca senaryo biter. İlk adımda dağılım, son olaydan t0'a kadar olay olmadığı
koşuluyla kesilir.

Sunucuda: python kod\\olasilik_model.py --ana turkiye.npz --kosu c3_T1 --ad c9_TR_model
Çıktı: sonuclar/<ad>/senaryo_t{tohum}.npz (yayin_t, tur, maxm[n_yayin, S], say4, say5, olay, tasma)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parent.parent
ap0 = argparse.ArgumentParser()
ap0.add_argument("--ana", required=True)
ap0.add_argument("--kosu", required=True)
ap0.add_argument("--ad", required=True)
ap0.add_argument("--S", type=int, default=400)
ap0.add_argument("--ufuk", type=float, default=1.0)
ap0.add_argument("--adim_siniri", type=int, default=4000)
ap0.add_argument("--cihaz", default="auto", choices=["auto", "cuda", "mps", "cpu"])
a0 = ap0.parse_args()

sys.argv = ["egit.py", "--L", "512", "--ad", "c9_gecici", "--ana", a0.ana, "--cihaz", a0.cihaz]
kaynak = (KOK / "kod/egit.py").read_text(encoding="utf-8").split("# ---------------- eğitim ----------------")[0]
exec(kaynak)                                                                    # noqa: S102


def yayin_zamanlari():
    test = np.where(bolum_ca == 3)[0]
    ilk = np.floor(t_ca[test[0]] + 1e-9)                       # test başlangıç günü 00:00
    son = t_ca[test[-1]]
    gunluk = np.arange(ilk, son - a0.ufuk + 1e-9, 1.0)
    ana_t = t_ca[int(v["ana_sok_idx"])]
    ek = ana_t + np.concatenate([[1.0 / 1440], np.arange(1, 25) * 3.0 / 24])   # ana şoktan 1 dk, sonra 3 saatte bir, 72 saat
    return np.concatenate([gunluk, ek]), np.concatenate([np.zeros(len(gunluk), int), np.ones(len(ek), int)])


@torch.no_grad()
def senaryo(t0):
    S = a0.S
    i_son = int(np.searchsorted(t_ca, t0, side="left")) - 1
    idx = torch.arange(i_son - L + 1, i_son + 1, device=cihaz)
    X = X_g[idx][None].expand(S, -1, -1).clone()
    C = C_g[idx][None].expand(S, -1).clone()
    T = T_g[idx][None].expand(S, -1).clone()
    t_son = torch.full((S,), float(t_ca[i_son]), dtype=torch.float64, device=cihaz)
    bitis = t0 + a0.ufuk
    aktif = torch.ones(S, dtype=torch.bool, device=cihaz)
    maxm = torch.full((S,), -1.0, device=cihaz)
    say4 = torch.zeros(S, dtype=torch.long, device=cihaz)
    say5 = torch.zeros(S, dtype=torch.long, device=cihaz)
    olay = torch.zeros(S, dtype=torch.long, device=cihaz)
    y_min = float(np.log(max(t0 - t_ca[i_son], 1e-9)))
    for adim in range(a0.adim_siniri):
        with amp_baglami(cihaz):
            h = model(X, C, T)[:, -1]
        h = h.float()
        w, mu, ls = model.zaman(h).split(model.K, -1)
        sd = torch.exp(ls.clamp(-6, 4))
        lw = torch.log_softmax(w, -1)
        if adim == 0:
            z = (y_min - mu) / sd
            lw = lw + torch.special.log_ndtr(-z)
            lo_k = torch.special.ndtr(z)
        else:
            lo_k = torch.zeros_like(mu)
        k = torch.multinomial(torch.softmax(lw, -1), 1)[:, 0]
        mu_k, sd_k, lo = mu.gather(1, k[:, None])[:, 0], sd.gather(1, k[:, None])[:, 0], lo_k.gather(1, k[:, None])[:, 0]
        u = lo + (1 - lo) * torch.rand(S, device=cihaz)
        u = u.clamp(1e-7, 1 - 1e-7)
        y = mu_k + sd_k * torch.special.ndtri(u)
        if adim == 0:
            y = torch.maximum(y, torch.tensor(y_min, device=cihaz))
        dtg = torch.exp(y.double()).clamp_min(DT_MIN)
        t_yeni = t_son + dtg
        sinif = torch.multinomial(torch.softmax(model.buyuk(h), -1), 1)[:, 0]
        m_yeni = sinif.float() * 0.1
        icinde = aktif & (t_yeni <= bitis)
        maxm = torch.where(icinde, torch.maximum(maxm, m_yeni), maxm)
        say4 += (icinde & (m_yeni >= 1.0 - 1e-6)).long()
        say5 += (icinde & (m_yeni >= 2.0 - 1e-6)).long()
        olay += icinde.long()
        aktif = icinde
        if not aktif.any():
            break
        oz = torch.stack([(torch.log10(dtg.float()) + 1.0) / 1.5, m_yeni], 1)
        X = torch.cat([X[:, 1:], oz[:, None]], 1)
        C = torch.cat([C[:, 1:], sinif[:, None]], 1)
        T = torch.cat([T[:, 1:], t_yeni[:, None]], 1)
        t_son = t_yeni
    return maxm.cpu().numpy(), say4.cpu().numpy(), say5.cpu().numpy(), olay.cpu().numpy(), aktif.cpu().numpy(), adim + 1


yayin, tur = yayin_zamanlari()
cikti = KOK / "sonuclar" / a0.ad
cikti.mkdir(parents=True, exist_ok=True)
for tohum in (1, 2, 3):
    torch.manual_seed(1000 + tohum)
    model.load_state_dict(torch.load(KOK / "sonuclar" / f"{a0.kosu}_t{tohum}" / "en_iyi.pt", map_location=cihaz))
    model.eval()
    bas = time.time()
    sonuc = [senaryo(t0) for t0 in yayin]
    np.savez(cikti / f"senaryo_t{tohum}.npz", yayin_t=yayin, tur=tur,
             maxm=np.stack([r[0] for r in sonuc]), say4=np.stack([r[1] for r in sonuc]), say5=np.stack([r[2] for r in sonuc]),
             olay=np.stack([r[3] for r in sonuc]), tasma=np.stack([r[4] for r in sonuc]), adim=np.array([r[5] for r in sonuc]))
    print(f"tohum {tohum}: {len(yayin)} yayın, {time.time() - bas:.0f}s, en çok adım {max(r[5] for r in sonuc)}, "
          f"taşan senaryo {sum(int(r[4].sum()) for r in sonuc)}", flush=True)
print("IS_BITTI", flush=True)
