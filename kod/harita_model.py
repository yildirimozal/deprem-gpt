"""Haritalı artçı tahmini (ON_KAYIT_10.md, ikincil): S2 modeli (işaretçi konum başlığı) ile otoregresif benzetim.

Ana şoktan 12 saat sonra yayınlanan tahmin: önümüzdeki 24 saatte olayların zamanı, büyüklüğü ve yeri.
Her senaryoda model bir sonraki olayın zamanını, büyüklüğünü ve yerini çeker (yer: bağlamdaki bir olaya
işaretçi + Gauss ya da arka plan karışımı), olayı bağlama ekler. 3 tohum × 300 senaryo.

Sunucuda: python kod\\harita_model.py --ana turkiye.npz --ek ek_tr.npz --kosu c10_TR_S2 --ad c10_TR_harita
Çıktı: sonuclar/<ad>/senaryo_konum.npz (x, y [1000 km, katalog merkezine göre], m göreli, t, senaryo no)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parent.parent
ap0 = argparse.ArgumentParser()
ap0.add_argument("--ana", required=True)
ap0.add_argument("--ek", required=True)
ap0.add_argument("--kosu", required=True)
ap0.add_argument("--ad", required=True)
ap0.add_argument("--S", type=int, default=300)
ap0.add_argument("--saat", type=float, default=12.0)
ap0.add_argument("--cihaz", default="auto", choices=["auto", "cuda", "mps", "cpu"])
a0 = ap0.parse_args()
sys.argv = ["egit3.py", "--L", "512", "--ad", "c10_gecici", "--ana", a0.ana, "--ek", a0.ek, "--uzay_isaretci", "--cihaz", a0.cihaz]
kaynak = (KOK / "kod/egit3.py").read_text(encoding="utf-8").split("# ---------------- eğitim ----------------")[0]
exec(kaynak)                                                                    # noqa: S102


@torch.no_grad()
def senaryo(t0, ufuk=1.0, adim_siniri=3000):
    S = a0.S
    i_son = int(np.searchsorted(t_ca, t0, side="left")) - 1
    idx = torch.arange(i_son - L + 1, i_son + 1, device=cihaz)
    X = X_g[idx][None].expand(S, -1, -1).clone()
    C = C_g[idx][None].expand(S, -1).clone()
    T = T_g[idx][None].expand(S, -1).clone()
    Kk = K_g[idx][None].expand(S, -1).clone()
    P = P_g[idx][None].expand(S, -1, -1).clone()
    t_son = torch.full((S,), float(t_ca[i_son]), dtype=torch.float64, device=cihaz)
    aktif = torch.ones(S, dtype=torch.bool, device=cihaz)
    y_min = float(np.log(max(t0 - t_ca[i_son], 1e-9)))
    kayit = []
    for adim in range(adim_siniri):
        with amp_baglami(cihaz):
            h_tum = model(X, C, T, Kk)
        h_tum = h_tum.float()
        h = h_tum[:, -1]
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
        g = lambda q: q.gather(1, k[:, None])[:, 0]                              # noqa: E731
        u = (g(lo_k) + (1 - g(lo_k)) * torch.rand(S, device=cihaz)).clamp(1e-7, 1 - 1e-7)
        yy = g(mu) + g(sd) * torch.special.ndtri(u)
        if adim == 0:
            yy = torch.maximum(yy, torch.tensor(y_min, device=cihaz))
        dtg = torch.exp(yy.double()).clamp_min(DT_MIN)
        t_yeni = t_son + dtg
        sinif = torch.multinomial(torch.softmax(model.buyuk(h), -1), 1)[:, 0]
        m_yeni = sinif.float() * 0.1
        # konum: arka plan (8 bileşenli GMM) ya da bağlamdaki bir olaya işaretçi
        skor = (model.uq(h)[:, None, :] @ model.uk(h_tum).transpose(1, 2))[:, 0] / 8.0       # (S, L)
        pi = torch.softmax(torch.cat([model.ubg(h), skor], -1), -1)
        sec = torch.multinomial(pi, 1)[:, 0]
        bw, bmx, bmy, blx, bly = model.uzay(h).split(8, -1)
        kb = torch.multinomial(torch.softmax(bw, -1), 1)[:, 0]
        gb = lambda q: q.gather(1, kb[:, None])[:, 0]                            # noqa: E731
        bx = gb(bmx) + torch.exp(gb(blx).clamp(LS_ALT, LS_UST)) * torch.randn(S, device=cihaz)
        by = gb(bmy) + torch.exp(gb(bly).clamp(LS_ALT, LS_UST)) * torch.randn(S, device=cihaz)
        j = (sec - 1).clamp_min(0)
        pj = P.gather(1, j[:, None, None].expand(-1, 1, 2))[:, 0]
        sj = torch.exp(model.us(h_tum).clamp(LS_ALT, LS_UST)[:, :, 0].gather(1, j[:, None])[:, 0])
        px = torch.where(sec == 0, bx, pj[:, 0] + sj * torch.randn(S, device=cihaz))
        py = torch.where(sec == 0, by, pj[:, 1] + sj * torch.randn(S, device=cihaz))
        icinde = aktif & (t_yeni <= t0 + ufuk)
        if icinde.any():
            s_no = torch.where(icinde)[0]
            kayit.append(torch.stack([s_no.double(), t_yeni[s_no], px[s_no].double(), py[s_no].double(), m_yeni[s_no].double()], 1).cpu().numpy())
        aktif = icinde
        if not aktif.any():
            break
        uz = torch.log10(torch.hypot(px - P[:, -1, 0], py - P[:, -1, 1]) * 1000 + 1) / 3
        oz = torch.stack([(torch.log10(dtg.float()) + 1.0) / 1.5, m_yeni, px, py, uz, torch.ones_like(px)], 1)
        X = torch.cat([X[:, 1:], oz[:, None]], 1)
        C = torch.cat([C[:, 1:], sinif[:, None]], 1)
        T = torch.cat([T[:, 1:], t_yeni[:, None]], 1)
        Kk = torch.cat([Kk[:, 1:], Kk[:, -1:]], 1)
        P = torch.cat([P[:, 1:], torch.stack([px, py], 1)[:, None]], 1)
        t_son = t_yeni
    return np.concatenate(kayit) if kayit else np.zeros((0, 5)), adim + 1


t0 = float(t_ca[int(v["ana_sok_idx"])] + a0.saat / 24)
cikti = KOK / "sonuclar" / a0.ad
cikti.mkdir(parents=True, exist_ok=True)
tum = []
for tohum in (1, 2, 3):
    torch.manual_seed(2000 + tohum)
    model.load_state_dict(torch.load(KOK / "sonuclar" / f"{a0.kosu}_t{tohum}" / "en_iyi.pt", map_location=cihaz))
    model.eval()
    r, n = senaryo(t0)
    r[:, 0] += (tohum - 1) * a0.S
    tum.append(r)
    print(f"tohum {tohum}: {len(r)} olay, {n} adım", flush=True)
r = np.concatenate(tum)
np.savez(cikti / "senaryo_konum.npz", t0=t0, senaryo=r[:, 0], t=r[:, 1], x=r[:, 2], y=r[:, 3], m=r[:, 4], n_senaryo=3 * a0.S)
print("IS_BITTI", flush=True)
