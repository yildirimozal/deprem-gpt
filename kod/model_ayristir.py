"""Üçüncü testin modellerinin (c3_T{0,1,2}_t{1,2,3}) test olaylarındaki TLL'sini ETAS'la aynı biçimde
ayırır. Model bir sonraki olaya kadar geçen süre için yoğunluk p(Δt) veriyor. Bu bir yenilenme (renewal)
biçimi; tehlike oranı h = p / S, birikimli tehlike H = -log S (S = 1 - F). Böylece
    TLL_i = log h(Δt_i) - H(Δt_i)
ETAS'taki log lambda(t_i) ve Lambda_i'nin karşılığı. Ayrıca PIT = F(Δt_i) ve öngörülen medyan Δt.

Sunucuda: python kod\\model_ayristir.py   (egit.py'nin model tanımını kullanır; eğitim yapmaz)
Çıktı: sonuclar/c5_model_ayristirma.npz
"""
import sys
from pathlib import Path

import numpy as np
import torch

KOK = Path(__file__).resolve().parent.parent
sys.argv = ["egit.py", "--L", "512", "--ad", "c5_gecici", "--ana", "turkiye.npz"]
kaynak = (KOK / "kod/egit.py").read_text(encoding="utf-8").split("# ---------------- eğitim ----------------")[0]
exec(kaynak)                                        # veri, model sınıfı, degerlendir tanımlanır  # noqa: S102


@torch.no_grad()
def ayristir(m):
    m.eval()
    hedef = torch.tensor(test_idx, device=cihaz)
    ofs = torch.arange(-L, 0, device=cihaz)
    out = {k: [] for k in ["tll", "log_h", "H", "pit", "medyan_ln"]}
    for i in range(0, len(hedef), 256):
        k = hedef[i:i + 256]
        idx = k[:, None] + ofs[None]
        with amp_baglami(cihaz):
            h = m(X_g[idx], C_g[idx], T_g[idx])[:, -1]
        h = h.float()
        w, mu, ls = m.zaman(h).split(m.K, -1)
        ls = ls.clamp(-6, 4)
        lw = torch.log_softmax(w.double(), -1)
        mu, sd = mu.double(), torch.exp(ls.double())
        y = Y_g[k].double()[:, None]
        z = (y - mu) / sd
        log_p_y = torch.logsumexp(lw - 0.5 * z * z - torch.log(sd) - 0.5 * np.log(2 * np.pi), -1)
        tll = log_p_y - y[:, 0]                                      # gün başına yoğunluk
        log_S = torch.logsumexp(lw + torch.special.log_ndtr(-z), -1)  # hayatta kalma
        F = 1 - torch.exp(log_S)
        # medyan: karışımın F(y)=0.5 noktası, ikiye bölmeyle
        lo, hi = torch.full_like(y[:, 0], -30.0), torch.full_like(y[:, 0], 15.0)
        for _ in range(60):
            o = (lo + hi) / 2
            Fo = (torch.exp(lw) * torch.special.ndtr((o[:, None] - mu) / sd)).sum(-1)
            lo, hi = torch.where(Fo < 0.5, o, lo), torch.where(Fo < 0.5, hi, o)
        out["tll"].append(tll)
        out["log_h"].append(tll - log_S)
        out["H"].append(-log_S)
        out["pit"].append(F)
        out["medyan_ln"].append((lo + hi) / 2)
    return {k: torch.cat(x).cpu().numpy() for k, x in out.items()}


sonuc = {}
for kod in ["T0", "T1", "T2"]:
    for s in (1, 2, 3):
        ad = f"c3_{kod}_t{s}"
        model.load_state_dict(torch.load(KOK / "sonuclar" / ad / "en_iyi.pt", map_location=cihaz))
        r = ayristir(model)
        kayitli = np.load(KOK / "sonuclar" / ad / "test.npz")["tll"]
        fark = float(np.abs(r["tll"] - kayitli).max())
        print(ad, "saklanan tll ile en büyük fark", f"{fark:.2e}", "ort H", round(float(r["H"].mean()), 3), flush=True)
        for k, x in r.items():
            sonuc[f"{ad}_{k}"] = x
np.savez(KOK / "sonuclar/c5_model_ayristirma.npz", idx=test_idx, **sonuc)
print("IS_BITTI", flush=True)
