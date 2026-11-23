"""Deprem GPT: her deprem bir jeton, sonraki depremin zamanı ve büyüklüğü için dağılım.

Kullanım:
  python egit.py --L 4096 --tohum 1 --ad L4096_t1
  python egit.py --L 4096 --tohum 1 --ad tam_L4096_t1 --egitim_bolum 1,2 --sabit_adim 1200

Çıktı (sonuclar/<ad>/): kayit.json, test.npz (olay başına zamansal ve büyüklük LL), en_iyi.pt
"""
import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KOK = Path(__file__).resolve().parent.parent
LN2PI = math.log(2 * math.pi)
N_BUYUK = 60          # 2.5, 2.6, ..., 8.4 (son bölme ≥ 8.4)
DT_MIN = 1e-6         # gün; 4 test olayında Δt = 0 (EarthquakeNPP kataloğu), bu tabana çekiliyor

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, required=True)
ap.add_argument("--tohum", type=int, default=1)
ap.add_argument("--ad", required=True)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--katman", type=int, default=4)
ap.add_argument("--bas", type=int, default=4)
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--hedef_jeton", type=int, default=16384)
ap.add_argument("--adim", type=int, default=4000)
ap.add_argument("--deger_her", type=int, default=100)
ap.add_argument("--sabir", type=int, default=8)
ap.add_argument("--egitim_bolum", default="1")
ap.add_argument("--sabit_adim", type=int, default=0, help=">0 ise erken durdurma yok, bu kadar adım")
ap.add_argument("--deger_jeton", type=int, default=262144)
ap.add_argument("--ek", default="", help="veri/ altında ek katalog npz")
ap.add_argument("--ana", default="olaylar.npz", help="veri/ altında ana katalog (doğrulama ve test buradan)")
ap.add_argument("--cihaz", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="çalışma cihazı")
arg = ap.parse_args()


def sec_cihaz(istek):
    if istek == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(istek)


def amp_baglami(c):
    return torch.autocast("cuda", dtype=torch.bfloat16) if c.type == "cuda" else contextlib.nullcontext()


torch.manual_seed(arg.tohum)
np.random.seed(arg.tohum)
cihaz = sec_cihaz(arg.cihaz)
if cihaz.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

# ---------------- veri ----------------
# California önce gelir; doğrulama/test indeksleri ek katalog eklense de değişmez.
v = np.load(KOK / "veri" / arg.ana)
MC_ANA = float(v["mc"]) if "mc" in v.files else 2.5
t_ca = v["t_gun"].astype(np.float64)
bolum_ca = v["bolum"]
seg = [(0, len(t_ca))]
t_l, mg_l, bolum_l = [t_ca], [np.round(v["buyukluk"] - MC_ANA, 1)], [bolum_ca]
if arg.ek:
    e = np.load(KOK / "veri" / arg.ek)
    for a, b in zip(e["seg_bas"], e["seg_son"]):
        n0 = sum(len(x) for x in t_l)
        t_l.append(e["t_gun"][a:b])
        mg_l.append(e["m_goreli"][a:b])
        bolum_l.append(np.full(b - a, 5, np.int8))      # 5: ek katalog, hep eğitim hedefi
        seg.append((n0, n0 + b - a))
t = np.concatenate(t_l)
mg = np.concatenate(mg_l)
bolum = np.concatenate(bolum_l)
dt = np.concatenate([np.diff(x, prepend=x[0] - 1.0) for x in t_l])   # her katalog kendi içinde
dt = np.maximum(dt, DT_MIN)
ozellik = np.stack([(np.log10(dt) + 1.0) / 1.5, mg], 1).astype(np.float32)
sinif = np.clip(np.round(mg / 0.1), 0, N_BUYUK - 1).astype(np.int64)

T_g = torch.tensor(t, dtype=torch.float64, device=cihaz)
X_g = torch.tensor(ozellik, device=cihaz)
C_g = torch.tensor(sinif, device=cihaz)
Y_g = torch.tensor(np.log(dt), dtype=torch.float32, device=cihaz)   # ln Δt (gün)

egitim_kume = [int(b) for b in arg.egitim_bolum.split(",") if b.strip()] + [5]
egitim_hedef = np.isin(bolum, egitim_kume)
H_g = torch.tensor(egitim_hedef, device=cihaz)
L = arg.L
# geçerli pencere başlangıçları: pencere s..s+L tek katalogda; California'da ilk testteki kural
ca_h = np.where(egitim_hedef[:len(t_ca)])[0]
baslar = [np.arange(max(0, ca_h[0] - L), ca_h[-1] - L + 1)] if len(ca_h) else []
for a, b in seg[1:]:
    if b - a > L + 1:
        baslar.append(np.arange(a, b - L))
baslar = np.concatenate(baslar)
basla_g = torch.tensor(baslar, device=cihaz)
dogrulama_idx = np.where(bolum == 2)[0]
test_idx = np.where(bolum == 3)[0]


# ---------------- model ----------------
class Konum(nn.Module):
    """Başın yarısı olay sırasıyla, yarısı olay zamanıyla döner (RoPE)."""

    def __init__(self, dh):
        super().__init__()
        n = dh // 2
        ni = n // 2
        nt = n - ni
        w_i = 1.0 / (10000 ** (torch.arange(ni, dtype=torch.float64) / ni))
        w_t = 1.0 / torch.logspace(-4, 4, nt, dtype=torch.float64)       # zaman ölçekleri 1e-4..1e4 gün
        self.register_buffer("w_i", w_i, persistent=False)
        self.register_buffer("w_t", w_t, persistent=False)

    def forward(self, tz):                       # tz: (B, T) float64, pencere başına göre
        B, T = tz.shape
        sira = torch.arange(T, device=tz.device, dtype=torch.float64)
        a_i = sira[None, :, None] * self.w_i[None, None, :]
        a_t = (tz - tz[:, :1])[..., None] * self.w_t[None, None, :]
        a = torch.cat([a_i.expand(B, -1, -1), a_t], -1)
        return torch.cos(a).float()[:, None], torch.sin(a).float()[:, None]


def dondur(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Blok(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.h, self.p = h, p
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, cos, sin):
        B, T, d = x.shape
        q, k, vv = self.qkv(self.ln1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = dondur(q, cos, sin), dondur(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, vv, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, d)))
        return x + self.drop(self.mlp(self.ln2(x)))


class DepremGPT(nn.Module):
    def __init__(self, d, n_k, h, K, p):
        super().__init__()
        self.giris = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.m_gom = nn.Embedding(N_BUYUK, d)
        self.konum = Konum(d // h)
        self.bloklar = nn.ModuleList([Blok(d, h, p) for _ in range(n_k)])
        self.ln = nn.LayerNorm(d)
        self.zaman = nn.Linear(d, 3 * K)
        self.buyuk = nn.Linear(d, N_BUYUK)
        self.K = K
        with torch.no_grad():                    # karışım ortalamalarını ln Δt aralığına yay
            self.zaman.bias[K:2 * K] = torch.linspace(-12, 2, K)

    def forward(self, x, c, tz):
        h = self.giris(x) + self.m_gom(c)
        cos, sin = self.konum(tz)
        for b in self.bloklar:
            h = b(h, cos, sin)
        return self.ln(h)

    def ll(self, h, y, c):
        """h: (..., d); y = ln Δt hedefi; c = büyüklük sınıfı. Gün biriminde log p(Δt), log p(m)."""
        h = h.float()
        w, mu, ls = self.zaman(h).split(self.K, -1)
        ls = ls.clamp(-6, 4)
        z = (y[..., None] - mu) * torch.exp(-ls)
        bilesen = -0.5 * z * z - ls - 0.5 * LN2PI
        tll = torch.logsumexp(F.log_softmax(w, -1) + bilesen, -1) - y      # ln p(Δt) = ln p(ln Δt) − ln Δt
        mll = -F.cross_entropy(self.buyuk(h).flatten(0, -2), c.flatten(), reduction="none").view_as(y)
        return tll, mll


model = DepremGPT(arg.d, arg.katman, arg.bas, arg.K, arg.dropout).to(cihaz)
n_param = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, betas=(0.9, 0.95), weight_decay=arg.wd)
toplam_adim = arg.adim                      # öğrenme hızı planı hep bu uzunlukta
son_adim = arg.sabit_adim or arg.adim


def lr_carpan(s):
    if s < 200:
        return (s + 1) / 200
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - 200) / max(1, toplam_adim - 200))))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_carpan)


@torch.no_grad()
def degerlendir(hedefler):
    """Her hedef k için bağlam tam olarak k−L..k−1; çıktı yalnız son konumdan."""
    model.eval()
    hedefler = torch.tensor(hedefler, device=cihaz)
    ofs = torch.arange(-L, 0, device=cihaz)
    tll, mll = [], []
    bs = max(1, arg.deger_jeton // L)
    for i in range(0, len(hedefler), bs):
        k = hedefler[i:i + bs]
        idx = k[:, None] + ofs[None]
        with amp_baglami(cihaz):
            h = model(X_g[idx], C_g[idx], T_g[idx])[:, -1]
        a, b = model.ll(h, Y_g[k], C_g[k])
        tll.append(a)
        mll.append(b)
    model.train()
    return torch.cat(tll).cpu().numpy().astype(np.float64), torch.cat(mll).cpu().numpy().astype(np.float64)


# ---------------- eğitim ----------------
B = max(1, arg.hedef_jeton // L)
cikti = KOK / "sonuclar" / arg.ad
cikti.mkdir(parents=True, exist_ok=True)
kayit = {"arg": vars(arg), "n_param": n_param, "B": B, "egitim_hedef": int(egitim_hedef.sum()),
         "pencere_baslangic": int(len(baslar)), "segmentler": [list(map(int, x)) for x in seg],
         "dogrulama": [], "egitim_kaybi": []}
print(f"{arg.ad}: L={L} B={B} param={n_param} hedef={egitim_hedef.sum()}", flush=True)
en_iyi, en_iyi_adim, sabir = -1e9, 0, 0
ofs_e = torch.arange(L + 1, device=cihaz)
bas_zaman = time.time()
kayip_top, kayip_n = 0.0, 0
for adim in range(1, son_adim + 1):
    s = basla_g[torch.randint(0, len(basla_g), (B,), device=cihaz)]
    idx = s[:, None] + ofs_e[None]                      # (B, L+1)
    girdi, hedef = idx[:, :-1], idx[:, 1:]
    with amp_baglami(cihaz):
        h = model(X_g[girdi], C_g[girdi], T_g[girdi])
    tll, mll = model.ll(h, Y_g[hedef], C_g[hedef])
    maske = H_g[hedef].float()
    kayip = -((tll + mll) * maske).sum() / maske.sum().clamp_min(1)
    opt.zero_grad(set_to_none=True)
    kayip.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    kayip_top += kayip.item()
    kayip_n += 1
    if adim % arg.deger_her == 0 or adim == son_adim:
        kayit["egitim_kaybi"].append([adim, kayip_top / kayip_n])
        kayip_top, kayip_n = 0.0, 0
        if arg.sabit_adim:
            print(f"adım {adim} kayıp {kayit['egitim_kaybi'][-1][1]:.4f}", flush=True)
            continue
        vt, vm = degerlendir(dogrulama_idx)
        kayit["dogrulama"].append([adim, float(vt.mean()), float(vm.mean())])
        print(f"adım {adim} kayıp {kayit['egitim_kaybi'][-1][1]:.4f} doğ.TLL {vt.mean():.4f} "
              f"doğ.MLL {vm.mean():.4f} {time.time() - bas_zaman:.0f}s", flush=True)
        if vt.mean() > en_iyi:
            en_iyi, en_iyi_adim, sabir = float(vt.mean()), adim, 0
            torch.save(model.state_dict(), cikti / "en_iyi.pt")
        else:
            sabir += 1
            if sabir >= arg.sabir:
                print("erken durdurma", flush=True)
                break

if arg.sabit_adim:
    torch.save(model.state_dict(), cikti / "en_iyi.pt")
    en_iyi_adim = son_adim
model.load_state_dict(torch.load(cikti / "en_iyi.pt", map_location=cihaz))
tt, tm = degerlendir(test_idx)
kayit.update({"en_iyi_adim": en_iyi_adim, "en_iyi_dogrulama_tll": en_iyi,
              "test_tll": float(tt.mean()), "test_mll": float(tm.mean()), "test_n": int(len(tt)),
              "sure_s": time.time() - bas_zaman})
np.savez(cikti / "test.npz", idx=test_idx, tll=tt, mll=tm)
json.dump(kayit, open(cikti / "kayit.json", "w"), indent=1)
print(f"TEST TLL {tt.mean():.4f}  MLL {tm.mean():.4f}  (en iyi adım {en_iyi_adim})", flush=True)
print("IS_BITTI", flush=True)
