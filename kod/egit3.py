"""Deprem GPT, konum çıktılı sürüm (ON_KAYIT_10.md): egit2.py + --konum_cikti (bir sonraki olayın yeri için
16 bileşenli 2B Gauss karışımı; ortalamalar son olayın konumuna göre öteleme). SLL 1000 km birimindedir;
km² başına çevirmek için 2·ln(1000) çıkarılır. Bayrak kapalıyken egit2.py ile aynı.

Önceki açıklama (egit2.py): mimari testi sürümü (ON_KAYIT_8.md). Bayraklar kapalıyken egit.py ile aynı model ve aynı
rastgelelik sırası. Ek bayraklar:
  --katalog_gomme   her jetona katalog kimliği gömmesi eklenir
  --konum           girdiye konum özellikleri (x, y [1000 km], log10(önceki olaya uzaklık + 1 km)/3, bayrak)
  --on_egitim npz   önce bu sentetik kataloglarda --on_adim adım ön eğitim, sonra gerçek veriyle eğitim

Deprem GPT: her deprem bir jeton, sonraki depremin zamanı ve büyüklüğü için dağılım.

Kullanım:
  python egit.py --L 4096 --tohum 1 --ad L4096_t1
  python egit.py --L 4096 --tohum 1 --ad tam_L4096_t1 --egitim_bolum 1,2 --sabit_adim 1200

Çıktı (sonuclar/<ad>/): kayit.json, test.npz (olay başına zamansal ve büyüklük LL), en_iyi.pt
"""
import argparse
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
ap.add_argument("--katalog_gomme", action="store_true")
ap.add_argument("--konum", action="store_true")
ap.add_argument("--konum_cikti", action="store_true")
ap.add_argument("--uzay_isaretci", action="store_true", help="keşif (ön kayıt dışı): bağlamdaki olaylara işaretçi + arka plan karışımı")
ap.add_argument("--on_egitim", default="")
ap.add_argument("--on_adim", type=int, default=3000)
arg = ap.parse_args()
if arg.uzay_isaretci:
    arg.konum_cikti = True
if arg.konum_cikti:
    arg.konum = True
KS = 16
LS_ALT, LS_UST = math.log(0.0005), math.log(2.0)          # 0.5 km .. 2000 km (1000 km biriminde)

torch.manual_seed(arg.tohum)
np.random.seed(arg.tohum)
cihaz = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------- veri ----------------
# California önce gelir; doğrulama/test indeksleri ek katalog eklense de değişmez.
v = np.load(KOK / "veri" / arg.ana)
MC_ANA = float(v["mc"]) if "mc" in v.files else 2.5
t_ca = v["t_gun"].astype(np.float64)
bolum_ca = v["bolum"]
ANA_AD = {"turkiye.npz": "Turkiye_KOERI", "japonya.npz": "Japonya_JMA", "olaylar.npz": "California_ComCat", "tayvan.npz": "Tayvan_TAP"}[arg.ana]
KAT_NO = {"Turkiye_KOERI": 0, "Japonya_JMA": 1, "California_ComCat": 2, "Italya_INGV": 3, "YeniZelanda_GeoNet": 4, "Sentetik": 5, "Tayvan_TAP": 6}
seg = [(0, len(t_ca))]
seg_ad = [ANA_AD]
t_l, mg_l, bolum_l = [t_ca], [np.round(v["buyukluk"] - MC_ANA, 1)], [bolum_ca]
if arg.ek:
    e = np.load(KOK / "veri" / arg.ek)
    for a, b, ad in zip(e["seg_bas"], e["seg_son"], e["seg_ad"]):
        n0 = sum(len(x) for x in t_l)
        t_l.append(e["t_gun"][a:b])
        mg_l.append(e["m_goreli"][a:b])
        bolum_l.append(np.full(b - a, 5, np.int8))      # 5: ek katalog, hep eğitim hedefi
        seg.append((n0, n0 + b - a))
        seg_ad.append(str(ad))
SENTETIK = []
if arg.on_egitim:
    s_ = np.load(KOK / "veri" / arg.on_egitim)
    for a, b in zip(s_["seg_bas"], s_["seg_son"]):
        n0 = sum(len(x) for x in t_l)
        t_l.append(s_["t_gun"][a:b])
        mg_l.append(s_["m_goreli"][a:b])
        bolum_l.append(np.full(b - a, 6, np.int8))      # 6: sentetik, yalnızca ön eğitim hedefi
        SENTETIK.append((n0, n0 + b - a))
        seg_ad.append("Sentetik")
t = np.concatenate(t_l)
mg = np.concatenate(mg_l)
bolum = np.concatenate(bolum_l)
dt = np.concatenate([np.diff(x, prepend=x[0] - 1.0) for x in t_l])   # her katalog kendi içinde
dt = np.maximum(dt, DT_MIN)
ozellik = np.stack([(np.log10(dt) + 1.0) / 1.5, mg], 1).astype(np.float32)
tum_seg = seg + SENTETIK
kat_id = np.concatenate([np.full(b - a, KAT_NO[ad], np.int64) for (a, b), ad in zip(tum_seg, seg_ad)])
if arg.konum:
    kn = np.load(KOK / "veri/konum.npz")
    konum = np.zeros((len(t), 4), np.float32)
    for (a, b), ad in zip(tum_seg, seg_ad):
        if ad == "Sentetik":
            continue
        assert np.array_equal(kn[f"{ad}_t"][:b - a], t[a:b]), ad
        la, lo = kn[f"{ad}_enlem"][:b - a], kn[f"{ad}_boylam"][:b - a]
        la0, lo0 = np.median(kn[f"{ad}_enlem"]), np.median(kn[f"{ad}_boylam"])
        x = (lo - lo0) * np.cos(np.radians(la0)) * 111.2 / 1000
        y = (la - la0) * 111.2 / 1000
        dx = np.diff(lo, prepend=lo[0]) * np.cos(np.radians(la)) * 111.2
        dy = np.diff(la, prepend=la[0]) * 111.2
        konum[a:b] = np.stack([x, y, np.log10(np.hypot(dx, dy) + 1) / 3, np.ones(b - a)], 1)
    ozellik = np.concatenate([ozellik, konum], 1)
sinif = np.clip(np.round(mg / 0.1), 0, N_BUYUK - 1).astype(np.int64)

T_g = torch.tensor(t, dtype=torch.float64, device=cihaz)
X_g = torch.tensor(ozellik, device=cihaz)
C_g = torch.tensor(sinif, device=cihaz)
Y_g = torch.tensor(np.log(dt), dtype=torch.float32, device=cihaz)   # ln Δt (gün)
K_g = torch.tensor(kat_id, device=cihaz)
P_g = torch.tensor(konum[:, :2], device=cihaz) if arg.konum else None

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
on_baslar = np.concatenate([np.arange(a, b - L) for a, b in SENTETIK if b - a > L + 1]) if SENTETIK else None
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
        self.giris = nn.Sequential(nn.Linear(2 + (4 if arg.konum else 0), d), nn.GELU(), nn.Linear(d, d))
        self.m_gom = nn.Embedding(N_BUYUK, d)
        self.konum = Konum(d // h)
        self.bloklar = nn.ModuleList([Blok(d, h, p) for _ in range(n_k)])
        self.ln = nn.LayerNorm(d)
        self.zaman = nn.Linear(d, 3 * K)
        self.buyuk = nn.Linear(d, N_BUYUK)
        self.K = K
        with torch.no_grad():                    # karışım ortalamalarını ln Δt aralığına yay
            self.zaman.bias[K:2 * K] = torch.linspace(-12, 2, K)
        if arg.katalog_gomme:                    # yalnızca bayrak açıkken: rastgelelik sırası egit.py ile aynı kalsın
            self.k_gom = nn.Embedding(len(KAT_NO), d)
        if arg.uzay_isaretci:
            self.uq, self.uk = nn.Linear(d, 64), nn.Linear(d, 64)
            self.us, self.ubg = nn.Linear(d, 1), nn.Linear(d, 1)
            self.uzay = nn.Linear(d, 5 * 8)                       # arka plan: mutlak ortalamalı 8 bileşen
            with torch.no_grad():
                self.us.bias.fill_(math.log(0.02))
                self.uzay.bias[3 * 8:] = math.log(0.2)
        elif arg.konum_cikti:
            self.uzay = nn.Linear(d, 5 * KS)
            with torch.no_grad():
                self.uzay.bias[KS:3 * KS] = 0.0
                self.uzay.bias[3 * KS:] = math.log(0.05)

    def forward(self, x, c, tz, k=None):
        h = self.giris(x) + self.m_gom(c)
        if arg.katalog_gomme:
            h = h + self.k_gom(k)
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

    def sll_isaretci(self, hq, hk, xy, xy_k, maske):
        """hq (B,Tq,d) hedef konumları, hk (B,Tk,d) bağlam; xy (B,Tq,2), xy_k (B,Tk,2); maske (Tq,Tk) izinli."""
        hq, hk = hq.float(), hk.float()
        skor = self.uq(hq) @ self.uk(hk).transpose(1, 2) / 8.0
        skor = skor.masked_fill(~maske, float("-inf"))
        lpi = torch.log_softmax(torch.cat([self.ubg(hq), skor], -1), -1)        # (B,Tq,1+Tk)
        ls = self.us(hk).clamp(LS_ALT, LS_UST).transpose(1, 2)                  # (B,1,Tk)
        d2 = ((xy[:, :, None, :] - xy_k[:, None, :, :]) ** 2).sum(-1)
        lnk = -0.5 * d2 * torch.exp(-2 * ls) - 2 * ls - LN2PI
        w, mx, my, lx, ly = self.uzay(hq).split(8, -1)
        lx, ly = lx.clamp(LS_ALT, LS_UST), ly.clamp(LS_ALT, LS_UST)
        zx, zy = (xy[..., :1] - mx) * torch.exp(-lx), (xy[..., 1:] - my) * torch.exp(-ly)
        bg = torch.logsumexp(F.log_softmax(w, -1) - 0.5 * (zx * zx + zy * zy) - lx - ly - LN2PI, -1)
        return torch.logsumexp(torch.cat([(lpi[..., :1] + bg[..., None]), lpi[..., 1:] + lnk], -1), -1)

    def sll(self, h, xy, son):
        """Bir sonraki olayın konumu için log yoğunluk (1000 km biriminde). xy, son: (..., 2)."""
        h = h.float()
        w, ox, oy, lx, ly = self.uzay(h).split(KS, -1)
        lx, ly = lx.clamp(LS_ALT, LS_UST), ly.clamp(LS_ALT, LS_UST)
        zx = (xy[..., :1] - son[..., :1] - ox) * torch.exp(-lx)
        zy = (xy[..., 1:] - son[..., 1:] - oy) * torch.exp(-ly)
        bil = -0.5 * (zx * zx + zy * zy) - lx - ly - LN2PI
        return torch.logsumexp(F.log_softmax(w, -1) + bil, -1)


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
    tll, mll, sll = [], [], []
    bs = max(1, arg.deger_jeton // L)
    for i in range(0, len(hedefler), bs):
        k = hedefler[i:i + bs]
        idx = k[:, None] + ofs[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h_tum = model(X_g[idx], C_g[idx], T_g[idx], K_g[idx])
        h = h_tum[:, -1]
        a, b = model.ll(h, Y_g[k], C_g[k])
        tll.append(a)
        mll.append(b)
        if arg.uzay_isaretci:
            sll.append(model.sll_isaretci(h[:, None], h_tum, P_g[k][:, None], P_g[idx],
                                          torch.ones(1, L, dtype=torch.bool, device=cihaz))[:, 0])
        elif arg.konum_cikti:
            sll.append(model.sll(h, P_g[k], P_g[k - 1]))
    model.train()
    son = (torch.cat(sll).cpu().numpy().astype(np.float64),) if arg.konum_cikti else ()
    return (torch.cat(tll).cpu().numpy().astype(np.float64), torch.cat(mll).cpu().numpy().astype(np.float64)) + son


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
if SENTETIK:
    on_g = torch.tensor(on_baslar, device=cihaz)
    opt_on = torch.optim.AdamW(model.parameters(), lr=arg.lr, betas=(0.9, 0.95), weight_decay=arg.wd)

    def on_carpan(s):
        if s < 200:
            return (s + 1) / 200
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - 200) / max(1, arg.on_adim - 200))))

    sch_on = torch.optim.lr_scheduler.LambdaLR(opt_on, on_carpan)
    kayit["on_egitim_kaybi"] = []
    top_on = 0.0
    for adim in range(1, arg.on_adim + 1):
        s = on_g[torch.randint(0, len(on_g), (B,), device=cihaz)]
        idx = s[:, None] + ofs_e[None]
        girdi, hedef = idx[:, :-1], idx[:, 1:]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(X_g[girdi], C_g[girdi], T_g[girdi], K_g[girdi])
        tll, mll = model.ll(h, Y_g[hedef], C_g[hedef])
        kayip = -(tll + mll).mean()
        opt_on.zero_grad(set_to_none=True)
        kayip.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_on.step()
        sch_on.step()
        top_on += kayip.item()
        if adim % 250 == 0:
            vt = degerlendir(dogrulama_idx)[0]
            kayit["on_egitim_kaybi"].append([adim, top_on / 250, float(vt.mean())])
            print(f"ön eğitim adım {adim} kayıp {top_on / 250:.4f} doğ.TLL {vt.mean():.4f} {time.time() - bas_zaman:.0f}s", flush=True)
            top_on = 0.0
    del opt_on, sch_on
kayip_top, kayip_n = 0.0, 0
NEDENSEL = torch.tril(torch.ones(L, L, dtype=torch.bool, device=cihaz))     # hedef i, girdi 0..i (i = önceki olay)
for adim in range(1, son_adim + 1):
    s = basla_g[torch.randint(0, len(basla_g), (B,), device=cihaz)]
    idx = s[:, None] + ofs_e[None]                      # (B, L+1)
    girdi, hedef = idx[:, :-1], idx[:, 1:]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(X_g[girdi], C_g[girdi], T_g[girdi], K_g[girdi])
    tll, mll = model.ll(h, Y_g[hedef], C_g[hedef])
    if arg.uzay_isaretci:
        tll = tll + model.sll_isaretci(h, h, P_g[hedef], P_g[girdi], NEDENSEL)
    elif arg.konum_cikti:
        tll = tll + model.sll(h, P_g[hedef], P_g[girdi])      # son olay = hedefin bir öncesi = girdi konumu
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
        r_ = degerlendir(dogrulama_idx)
        vt, vm = r_[0], r_[1]
        vs = r_[2] if arg.konum_cikti else np.zeros(1)
        olcut = float(vt.mean() + vs.mean())                    # konum çıktısında erken durdurma TLL + SLL ile
        kayit["dogrulama"].append([adim, float(vt.mean()), float(vm.mean())] + ([float(vs.mean())] if arg.konum_cikti else []))
        print(f"adım {adim} kayıp {kayit['egitim_kaybi'][-1][1]:.4f} doğ.TLL {vt.mean():.4f} "
              f"doğ.MLL {vm.mean():.4f}" + (f" doğ.SLL {vs.mean():.4f}" if arg.konum_cikti else "") + f" {time.time() - bas_zaman:.0f}s", flush=True)
        if olcut > en_iyi:
            en_iyi, en_iyi_adim, sabir = olcut, adim, 0
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
r_ = degerlendir(test_idx)
tt, tm = r_[0], r_[1]
kayit.update({"en_iyi_adim": en_iyi_adim, "en_iyi_dogrulama_tll": en_iyi,
              "test_tll": float(tt.mean()), "test_mll": float(tm.mean()), "test_n": int(len(tt)),
              "sure_s": time.time() - bas_zaman})
ek_ = {}
if arg.konum_cikti:
    ek_["sll"] = r_[2]
    kayit["test_sll_km2"] = float(r_[2].mean() - 2 * math.log(1000))
    print(f"TEST SLL (km²) {kayit['test_sll_km2']:.4f}", flush=True)
np.savez(cikti / "test.npz", idx=test_idx, tll=tt, mll=tm, **ek_)
with open(cikti / "kayit.json", "w", encoding="utf-8") as f:
    json.dump(kayit, f, indent=1)
print(f"TEST TLL {tt.mean():.4f}  MLL {tm.mean():.4f}  (en iyi adım {en_iyi_adim})", flush=True)
print("IS_BITTI", flush=True)
