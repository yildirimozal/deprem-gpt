import json
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent
SONUCLAR = KOK / "sonuclar"


def test_all_summary_jsons_valid():
    dosyalar = sorted(SONUCLAR.glob("*.json"))
    assert dosyalar, "sonuclar/ içinde JSON dosyası bulunamadı"
    for yol in dosyalar:
        with open(yol, encoding="utf-8") as f:
            veri = json.load(f)
        assert isinstance(veri, dict) and veri, f"{yol.name} boş veya sözlük değil"


def test_sekil_verisi_integrity():
    with np.load(SONUCLAR / "sekil_verisi.npz") as z:
        beklenen_f1 = {
            "TR": 4264,
            "JP": 4362,
            "TW": 5724,
        }
        for te, n in beklenen_f1.items():
            for ek in ("gun", "cok_etas_kumulatif", "tek_etas_kumulatif", "cok_kahin_kumulatif"):
                anahtar = f"F1_{te}_{ek}"
                assert anahtar in z.files
                dizi = z[anahtar]
                assert dizi.shape == (n,)
                assert np.all(np.isfinite(dizi))
            assert np.all(np.diff(z[f"F1_{te}_gun"]) >= 0)

        for ek in ("saat", "gozlenen", "etas_altkritik_beklenen", "etas_kahin_beklenen", "tek_beklenen", "cok_beklenen"):
            anahtar = f"F2a_{ek}"
            assert anahtar in z.files
            dizi = z[anahtar]
            assert dizi.shape == (937,)
            assert np.all(np.isfinite(dizi))
        assert np.all(np.diff(z["F2a_saat"]) >= 0)


def test_makale_sayilar_matches_summaries():
    with open(SONUCLAR / "makale_sayilar.json", encoding="utf-8") as f:
        ms = json.load(f)
    with open(SONUCLAR / "c5_ozet.json", encoding="utf-8") as f:
        c5 = json.load(f)
    with open(SONUCLAR / "c9_ozet.json", encoding="utf-8") as f:
        c9 = json.load(f)
    with np.load(SONUCLAR / "sekil_verisi.npz") as sv:
        for te in ("TR", "JP", "TW"):
            assert ms[te]["n_test"] == len(sv[f"F1_{te}_gun"])
            assert np.isclose(sv[f"F1_{te}_cok_etas_kumulatif"][-1] / ms[te]["n_test"], ms[te]["cok-etas_tum"])
            assert np.isclose(sv[f"F1_{te}_tek_etas_kumulatif"][-1] / ms[te]["n_test"], ms[te]["tek-etas_tum"])
            assert np.isclose(sv[f"F1_{te}_cok_kahin_kumulatif"][-1] / ms[te]["n_test"], ms[te]["cok-kahin_tum"])

    assert ms["mekanizma_TR"]["aralik"] == c5["aralik"]
    assert ms["olasilik"]["karar"] == c9["karar"]


def test_preregistered_decisions_present():
    for ad in (
        "c3_ozet.json",
        "c4_ozet.json",
        "c6_ozet.json",
        "c7_ozet.json",
        "c8_ozet.json",
        "c9_ozet.json",
        "c10_ozet.json",
        "c11_ozet.json",
    ):
        with open(SONUCLAR / ad, encoding="utf-8") as f:
            veri = json.load(f)
        assert "karar" in veri, f"{ad} içinde 'karar' anahtarı eksik"
