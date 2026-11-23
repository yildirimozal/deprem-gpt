import py_compile
from pathlib import Path

import numpy as np
from scipy import integrate

KOK = Path(__file__).resolve().parent.parent


def test_all_kod_scripts_compile():
    betikler = sorted((KOK / "kod").glob("*.py"))
    assert len(betikler) == 37
    for yol in betikler:
        py_compile.compile(str(yol), doraise=True)


def test_hazirla_cok_helpers():
    # hazirla_konum.py aynı ayırma noktasını kullandığı için bu sözleşmeyi de doğrular
    kaynak = (KOK / "kod/hazirla_cok.py").read_text(encoding="utf-8").split('t0 = pd.Timestamp("1971-01-01")')[0]
    ns = {"__file__": str(KOK / "kod/hazirla_cok.py")}
    exec(kaynak, ns)  # noqa: S102

    assert np.allclose(ns["yuvarla"]([2.44, 2.45, 2.55]), [2.4, 2.5, 2.6])
    assert np.isclose(ns["mc_maxc"](np.array([2.5, 2.6, 2.6, 2.7])), 2.8)


def test_sentetik_etas_helpers():
    kaynak = (KOK / "kod/sentetik_etas.py").read_text(encoding="utf-8").split("def katalog():")[0]
    ns = {"__file__": str(KOK / "kod/sentetik_etas.py")}
    exec(kaynak, ns)  # noqa: S102

    m = ns["gr"](500, 1.0 * np.log(10))
    assert m.shape == (500,)
    assert np.all((m >= -0.05) & (m <= ns["M_UST"]))

    c, p, ust = 0.02, 1.15, 10.0
    beklenen, _ = integrate.quad(lambda x: (x + c) ** (-p), 0.0, ust)
    assert np.isclose(ns["omori_integral"](c, p, ust), beklenen, rtol=1e-9)

    dt = ns["omori_ornek"](500, c, p, np.full(500, ust))
    assert dt.shape == (500,)
    assert np.all((dt >= 0.0) & (dt <= ust))


def test_egit_split_markers():
    ayrac = "# ---------------- eğitim ----------------"
    for yol in ("kod/egit.py", "kod/egit3.py"):
        metin = (KOK / yol).read_text(encoding="utf-8")
        assert ayrac in metin, f"{yol} içinde beklenen ayraç bulunamadı"
