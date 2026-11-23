"""California (EarthquakeNPP), İtalya (INGV) ve Yeni Zelanda (GeoNet) kataloglarını indirir.

Kaynaklar (README.md ve ON_KAYIT_2.md):
  - California: EarthquakeNPP ComCat M >= 2.5 CSV -> veri/earthquakenpp/Experiments/ETAS/output_data_ComCat_25/
  - İtalya    : INGV FDSN (2005–2025, 35–48°K, 6–19°D, M >= 2.5) -> veri/ingv/ingv_<yil>.txt
  - Y.Zelanda : GeoNet FDSN (2012–2025, 48–34°G, 165–179°D, M >= 3.0) -> veri/geonet/geonet_<yil>.txt

Kullanım:
  python kod/indir_ek.py --katalog hepsi
  python kod/indir_ek.py --katalog ingv --bas 2020 --son 2025
"""
import argparse
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
CA_URL = "https://raw.githubusercontent.com/ss15859/EarthquakeNPP/main/Experiments/ETAS/output_data_ComCat_25/augmented_catalog.csv"
CA_YOL = KOK / "veri/earthquakenpp/Experiments/ETAS/output_data_ComCat_25/augmented_catalog.csv"
FDSN = {
    "ingv": dict(
        url="https://webservices.ingv.it/fdsnws/event/1/query",
        bas=2005,
        son=2025,
        kutu=(35, 48, 6, 19),
        m_min=2.5,
        dizin=KOK / "veri/ingv",
    ),
    "geonet": dict(
        url="https://service.geonet.org.nz/fdsnws/event/1/query",
        bas=2012,
        son=2025,
        kutu=(-48, -34, 165, 179),
        m_min=3.0,
        dizin=KOK / "veri/geonet",
    ),
}


def _ssl_baglami():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def getir_bayt(url, baglam, deneme_say=4):
    for deneme in range(deneme_say):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "deprem-arastirma"})
            with urllib.request.urlopen(req, context=baglam, timeout=180) as r:
                if r.status == 204:
                    return b""
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 204:
                return b""
            print("hata", url, e, flush=True)
        except Exception as e:  # noqa: BLE001
            print("hata", url, e, flush=True)
        time.sleep(5 * (deneme + 1))
    raise SystemExit(f"indirilemedi: {url}")


def indir_ca(baglam, uzerine=False):
    if CA_YOL.exists() and not uzerine:
        print(f"atlandı (mevcut): {CA_YOL}", flush=True)
        return
    CA_YOL.parent.mkdir(parents=True, exist_ok=True)
    veri = getir_bayt(CA_URL, baglam)
    CA_YOL.write_bytes(veri)
    print(f"ca: {len(veri)} bayt -> {CA_YOL}", flush=True)


def indir_fdsn(ad, baglam, bas=None, son=None, uzerine=False):
    c = FDSN[ad]
    c["dizin"].mkdir(parents=True, exist_ok=True)
    y0 = bas or c["bas"]
    y1 = son or c["son"]
    lat1, lat2, lon1, lon2 = c["kutu"]
    for yil in range(y0, y1 + 1):
        yol = c["dizin"] / f"{ad}_{yil}.txt"
        if yol.exists() and not uzerine:
            print(f"atlandı (mevcut): {yol.name}", flush=True)
            continue
        url = (
            f"{c['url']}?starttime={yil}-01-01T00:00:00&endtime={yil}-12-31T23:59:59"
            f"&minlat={lat1}&maxlat={lat2}&minlon={lon1}&maxlon={lon2}"
            f"&minmagnitude={c['m_min']}&format=text"
        )
        veri = getir_bayt(url, baglam)
        if not veri.strip():
            print(f"{ad} {yil}: boş yanıt (204)", flush=True)
            continue
        yol.write_bytes(veri)
        n_satir = max(0, len(veri.splitlines()) - 1)
        print(f"{ad} {yil}: {n_satir} satır -> {yol.name}", flush=True)
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser(description="EarthquakeNPP (CA), INGV ve GeoNet kataloglarını indirir.")
    ap.add_argument("--katalog", default="hepsi", choices=["hepsi", "ca", "ingv", "geonet"])
    ap.add_argument("--bas", type=int, default=None, help="FDSN başlangıç yılı (varsayılan katalog ayarı)")
    ap.add_argument("--son", type=int, default=None, help="FDSN bitiş yılı (varsayılan katalog ayarı)")
    ap.add_argument("--uzerine", action="store_true", help="Mevcut dosyaları yeniden indir")
    arg = ap.parse_args()

    baglam = _ssl_baglami()
    if arg.katalog in ("hepsi", "ca"):
        indir_ca(baglam, uzerine=arg.uzerine)
    if arg.katalog in ("hepsi", "ingv"):
        indir_fdsn("ingv", baglam, arg.bas, arg.son, uzerine=arg.uzerine)
    if arg.katalog in ("hepsi", "geonet"):
        indir_fdsn("geonet", baglam, arg.bas, arg.son, uzerine=arg.uzerine)


if __name__ == "__main__":
    main()
