"""etas_guclu.py'nin ilk sürümü havuzlu koşularda (E3, E5) etas.json yazarken çöktü (numpy int64 JSON'a
yazılamadı); etas_test.npz bundan önce kaydedilmişti. Bu betik etas.json'u koşunun günlüğündeki üç
başlangıç satırından yeniden kurar (en yüksek uydurma TLL'si seçilir, kodla aynı kural).
Kullanım: /usr/bin/python3 kod/etas_json_onar.py kayitlar/c4_g3.log etas4_E5
"""
import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parent.parent
log, ad = sys.argv[1], sys.argv[2]
with open(KOK / log, encoding="utf-8", errors="replace") as f:
    satir = [json.loads(x[x.index("{"):x.rindex("}") + 1]) for x in f if x.startswith('{"baslangic"')]
assert len(satir) == 3, len(satir)
en = max(satir, key=lambda r: r["uydurma_tll"])
tt = np.load(KOK / "sonuclar" / ad / "etas_test.npz")["tll"]
with open(KOK / "sonuclar" / ad / "etas.json", "w", encoding="utf-8") as f:
    json.dump({"not": f"{log} günlüğünden yeniden kuruldu (özgün yazım int64 hatasıyla çöktü)", "denemeler": satir,
               "secilen": en, "test_tll": float(tt.mean()), "n_test": int(len(tt))}, f, indent=1)
print(ad, "test", round(float(tt.mean()), 4), "dallanma", round(en["dallanma"], 3), "uydurma", [round(r["uydurma_tll"], 6) for r in satir])
