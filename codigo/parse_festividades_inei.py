"""
parse_festividades_inei.py - district festival calendar from INEI (2013),
"Directorio Nacional de Principales Festividades a Nivel Distrital" (Lib1107).
Reads the PDF by line positions: one row per festival with district, name, date (DD MON) and
days of duration; department and province come from header rows. Names are matched to the
INEI 2025 district polygons (accent- and punctuation-insensitive) to obtain UBIGEO.
Output: raw/festividades/festividades_distritales_2013.csv
"""
from __future__ import annotations
from pathlib import Path
import os
import re
import unicodedata
import fitz
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "raw" / "festividades" / "festividades_inei_2013.pdf"
SHP = Path(os.environ.get("INEI_DISTRICT_SHP", str(Path(__file__).resolve().parents[1] / "raw" / "geo" / "Limite_Distrital_INEI_2025_CPV.shp")))
OUTF = ROOT / "raw" / "festividades" / "festividades_distritales_2013.csv"
MON = {"ENE": 1, "FEB": 2, "MAR": 3, "ABR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AGO": 8, "SET": 9, "SEP": 9,
       "OCT": 10, "NOV": 11, "DIC": 12}
DEPS = {"AMAZONAS", "ANCASH", "APURIMAC", "AREQUIPA", "AYACUCHO", "CAJAMARCA", "CALLAO", "PROV CONST DEL CALLAO",
        "PROVINCIA CONSTITUCIONAL DEL CALLAO", "CUSCO", "HUANCAVELICA", "HUANUCO", "ICA", "JUNIN", "LA LIBERTAD",
        "LAMBAYEQUE", "LIMA", "LORETO", "MADRE DE DIOS", "MOQUEGUA", "PASCO", "PIURA", "PUNO", "SAN MARTIN",
        "TACNA", "TUMBES", "UCAYALI"}
DATE = re.compile(r"^(\d{1,2})\s+([A-Z]{3})\.?$")


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def rows_of(page):
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            t = "".join(s["text"] for s in l["spans"]).strip()
            if t:
                lines.append((round(l["bbox"][1] / 3), l["bbox"][0], t))
    rows = {}
    for y, x, t in lines:
        rows.setdefault(y, []).append((x, t))
    return [sorted(v) for _, v in sorted(rows.items())]


def parse():
    doc = fitz.open(str(PDF))
    recs, dep, prov, dist = [], None, None, None
    for pno, page in enumerate(doc):
        txt = page.get_text()
        if "Nombre de la Festividad" not in txt or "FECHA DE CELEBRACI" in txt:
            continue  # the second half of the book repeats the same festivals sorted by date
        rows = rows_of(page)
        date_rows = [r for r in rows if any(DATE.match(t) for _, t in r)]
        if not date_rows:
            continue
        xd = sorted(r[0][0] for r in date_rows if len(r) >= 3)
        x_dist = xd[len(xd) // 2]
        xn = sorted(r[1][0] for r in date_rows if len(r) >= 4)
        x_name = xn[len(xn) // 2] if xn else x_dist + 100
        for r in rows:
            toks = [t for _, t in r]
            if any(t.startswith(("Nombre de la", "Fecha de", "Celebraci", "Dias de", "Duraci", "Continúa", "PRINCIPALES",
                                 "Departamento /", "- ", "Fuente", "Nota", "Elaboraci")) for t in toks):
                continue
            di = [i for i, t in enumerate(toks) if DATE.match(t)]
            if di:
                i = di[0]
                m = DATE.match(toks[i])
                days = toks[i + 1] if i + 1 < len(toks) and toks[i + 1].isdigit() else None
                left = r[:i]
                dist_tok = [t for x, t in left if abs(x - x_dist) < 15]
                name_tok = [t for x, t in left if x >= x_name - 15]
                if dist_tok:
                    dist = dist_tok[0]
                recs.append(dict(page=pno + 1, dep=dep, prov=prov, dist=dist, fiesta=" ".join(name_tok).strip(),
                                 day=int(m.group(1)), month=MON.get(m.group(2)), days=int(days) if days else None))
            elif len(r) == 1:
                x, t = r[0]
                if x < x_dist - 10:
                    if norm(t) in DEPS and (dep is None or norm(t) != norm(dep)):
                        dep, prov, dist = t, None, None
                    else:
                        prov, dist = t, None  # never inherit the previous province's district
                elif x >= x_name - 15 and recs:
                    recs[-1]["fiesta"] += " " + t  # wrapped name
                elif abs(x - x_dist) < 15:
                    dist = t
    return pd.DataFrame(recs)


def main():
    f = parse()
    # when the duration sits on a slightly higher text line it is read as the end of the name
    # (checked on the PDF: Capachica, Virgen de la Candelaria, 02 FEB, 01 day)
    tail = f.fiesta.str.extract(r"^(.*\S)\s+(\d{1,2})$")
    fix = f.days.isna() & tail[1].notna()
    f["duration_recovered"] = fix.astype(int)
    f.loc[fix, "days"] = tail.loc[fix, 1].astype(int)
    f.loc[fix, "fiesta"] = tail.loc[fix, 0]
    n0 = len(f)
    f = f.drop_duplicates(["dep", "prov", "dist", "fiesta", "day", "month", "days"]).reset_index(drop=True)
    print(f"durations recovered from the name: {int(fix.sum())}; exact duplicates dropped: {n0 - len(f)}")
    geo = gpd.read_file(SHP).drop(columns="geometry")
    geo["k"] = geo.DEPARTAMEN.map(norm) + "|" + geo.PROVINCIA.map(norm) + "|" + geo.DISTRITO.map(norm)
    geo["k2"] = geo.DEPARTAMEN.map(norm) + "|" + geo.DISTRITO.map(norm)
    f["k"] = f.dep.map(norm) + "|" + f.prov.map(norm) + "|" + f.dist.map(norm)
    f["k2"] = f.dep.map(norm) + "|" + f.dist.map(norm)
    geo["k3"] = geo.PROVINCIA.map(norm) + "|" + geo.DISTRITO.map(norm)
    f["k3"] = f.prov.map(norm) + "|" + f.dist.map(norm)
    f = f.merge(geo[["k", "UBIGEO"]], on="k", how="left")
    u2 = geo.drop_duplicates("k2", keep=False).set_index("k2").UBIGEO
    u3 = geo.drop_duplicates("k3", keep=False).set_index("k3").UBIGEO
    f["match"] = np.where(f.UBIGEO.notna(), "dep|prov|dist", None)
    # dep|dist matches are accepted only for documented province renames or splits after 2013
    ok_prov = {("CHAVINILLO", "YAROWILCA"), ("NAZCA", "NASCA"), ("MAYNAS", "PUTUMAYO")}
    prov_of = geo.set_index("UBIGEO").PROVINCIA.map(norm)
    for key, u, lab in [("k3", u3, "prov|dist"), ("k2", u2, "dep|dist")]:
        new = f.UBIGEO.isna() & f[key].map(u).notna()
        if lab == "dep|dist":
            cand = f[key].map(u)
            new &= pd.Series([(norm(p), prov_of.get(c, "")) in ok_prov for p, c in zip(f.prov, cand)], index=f.index)
        f.loc[new, "UBIGEO"] = f.loc[new, key].map(u)
        f.loc[new, "match"] = lab
    print(f.match.value_counts(dropna=False).to_dict())
    print(f"festivals {len(f):,}  districts named {f.k.nunique():,}  matched to UBIGEO {f.UBIGEO.notna().mean():.3f}  "
          f"distinct UBIGEO {f.UBIGEO.nunique():,}  missing month {f.month.isna().sum()}  missing days {f.days.isna().sum()}")
    print("unmatched districts:", f[f.UBIGEO.isna()].drop_duplicates("k").k.tolist())
    part = f[f.match.isin(["prov|dist", "dep|dist"])].drop_duplicates("k")
    print("partial matches to review:")
    print(part[["dep", "prov", "dist", "UBIGEO", "match"]].merge(geo[["UBIGEO", "PROVINCIA"]], on="UBIGEO", how="left").to_string())
    f.drop(columns=["k", "k2", "k3"]).to_csv(OUTF, index=False, encoding="utf-8")
    pu = f[f.dep.map(norm) == "PUNO"]
    print("Puno festivals", len(pu), "districts", pu.UBIGEO.nunique())
    print(pu.head(12).to_string())


if __name__ == "__main__":
    main()
