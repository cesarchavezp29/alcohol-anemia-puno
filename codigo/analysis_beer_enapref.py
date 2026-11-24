"""
analysis_beer_enapref.py - beer liters per person from the ENAPREF 2019-2020 7-day diaries
(INEI survey 753): household diary (CAP 401-404) + personal diary (CAP 500), which also
captures beer bought and drunk outside the home (bars, restaurants, fiestas).

Beer = product codes whose description contains "CERVEZA" in the household diary
(COICOP 0213... beer for home; 1111... beer served). The same code list is applied to the
personal diary (which has codes only). Liters: P400_EQUIV 3 = liters (ENT + DEC/1000),
4 = milliliters; otherwise imputed from the amount spent and the median price per liter of
beer records with known volume (same code family). Annual = 7-day diary x 52.
Per person = sum(liters x FACTORPOBLACION) / sum(MIE x FACTORPOBLACION).
ENAPREF is representative for the national total, urban/rural, Lima Metropolitana and the
cities (department capitals), not for whole departments.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

R = Path(__file__).resolve().parents[1] / "raw" / "enapref" / "2020"
OUT = Path(__file__).resolve().parents[1] / "analysis" / "alcohol_anemia"
K = ["CONGLOMERADO", "NSELV", "HOGAR_ID"]


def num(ent, dec):
    e = pd.to_numeric(ent, errors="coerce")
    d = pd.to_numeric(pd.Series(dec).astype(str).str.strip().replace({"": "0", "nan": "0"}), errors="coerce").fillna(0)
    return e + d / 1000


def liters(df, p):
    q = num(df[f"{p}_B3_ENT"], df[f"{p}_B3_DEC"])
    eq = df[f"{p}_EQUIV"]
    L = np.where(eq == 3, q, np.where(eq == 4, q / 1000, np.nan))
    return pd.Series(L, index=df.index)


# liters per unit (P400_B2_COD / P500_B2_COD) when no volume equivalence is recorded
UNIT_L = {"024": 0.62, "026": 0.62, "025": 0.33, "027": 1.10, "032": 7.44, "003": 1.0, "004": 0.001,
          "061": 0.355, "062": 0.31, "063": 0.473, "166": 0.25, "206": 1.5, "170": 0.62}


def unit_liters(df, p):
    q = num(df[f"{p}_B1_ENT"], df[f"{p}_B1_DEC"])
    u = df[f"{p}_B2_COD"].astype(str).str.strip().map(UNIT_L)
    return q * u


def main():
    c4 = ["P400_A", "P400_A_COD", "P400_EQUIV", "P400_B1_ENT", "P400_B1_DEC", "P400_B2_COD", "P400_B3_ENT", "P400_B3_DEC", "P400_G_ENT", "P400_G_DEC", "UBIGEO", "AREA"] + K
    h, _ = pyreadstat.read_sav(str(R / "ENAPREF_CAP_401_404.sav"), usecols=c4)
    # keep only codes where beer is the majority of descriptions (drops catch-all "other" codes)
    isb = h.P400_A.str.upper().str.contains("CERVEZA", na=False)
    share = isb.groupby(h.P400_A_COD).mean()
    cnt = isb.groupby(h.P400_A_COD).sum()
    codes = set(share[(share > 0.5) & (cnt >= 20)].index)
    print("beer codes:", sorted(codes), share[sorted(codes)].round(2).to_dict())
    area = h.assign(**{c: h[c].astype(str).str.strip() for c in K}).groupby(K).AREA.first().reset_index()
    hb = h[h.P400_A_COD.isin(codes)].copy()
    hb["Ldecl"] = liters(hb, "P400")
    hb["Lunit"] = unit_liters(hb, "P400")
    hb["S"] = num(hb.P400_G_ENT, hb.P400_G_DEC)
    c5 = ["P500_A_COD", "P500_EQUIV", "P500_B1_ENT", "P500_B1_DEC", "P500_B2_COD", "P500_B3_ENT", "P500_B3_DEC",
          "P500_G_ENT", "P500_G_DEC", "UBIGEO", "DIA_ID"] + K
    p, _ = pyreadstat.read_sav(str(R / "ENAPREF_CAP500.sav"), usecols=c5)
    pb = p[p.P500_A_COD.isin(codes)].copy()
    pb["Ldecl"] = liters(pb, "P500")
    pb["Lunit"] = unit_liters(pb, "P500")
    pb["S"] = num(pb.P500_G_ENT, pb.P500_G_DEC)
    hb = hb.rename(columns={"P400_A_COD": "code"}); pb = pb.rename(columns={"P500_A_COD": "code"})
    allb = pd.concat([hb.assign(diary="household"), pb.assign(diary="personal")], ignore_index=True)
    for c in K:
        allb[c] = allb[c].astype(str).str.strip()
    allb["served"] = allb.code.astype(str).str.startswith("11")
    allb["dept"] = allb.UBIGEO.astype(str).str.zfill(6).str[:2].astype(int)
    okd = allb.Ldecl.between(0.1, 200)
    oku = ~okd & allb.Lunit.between(0.1, 200)
    allb["method"] = np.where(okd, "A_declared", np.where(oku, "B_unit", "C_price"))
    allb["LA"] = allb.Ldecl.where(okd, 0)
    allb["LB"] = allb.LA + allb.Lunit.where(oku, 0)
    known = (okd | oku) & (allb.S > 0)
    allb["Lk"] = np.where(okd, allb.Ldecl, allb.Lunit)
    pnat = (allb[known].S / allb[known].Lk).groupby(allb[known].served).median()
    pdep = (allb[known].S / allb[known].Lk).groupby([allb[known].dept, allb[known].served]).median()
    impC = (allb.S / allb.served.map(pnat)).where(allb.method == "C_price", 0)
    key = list(zip(allb.dept, allb.served))
    pd_ = pd.Series([pdep.get(k, np.nan) for k in key], index=allb.index).fillna(allb.served.map(pnat))
    impD = (allb.S / pd_).where(allb.method == "C_price", 0)
    allb["LC"] = allb.LB + impC.fillna(0)
    allb["LD"] = allb.LB + impD.fillna(0)
    print("records: household", len(hb), "personal", len(pb))
    print("share of records by method:", allb.method.value_counts(normalize=True).round(3).to_dict())
    print("share of liters (C) by method:", (allb.groupby("method").LC.sum() / allb.LC.sum()).round(3).to_dict())
    print("price per liter national (home, served):", pnat.round(2).to_dict(), " Puno:", pdep.get(21, pd.Series()).round(2).to_dict())
    hh = allb.groupby(K).agg(LA=("LA", "sum"), LB=("LB", "sum"), LC=("LC", "sum"), LD=("LD", "sum"),
                             served=("LC", lambda v: v[allb.loc[v.index, "served"]].sum()),
                             personal=("LC", lambda v: v[allb.loc[v.index, "diary"] == "personal"].sum())).reset_index()
    s, _ = pyreadstat.read_sav(str(R / "ENAPREF_SUMARIA.sav"), usecols=K + ["UBIGEO", "MIE", "FACTORPOBLACION"])
    for c in K:
        s[c] = s[c].astype(str).str.strip()
    d = s.merge(hh, on=K, how="left").merge(area, on=K, how="left")
    d[["LA", "LB", "LC", "LD", "served", "personal"]] = d[["LA", "LB", "LC", "LD", "served", "personal"]].fillna(0)
    d["dept"] = d.UBIGEO.astype(str).str.zfill(6).str[:2].astype(int)
    d["w"] = d.FACTORPOBLACION
    print("AREA codes:", d.AREA.value_counts().to_dict())

    def pc(x, col):
        return 52 * (x[col] * x.w).sum() / (x.MIE * x.w).sum()
    rows = []
    groups = [("Peru", d), ("Peru sin Puno", d[d.dept != 21]), ("Peru urban", d[d.AREA == 1]), ("Peru rural", d[d.AREA == 2])]
    groups += [(f"dept {k:02d}", x) for k, x in d.groupby("dept")]
    groups += [("Puno urban", d[(d.dept == 21) & (d.AREA == 1)]), ("Puno rural", d[(d.dept == 21) & (d.AREA == 2)])]
    for name, x in groups:
        rows.append(dict(area=name, A_declared=pc(x, "LA"), B_plus_units=pc(x, "LB"), C_plus_price_nat=pc(x, "LC"),
                         D_plus_price_dept=pc(x, "LD"), served=pc(x, "served"), personal=pc(x, "personal"),
                         hh_any_beer_7d=100 * np.average(x.LC > 0, weights=x.w), n_hh=len(x)))
    t = pd.DataFrame(rows)
    deps = t.area.str.startswith("dept")
    for col in ["A_declared", "B_plus_units", "C_plus_price_nat", "D_plus_price_dept"]:
        t.loc[deps, "rank_" + col[0]] = t.loc[deps, col].rank(ascending=False, method="min")
    t.to_csv(OUT / "enapref2020_beer_liters.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(1).to_string())
    # duplicate check: same household, day, product and amount in both diaries
    hk = allb[allb.diary == "household"]
    pk = allb[allb.diary == "personal"]
    m = pk.merge(hk[K + ["code", "S"]].drop_duplicates(), on=K + ["code", "S"], how="inner")
    print(f"personal beer records with an identical household-diary record (same hh, product, amount): "
          f"{len(m)} of {len(pk)} ({100*len(m)/max(len(pk),1):.1f}%)")


if __name__ == "__main__":
    main()
