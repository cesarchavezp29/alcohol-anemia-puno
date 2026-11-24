"""
analysis_alcohol_consequences.py - counterfactuals for child anemia, and other outcomes
associated with alcohol CONSUMPTION (ENDES) and alcohol SPENDING (ENAHO). Descriptive.

Part 1. Anemia counterfactuals (children 6-35 months, ENDES 2014-2024, WHO 2001 and 2024):
  1a "Behavioral": joint LPM of anemia on the four drinking margins of the selected adult
     (drank_30d, occasions, heaviest-occasion liters, frequent+harm) x Puno, with v2 controls
     and dept + year FE. Counterfactual: all margins set to zero, i.e. change = -(b'x_bar),
     evaluated for Puno (Puno-specific and national coefficients) and Peru. Delta-method SE.
  1b "Budget": households stop spending on alcohol and the money becomes ordinary household
     spending. Alcohol share of total spending from ENAPREF 2019-20 (diaries; Puno and Peru).
     Anemia-spending gradient: ENDES anemia by wealth quintile vs ENAHO median per-capita
     spending of the same national quintile (2014-2024), slope of anemia on log spending.
     Change = slope x log(1 + share). The wealth gradient bundles everything that comes with
     being richer, so this is an upper bound for a pure income effect.
Part 2. Consumption -> domestic violence (ENDES women 15-49 in the DV module, 2010-2024 no
  2020): partner drinks (D113) and gets drunk often/sometimes (D114) vs physical or sexual
  violence in the last 12 months (and emotional). Controls: age, education, wealth quintile,
  area, dept and year FE. Weight wt_dv. Puno interaction.
Part 3. Spending -> children's schooling (ENAHO annual 2010-2024 no 2020): children 6-16,
  enrolled this year (p306) and attending (p307), on household alcohol purchase and alcohol
  budget share; controls log pc spending, log hh size, age and sex FE; dept x year FE absorbed.
Part 4. Spending -> future household welfare (ENAHO panels 2011-15, 2015-19, 2019-23 linked
  to the annual food module through wave-specific ids): change in log per-capita spending
  and poverty in t+1 and t+2 on alcohol purchase in t, given t-level spending, poverty, size.
"""
from __future__ import annotations
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import endes_units as eu  # noqa: E402
from analysis_alcohol_extended import rd_sav, rd_dta  # noqa: E402
import analysis_alcohol_v2 as V2  # noqa: E402
from analysis_alcohol_engel import demean, wls_cluster  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "analysis" / "alcohol_anemia"
K = ["conglome", "vivienda", "hogar"]
pd.set_option("display.width", 230)


# ------------------------------------------------------------------ part 1
def part1():
    g = V2.build(pd.read_parquet(OUT / "_cache_v2_endes.parquet"))
    a = g[g.year >= 2014].dropna(subset=["hc27", "hv270", "alt_bin", "meduc", "qssexo", "qs23",
                                         "drank_30d", "occasions", "heavy_L", "freq_problem"])
    X = ["drank_30d", "occasions", "heavy_L", "freq_problem"]
    rows = []
    for outcome in ["anemia_2001", "anemia_2024"]:
        d = a.dropna(subset=[outcome])
        f = f"{outcome} ~ " + " + ".join(X + [f"{x}:puno" for x in X]) + f" + {V2.CTRL} + C(year) + C(dept)"
        m = smf.wls(f, data=d, weights=d.wt).fit(cov_type="cluster", cov_kwds={"groups": d.cluster})
        V = m.cov_params()
        for area, sub in [("Puno", d[d.puno == 1]), ("Peru", d)]:
            xbar = np.array([np.average(sub[x], weights=sub.wt) for x in X])
            for coefs in (["Puno-specific", "national (rest of Peru)"] if area == "Puno" else ["national (rest of Peru)"]):
                names = X + ([f"{x}:puno" for x in X] if coefs == "Puno-specific" else [])
                gvec = pd.Series(0.0, index=m.params.index)
                for i, x in enumerate(X):
                    gvec[x] = -xbar[i]
                    if coefs == "Puno-specific":
                        gvec[f"{x}:puno"] = -xbar[i]
                eff = float(gvec @ m.params)
                se = float(np.sqrt(gvec @ V @ gvec))
                base = 100 * np.average(sub[outcome], weights=sub.wt)
                rows.append(dict(part="1a behavioral", outcome=outcome, area=area, coefficients=coefs,
                                 anemia_now=base, change_pp=100 * eff, ci_low=100 * (eff - 1.96 * se),
                                 ci_high=100 * (eff + 1.96 * se)))
    # 1b budget
    share = enapref_alcohol_share()
    grad = wealth_gradient(g)
    for outcome, slope in grad.items():
        for area, sh in share.items():
            rows.append(dict(part="1b budget (upper bound)", outcome=outcome, area=area,
                             coefficients=f"slope {slope:.3f} per log-point; alcohol share {100*sh:.2f}%",
                             anemia_now=np.nan, change_pp=100 * slope * np.log(1 + sh), ci_low=np.nan, ci_high=np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cons_part1_counterfactual.csv", index=False)
    print(t.round(3).to_string())


def enapref_alcohol_share():
    R = RAW / "enapref" / "2020"
    K2 = ["CONGLOMERADO", "NSELV", "HOGAR_ID"]
    h, _ = pyreadstat.read_sav(str(R / "ENAPREF_CAP_401_404.sav"), usecols=K2 + ["P400_A", "P400_A_COD", "P400_G_ENT", "P400_G_DEC"])
    nm = h.P400_A.str.upper().fillna("")
    isalc = nm.str.contains(r"CERVEZA|PISCO|\bRON\b|WHISK|VINO\b|CHICHA DE JORA|CA[ÑN]AZO|AGUARDIENTE|LICOR|VODKA|CHAMP|ANISADO|YONQUE|MACERADO|SANGRIA", regex=True)
    share = isalc.groupby(h.P400_A_COD).mean()
    cnt = isalc.groupby(h.P400_A_COD).sum()
    codes = set(share[(share > 0.5) & (cnt >= 10)].index)
    def spend(df, p):
        e = pd.to_numeric(df[f"{p}_G_ENT"], errors="coerce").fillna(0)
        dd = pd.to_numeric(df[f"{p}_G_DEC"].astype(str).str.strip().replace({"": "0", "nan": "0"}), errors="coerce").fillna(0)
        return e + dd / 1000
    h = h[h.P400_A_COD.isin(codes)].copy(); h["S"] = spend(h, "P400")
    p, _ = pyreadstat.read_sav(str(R / "ENAPREF_CAP500.sav"), usecols=K2 + ["P500_A_COD", "P500_G_ENT", "P500_G_DEC"])
    p = p[p.P500_A_COD.isin(codes)].copy(); p["S"] = spend(p, "P500")
    a = pd.concat([h[K2 + ["S"]], p[K2 + ["S"]]])
    for c in K2:
        a[c] = a[c].astype(str).str.strip()
    a = a.groupby(K2).S.sum().mul(52).rename("alc").reset_index()
    s, _ = pyreadstat.read_sav(str(R / "ENAPREF_SUMARIA.sav"), usecols=K2 + ["UBIGEO", "GAS_TOT", "FACTORPOBLACION"])
    for c in K2:
        s[c] = s[c].astype(str).str.strip()
    s = s.merge(a, on=K2, how="left").fillna({"alc": 0})
    s["dept"] = s.UBIGEO.astype(str).str.zfill(6).str[:2].astype(int)
    out = {}
    for area, x in [("Peru", s), ("Puno", s[s.dept == 21])]:
        out[area] = (x.alc * x.FACTORPOBLACION).sum() / (x.GAS_TOT * x.FACTORPOBLACION).sum()
    print(f"  ENAPREF alcohol codes: {len(codes)}; alcohol share of total spending: "
          f"Peru {100*out['Peru']:.2f}%, Puno {100*out['Puno']:.2f}%  (GAS_TOT assumed annual)")
    return out


def wealth_gradient(g):
    d = g[g.year >= 2014].dropna(subset=["hv270"])
    hh = pd.read_parquet(OUT / "_cache_enaho_households.parquet")
    hh = hh[hh.year >= 2014].copy()
    hh["pc"] = hh.gashog2d / hh.mieperho
    med = []
    for y, s in hh.groupby("year"):
        q = pd.qcut(s.pc.rank(method="first"), 5, labels=False)    # unweighted approx. of national quintiles
        for k in range(5):
            x = s[q == k]
            med.append(dict(year=y, q=k + 1, lpc=np.log(np.average(x.pc, weights=x.pw))))
    med = pd.DataFrame(med)
    out = {}
    for outcome in ["anemia_2001", "anemia_2024"]:
        an = d.dropna(subset=[outcome]).groupby(["year", "hv270"]).apply(
            lambda s: np.average(s[outcome], weights=s.wt)).rename("an").reset_index().rename(columns={"hv270": "q"})
        m = an.merge(med, on=["year", "q"])
        fit = smf.ols("an ~ lpc + C(year)", data=m).fit()
        out[outcome] = fit.params["lpc"]
        print(f"  anemia-spending gradient {outcome}: {fit.params['lpc']:.3f} per log-point (se {fit.bse['lpc']:.3f})")
    return out


# ------------------------------------------------------------------ part 2
def part2():
    fr = []
    for y in [y for y in range(2010, 2025) if y != 2020]:
        d = eu.dir_for(y)
        from analysis_alcohol_extended import find_sav
        p, cl = find_sav(d, ["caseid", "d113"])
        dv, _ = pyreadstat.read_sav(str(p), usecols=[cl[c] for c in ["caseid", "d113", "d114"] if c in cl])
        dv.columns = [c.lower() for c in dv.columns]
        q, cq = find_sav(d, ["caseid", "v190", "v106"])
        w, _ = pyreadstat.read_sav(str(q), usecols=[cq[c] for c in ["caseid", "v190", "v106", "v012", "v025", "v024"] if c in cq])
        w.columns = [c.lower() for c in w.columns]
        x = dv.merge(w, on="caseid", how="left")
        x["caseid"] = x.caseid.astype(str).str.strip()
        x["survey_year"] = y
        fr.append(x)
    dv = pd.concat(fr, ignore_index=True)
    wf = pd.read_csv(os.environ.get("ENDES_WOMEN_CSV", str(ROOT / "raw" / "endes_women_2007_2025.csv")),
                     usecols=["survey_year", "caseid", "wt_dv", "dv_sample", "physsex_12m", "emo_12m", "psu"], low_memory=False)
    wf["caseid"] = wf.caseid.astype(str).str.strip()
    d = dv.merge(wf, on=["survey_year", "caseid"], how="inner")
    print(f"  DV merge: {len(d):,} women matched of {len(dv):,} in REC84DV")
    assert not d.duplicated(["survey_year", "caseid"]).any()
    d = d[(d.dv_sample == 1) & d.d113.isin([0, 1]) & d.wt_dv.notna()].copy()
    d["partner_drinks"] = d.d113.astype(float)
    ok114 = (d.d113 == 0) | d.d114.isin([0, 1, 2])
    d["drunk_often"] = (d.d114 == 1).astype(float).where(ok114)
    d["drunk_any"] = d.d114.isin([1, 2]).astype(float).where(ok114)
    d["cat"] = np.select([d.d113 == 0, d.d114 == 0, d.d114 == 2, d.d114 == 1],
                         ["0 no bebe", "1 bebe, nunca se embriaga", "2 se embriaga a veces", "3 se embriaga con frecuencia"], "")
    d.loc[~ok114, "cat"] = ""
    d["period"] = np.where(d.survey_year <= 2022, "2010-2022", "2023-2024")
    d["puno"] = (d.v024 == 21).astype(float)
    d["cl"] = d.survey_year.astype(str) + "_" + d.psu.astype(str)
    rows = []
    cats = []
    for outcome in ["physsex_12m", "emo_12m"]:
        for per in ["2010-2022", "2023-2024"]:
            s = d[(d.period == per) & (d.cat != "")].dropna(subset=[outcome, "v190", "v106", "v012"])
            s = s.copy()
            levels = sorted(s.cat.unique())[1:]
            for j, c in enumerate(levels, 1):
                s[f"k{j}"] = (s.cat == c).astype(float)
                s[f"k{j}p"] = s[f"k{j}"] * s.puno
            terms = " + ".join([f"k{j} + k{j}p" for j in range(1, len(levels) + 1)])
            f = f"{outcome} ~ {terms} + C(v190) + C(v106) + v012 + I(v012**2) + C(v025) + C(v024) + C(survey_year)"
            m = smf.wls(f, data=s, weights=s.wt_dv).fit(cov_type="cluster", cov_kwds={"groups": s.cl})
            Vc = m.cov_params()
            for j, c in enumerate(levels, 1):
                k = f"k{j}"
                ki = f"k{j}p"
                bp = m.params[k] + m.params[ki]
                sep = np.sqrt(Vc.loc[k, k] + Vc.loc[ki, ki] + 2 * Vc.loc[k, ki])
                cats.append(dict(outcome=outcome, period=per, category=c, rest_pp=100 * m.params[k], rest_se=100 * m.bse[k],
                                 puno_pp=100 * bp, puno_se=100 * sep, diff_p=m.pvalues[ki], n=int(m.nobs)))
        s = d[d.period == "2010-2022"].dropna(subset=[outcome, "v190", "v106", "v012"])
        for x in ["partner_drinks", "drunk_any", "drunk_often"]:
            s = s.dropna(subset=[x])
            f = f"{outcome} ~ {x} + {x}:puno + C(v190) + C(v106) + v012 + I(v012**2) + C(v025) + C(v024) + C(survey_year)"
            m = smf.wls(f, data=s, weights=s.wt_dv).fit(cov_type="cluster", cov_kwds={"groups": s.cl})
            Vc = m.cov_params()
            bp = m.params[x] + m.params[f"{x}:puno"]
            sep = np.sqrt(Vc.loc[x, x] + Vc.loc[f"{x}:puno", f"{x}:puno"] + 2 * Vc.loc[x, f"{x}:puno"])
            rows.append(dict(outcome=outcome, regressor=x, coef_pp=100 * m.params[x], se=100 * m.bse[x], p=m.pvalues[x],
                             puno_pp=100 * bp, puno_se=100 * sep, mean_outcome=100 * np.average(s[outcome], weights=s.wt_dv),
                             share_x=100 * np.average(s[x], weights=s.wt_dv),
                             share_x_puno=100 * np.average(s[s.puno == 1][x], weights=s[s.puno == 1].wt_dv), n=int(m.nobs)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cons_part2_violence.csv", index=False)
    pd.DataFrame(cats).to_csv(OUT / "cons_part2_violence_categories.csv", index=False)
    print(pd.DataFrame(cats).round(2).to_string())
    desc = []
    dd = d[(d.period == "2010-2022") & (d.cat != "")]
    for area, s in [("Puno", dd[dd.puno == 1]), ("Resto del Peru", dd[dd.puno == 0])]:
        for cat, x in s.groupby("cat"):
            ph = x.physsex_12m.dropna()
            wph = x.loc[ph.index, "wt_dv"]
            p_ = np.average(ph, weights=wph)
            neff = wph.sum() ** 2 / (wph ** 2).sum()
            desc.append(dict(area=area, categoria=cat, share=100 * x.wt_dv.sum() / s.wt_dv.sum(),
                             physsex_se=100 * np.sqrt(p_ * (1 - p_) / neff),
                             physsex=100 * np.average(x.physsex_12m.fillna(0), weights=x.wt_dv * x.physsex_12m.notna()),
                             emo=100 * np.average(x.emo_12m.fillna(0), weights=x.wt_dv * x.emo_12m.notna()), n=len(x)))
    pd.DataFrame(desc).to_csv(OUT / "cons_part2_violence_desc.csv", index=False)
    print(pd.DataFrame(desc).round(1).to_string())
    print(t.round(3).to_string())


# ------------------------------------------------------------------ part 3
def part3():
    hh = pd.read_parquet(OUT / "_cache_enaho_households.parquet")
    hh["any_alc"] = ((hh.beer + hh.oalc) > 0).astype(float)
    hh["alc_share"] = 100 * (hh.beer + hh.oalc) / hh.gashog2d
    hh["lpc"] = np.log(hh.gashog2d / hh.mieperho)
    hh["lsize"] = np.log(hh.mieperho)
    fr = []
    for y in sorted(hh.year.unique()):
        e = rd_dta(RAW / "educacion" / f"enaho-{y}-03.dta", K + ["p208a", "p207", "p306", "p307", "mes"])
        for c in K:
            e[c] = e[c].astype(str).str.strip()
        e = e[pd.to_numeric(e.p208a, errors="coerce").between(6, 16)]
        e["year"] = y
        fr.append(e)
    e = pd.concat(fr, ignore_index=True)
    for c in K:
        hh[c] = hh[c].astype(str).str.strip()
    d = e.merge(hh[K + ["year", "any_alc", "alc_share", "lpc", "lsize", "factor07", "dept"]], on=K + ["year"], how="inner")
    d = d[d.lpc.notna() & np.isfinite(d.lpc)]
    d["enrolled"] = (d.p306 == 1).astype(float).where(d.p306.isin([1, 2]))
    d["attends_if_enrolled"] = (d.p307 == 1).astype(float).where(d.p307.isin([1, 2]))
    d["attends"] = np.where(d.p306 == 2, 0.0, d.attends_if_enrolled)
    d["school_months"] = pd.to_numeric(d.mes, errors="coerce").between(4, 11)
    d["dy"] = d.dept.astype(str) + "_" + d.year.astype(str) + "_" + pd.to_numeric(d.mes, errors="coerce").astype("Int64").astype(str)
    d["cl"] = d.conglome + "_" + d.year.astype(str)
    for a in range(6, 17):
        d[f"age{a}"] = (d.p208a == a).astype(float)
    d["female"] = (d.p207 == 2).astype(float)
    rows = []
    for outcome in ["enrolled", "attends", "attends_if_enrolled"]:
        for x in ["any_alc", "alc_share"]:
            for lab, s in [("all", d), ("poor half (lpc below median)", d[d.lpc < d.lpc.median()]),
                           ("school months Apr-Nov", d[d.school_months])]:
                s = s.dropna(subset=[outcome]).reset_index(drop=True)
                xs = [x, "lpc", "lsize", "female"] + [f"age{a}" for a in range(7, 17)]
                z = demean(s, [outcome] + xs, "dy", "factor07")
                b, se = wls_cluster(z[outcome].values, z[xs].values, s.factor07.values, s.cl.values)
                rows.append(dict(outcome=outcome, regressor=x, sample=lab, coef_pp=100 * b[0], se=100 * se[0],
                                 t=b[0] / se[0], mean=100 * np.average(s[outcome], weights=s.factor07), n=len(s)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cons_part3_schooling.csv", index=False)
    print(t.round(3).to_string())


# ------------------------------------------------------------------ part 4
REL = {"2015_529": range(2011, 2016), "2019_699": range(2015, 2020), "2023_912": range(2019, 2024)}


def part4():
    hh = pd.read_parquet(OUT / "_cache_enaho_households.parquet")
    for c in K:
        hh[c] = hh[c].astype(str).str.strip()
    hh["any_alc"] = ((hh.beer + hh.oalc) > 0).astype(float)
    hh["alc_share"] = 100 * (hh.beer + hh.oalc) / hh.gashog2d
    hh["alc_soles"] = hh.beer + hh.oalc
    alc = hh[K + ["year", "any_alc", "alc_share", "alc_soles"]]
    pairs = []
    for rel, yrs in REL.items():
        f = next((RAW / "panel" / rel).glob("sumaria*panel.dta"), None) or next((RAW / "panel" / rel).glob("sumaria*.dta"))
        _, m = pyreadstat.read_dta(str(f), metadataonly=True)
        cols = set(c.lower() for c in m.column_names)
        need = []
        for y in yrs:
            yy = f"{y % 100:02d}"
            need += [f"{v}_{yy}" for v in ["conglome", "vivienda", "hogar", "gashog2d", "mieperho", "pobreza", "factor07", "ubigeo", "mes"]
                     if f"{v}_{yy}" in cols]
        p = rd_dta(f, ["numpanh"] + need)
        for t0 in yrs:
            for h in (1, 2):
                t1 = t0 + h
                if t1 not in yrs or t0 == 2020 or t1 == 2020:
                    continue
                a, b = f"{t0 % 100:02d}", f"{t1 % 100:02d}"
                req = [f"conglome_{a}", f"gashog2d_{a}", f"gashog2d_{b}", f"mieperho_{a}", f"mieperho_{b}"]
                if not all(c in p for c in req):
                    continue
                x = p.copy()
                for c in x.columns:
                    if c.startswith(("gashog2d_", "mieperho_", "pobreza_", "factor07_")):
                        x[c] = pd.to_numeric(x[c], errors="coerce")
                x = x.dropna(subset=req)
                x = x[(x[f"gashog2d_{a}"] > 0) & (x[f"gashog2d_{b}"] > 0)]
                z = pd.DataFrame({"conglome": x[f"conglome_{a}"].astype(str).str.strip(),
                                  "vivienda": x[f"vivienda_{a}"].astype(str).str.strip(),
                                  "hogar": x[f"hogar_{a}"].astype(str).str.strip(),
                                  "year": t0, "h": h, "rel": rel,
                                  "lpc0": np.log(x[f"gashog2d_{a}"] / x[f"mieperho_{a}"]),
                                  "lpc1": np.log(x[f"gashog2d_{b}"] / x[f"mieperho_{b}"]),
                                  "poor0": x[f"pobreza_{a}"].isin([1, 2]).astype(float).where(x[f"pobreza_{a}"].notna()),
                                  "poor1": x[f"pobreza_{b}"].isin([1, 2]).astype(float).where(x[f"pobreza_{b}"].notna()),
                                  "gas0": x[f"gashog2d_{a}"], "gas1": x[f"gashog2d_{b}"],
                                  "mie0": x[f"mieperho_{a}"], "mie1": x[f"mieperho_{b}"],
                                  "c1": x[f"conglome_{b}"].astype(str).str.strip(),
                                  "v1": x[f"vivienda_{b}"].astype(str).str.strip(),
                                  "h1": x[f"hogar_{b}"].astype(str).str.strip(),
                                  "lsize": np.log(x[f"mieperho_{a}"]),
                                  "w": x[f"factor07_{a}"],
                                  "dept": x[f"ubigeo_{a}"].astype(str).str.zfill(6).str[:2],
                                  "mes0": x[f"mes_{a}"].astype(str) if f"mes_{a}" in x else "na"})
                pairs.append(z)
        print(f"  panel {rel}: pairs so far {sum(len(z) for z in pairs):,}", flush=True)
    d = pd.concat(pairs, ignore_index=True)
    # zero-pad ids like the annual files
    d = d.merge(alc, on=K + ["year"], how="left")
    print(f"  linked to annual food module: {d.any_alc.notna().mean():.1%} of panel pairs")
    d = d.dropna(subset=["any_alc"]).reset_index(drop=True)
    a1 = alc[K + ["year", "alc_soles"]].rename(columns={"conglome": "c1", "vivienda": "v1", "hogar": "h1",
                                                         "alc_soles": "alc1", "year": "t1"})
    d["t1"] = d.year + d.h
    d = d.merge(a1, on=["c1", "v1", "h1", "t1"], how="left")
    print(f"  alcohol also observed in t+h for {d.alc1.notna().mean():.1%} of linked pairs")
    d["dlpc"] = d.lpc1 - d.lpc0
    d["lnc0"] = np.log((d.gas0 - d.alc_soles).clip(lower=1) / d.mie0)
    d["lnc1"] = np.log((d.gas1 - d.alc1).clip(lower=1) / d.mie1)
    d["dlnc"] = d.lnc1 - d.lnc0
    d["dy"] = d.dept + "_" + d.year.astype(str) + "_" + d.h.astype(str) + "_" + d.mes0.astype(str)
    d["cl"] = d.rel + "_" + d.conglome
    rows = []
    for h in (1, 2):
        for outcome, base in [("dlpc", "lpc0"), ("dlnc", "lnc0"), ("poor1", "lpc0")]:
            for x in ["any_alc", "alc_share"]:
                s = d[d.h == h].dropna(subset=[outcome, base, "poor0"]).reset_index(drop=True)
                xs = [x, base, "poor0", "lsize"]
                z = demean(s, [outcome] + xs, "dy", "w")
                b, se = wls_cluster(z[outcome].values, z[xs].values, s.w.values, s.cl.values)
                rows.append(dict(horizon=f"t+{h}", outcome=outcome, regressor=x, coef=b[0], se=se[0], t=b[0] / se[0],
                                 n=len(s), share_alc=np.average(s.any_alc, weights=s.w)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cons_part4_future_welfare.csv", index=False)
    print(t.round(4).to_string())


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for k in which:
        print(f"===== part {k}", flush=True)
        {"1": part1, "2": part2, "3": part3, "4": part4}[k]()
