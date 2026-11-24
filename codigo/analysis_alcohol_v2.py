"""
analysis_alcohol_v2.py - second round after ChatGPT's audit (2026-11-24).

Fixes relative to analysis_alcohol_extended.py:
 1. Anemia rebuilt from raw hemoglobin (hc53, g/dL x10) with ONE criterion for all years:
      anemia_2001: Hb - CDC altitude adjustment < 11.0 g/dL. Adjustment (g/dL)
                   = -0.032*A + 0.022*A^2, A = altitude_m*3.3/1000, applied only where
                   positive (from about 1000 m), as INEI/DHS.
      anemia_2024: Hb - WHO 2024 adjustment < 10.5 (6-23 months) or 11.0 (24-35 months).
                   Adjustment (g/L) = 0.0056384*alt + 0.0000003*alt^2, from 1 m.
    Both validated against Campos-Sanchez et al. 2025 (RPMESP 42(2), Tables 1-2).
 2. Alcohol skip patterns respected (audit in the log):
      drank_30d    QS210 == 1, among adults who answered QS208
      occasions    QS211C (number of drinking occasions in 30 days), 0 if no drinking in 30 days
      heavy_L      liters drunk on the HEAVIEST occasion of the month (QS212: "piense en la vez
                   que consumio mas"), all drinks: glasses*0.25 L + bottles*0.62 L summed over
                   beverage types; 0 if no drinking in 30 days; missing if quantities missing
      beer_heavy   beer drunk on the heaviest occasion (QS212A, or beer glasses/bottles > 0 when
                   QS212A is absent, as in 2015)
      freq_problem drinks 12+ days a year (QS209) AND reports >= 1 harm item QS713-QS717
                   (asked only to frequent drinkers); missing if frequent and all items missing
 3. Alcohol x Puno interactions; weights normalized to mean 1 within each year; merges checked.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import patsy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import endes_units as eu  # noqa: E402
from analysis_alcohol_extended import rd_sav  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "analysis" / "alcohol_anemia"
YEARS = [y for y in range(2010, 2025) if y != 2020]
HARM = ["qs713", "qs714", "qs716", "qs717"]
BEV = ["a", "b", "c", "d", "e", "f", "g", "x"]


def one_per_hh(df, name):
    dup = df.duplicated("hhid", keep=False)
    if dup.any():
        d = df[dup]
        same = d.groupby("hhid").nunique(dropna=False).max(axis=1).max() <= 1
        print(f"    {name}: {dup.sum()} duplicated hhid rows, identical={same}")
        assert same, f"{name}: conflicting duplicates"
    return df.drop_duplicates("hhid")


def load():
    fr = []
    for y in YEARS:
        d = eu.dir_for(y)
        c = rd_sav(d, ["hhid", "hc1", "hc53"], ["hc27", "hc61", "hc57"])
        h = one_per_hh(rd_sav(d, ["hhid", "hv005", "hv024", "hv025", "hv001", "hv040"], ["hv008"]), "RECH0")
        w = one_per_hh(rd_sav(d, ["hhid", "hv270"]), "RECH23")
        qv = ["qs209", "qs210", "qs211c", "qs212a", "qssexo", "qs23"] + HARM + \
             [f"qs212{b}v" for b in BEV] + [f"qs212{b}b" for b in BEV]
        s = rd_sav(d, ["hhid", "qs208"], qv)
        g = c.merge(h, on="hhid", how="left", validate="many_to_one").merge(w, on="hhid", how="left", validate="many_to_one")
        if s is not None:
            s = s.drop_duplicates("hhid")
            g = g.merge(s, on="hhid", how="left", validate="many_to_one")
        g = g[eu.true_year_mask(g, y, "hv008")]
        g["year"] = y
        fr.append(g)
        print(f"  {y}: children {len(g):,}", flush=True)
    return pd.concat(fr, ignore_index=True)


def build(g):
    g = g[g.hc1.between(6, 35) & g.hc53.between(30, 250)].copy()
    hb = g.hc53 / 10
    alt = g.hv040.where(g.hv040.between(0, 5500))
    A = alt * 3.3 / 1000
    adj01 = (-0.032 * A + 0.022 * A ** 2).clip(lower=0)
    adj24 = (0.0056384 * alt + 0.0000003 * alt ** 2) / 10
    g["hb_raw"] = hb
    g["hb_adj01"] = hb - adj01
    g["hb_adj24"] = hb - adj24
    cut24 = np.where(g.hc1 < 24, 10.5, 11.0)
    g["anemia_2001"] = (g.hb_adj01 < 11.0).astype(float).where(g.hb_adj01.notna())
    g["anemia_2024"] = (g.hb_adj24 < cut24).astype(float).where(g.hb_adj24.notna())
    # factorial: B = old altitude adjustment + new age thresholds; C = new adjustment + old 11.0 threshold
    g["anemia_B"] = (g.hb_adj01 < cut24).astype(float).where(g.hb_adj01.notna())
    g["anemia_C"] = (g.hb_adj24 < 11.0).astype(float).where(g.hb_adj24.notna())
    g["anemia_inei"] = g.hc57.isin([1, 2, 3]).astype(float).where(g.hc57.isin([1, 2, 3, 4]))
    g["wt"] = g.hv005 / 1e6
    g["wt"] = g.wt / g.groupby("year").wt.transform("mean")
    g["dept"] = g.hv024.astype(int)
    g["puno"] = (g.dept == 21).astype(float)
    g["alt_bin"] = pd.cut(g.hv040, [-1, 500, 1500, 2500, 3500, 6000], labels=False)
    g["meduc"] = g.hc61.where(g.hc61 < 8)
    g["cluster"] = g.year.astype(str) + "_" + g.hv001.astype(int).astype(str)
    # alcohol
    ok = g.qs208.isin([1, 2])
    no12 = g.qs208 == 2
    d30 = g.qs210 == 1
    d30v = pd.Series(np.nan, index=g.index)
    d30v[no12] = 0.0                                   # structural zero: no drinking in 12 months
    d30v[(g.qs208 == 1) & (g.qs210 == 2)] = 0.0
    d30v[(g.qs208 == 1) & d30] = 1.0                   # qs210 missing/8 among 12m drinkers stays NaN
    g["drank_30d"] = d30v
    occ = g.qs211c.where(g.qs211c.between(1, 97))
    g["occasions"] = np.where(d30v == 0, 0.0, np.where(d30v == 1, occ, np.nan))
    qcols = [c for c in g.columns if c.startswith("qs212") and c[-1] in "vb" and len(c) == 7]
    valid = {c: g[c].where(g[c].between(0, 97)) for c in qcols}       # 98/99 = don't know -> missing
    special = pd.concat([g[c].ge(98) for c in qcols], axis=1).any(axis=1)
    anyv = pd.concat([valid[c].notna() for c in qcols], axis=1).any(axis=1)
    gl = sum(valid[c].fillna(0) for c in qcols if c.endswith("v"))
    bo = sum(valid[c].fillna(0) for c in qcols if c.endswith("b"))
    hl = (gl * 0.25 + bo * 0.62).where(anyv & ~special)
    g["heavy_L"] = np.where(d30v == 0, 0.0, np.where(d30v == 1, hl, np.nan))
    bq = (valid.get("qs212av", pd.Series(0, index=g.index)).fillna(0) > 0) | (valid.get("qs212ab", pd.Series(0, index=g.index)).fillna(0) > 0)
    beer = pd.Series(np.where(g.qs212a.isin([1, 2]), (g.qs212a == 1).astype(float), np.where(anyv, bq.astype(float), np.nan)), index=g.index)
    g["beer_heavy"] = np.where(d30v == 0, 0.0, np.where(d30v == 1, beer, np.nan))
    freq = g.qs209 == 1
    freq_known = g.qs209.isin([1, 2]) | no12
    harm_any = g[HARM].eq(1).any(axis=1)
    harm_miss = g[HARM].isna().all(axis=1)
    fp = np.where(freq, harm_any.astype(float), 0.0)
    fp = pd.Series(fp, index=g.index).where(~(freq & harm_miss))
    g["freq_problem"] = fp.where(ok & freq_known)
    g["frequent"] = freq.astype(float).where(ok & freq_known)
    return g


def validate(g):
    rows = []
    for y in sorted(g.year.unique()):
        s = g[g.year == y]
        rows.append(dict(year=y, a2001=100 * np.average(s.anemia_2001, weights=s.wt),
                         a2024=100 * np.average(s.anemia_2024, weights=s.wt),
                         inei=100 * np.average(s.anemia_inei.fillna(0), weights=s.wt * s.anemia_inei.notna())))
    v = pd.DataFrame(rows)
    paper = {2010: (50.3, 43.4), 2011: (41.6, 35.6), 2012: (44.5, 38.6), 2013: (46.4, 39.7), 2014: (46.8, 40.1),
             2015: (43.5, 37.4), 2016: (43.6, 38.3), 2017: (43.6, 37.4), 2018: (43.5, 38.5), 2019: (40.3, 34.2),
             2021: (39.0, 30.9), 2022: (42.4, 36.5), 2023: (43.1, 37.7)}
    v["paper2001"] = v.year.map(lambda y: paper.get(y, (np.nan, np.nan))[0])
    v["paper2024"] = v.year.map(lambda y: paper.get(y, (np.nan, np.nan))[1])
    p = g[(g.puno == 1) & (g.year <= 2023)]
    print(v.round(1).to_string())
    print(f"  Puno pooled 2010-2023: 2001 {100*np.average(p.anemia_2001, weights=p.wt):.1f} (paper 2009-23: 73.8), "
          f"2024 {100*np.average(p.anemia_2024, weights=p.wt):.1f} (paper 47.2)")
    v.to_csv(OUT / "v2_anemia_validation.csv", index=False)


CTRL = "C(hc1) + C(hc27) + C(hv270) + C(hv025) + C(alt_bin) + C(meduc) + C(qssexo) + qs23"


def regs(g):
    rows = []
    a = g[g.year >= 2014].dropna(subset=["hc27", "hv270", "alt_bin", "meduc", "qssexo", "qs23"])
    for outcome in ["anemia_2001", "anemia_2024", "hb_raw"]:
        for lab, d0 in [("2014-2024", a), ("without 2024", a[a.year < 2024])]:
            for x in ["drank_30d", "occasions", "heavy_L", "beer_heavy", "freq_problem"]:
                d = d0.dropna(subset=[x, outcome])
                f = f"{outcome} ~ {x} + {x}:puno + {CTRL} + C(year) + C(dept)"
                m = smf.wls(f, data=d, weights=d.wt).fit(cov_type="cluster", cov_kwds={"groups": d.cluster})
                b, bi = m.params[x], m.params[f"{x}:puno"]
                V = m.cov_params()
                se_p = np.sqrt(V.loc[x, x] + V.loc[f"{x}:puno", f"{x}:puno"] + 2 * V.loc[x, f"{x}:puno"])
                from scipy.stats import norm
                rows.append(dict(outcome=outcome, years=lab, regressor=x,
                                 rest=b, rest_se=m.bse[x], rest_p=m.pvalues[x],
                                 puno=b + bi, puno_se=se_p, puno_p=2 * norm.sf(abs((b + bi) / se_p)),
                                 diff_p=m.pvalues[f"{x}:puno"], n=int(m.nobs), n_puno=int(d.puno.sum()),
                                 mean_x_puno=np.average(d[d.puno == 1][x], weights=d[d.puno == 1].wt),
                                 mean_x_rest=np.average(d[d.puno == 0][x], weights=d[d.puno == 0].wt)))
            print(f"  regs {outcome} {lab}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_regressions_puno.csv", index=False)
    return t


def oaxaca(g):
    out = []
    d = g[g.year >= 2014].dropna(subset=["drank_30d", "occasions", "heavy_L", "freq_problem", "hc27", "hv270",
                                         "alt_bin", "meduc", "wt"]).copy()
    groups = {"alcohol (30d, occasions, heaviest-occasion liters, frequent+problem)":
              ["drank_30d", "occasions", "heavy_L", "freq_problem"],
              "wealth quintile": ["C(hv270)"], "mother's education": ["C(meduc)"], "urban/rural": ["C(hv025)"],
              "altitude": ["C(alt_bin)"], "child age and sex": ["C(hc1)", "C(hc27)"]}
    terms = sum(groups.values(), [])
    for outcome in ["anemia_2001", "anemia_2024"]:
        y, X = patsy.dmatrices(f"{outcome} ~ " + " + ".join(terms) + " + C(year)", d, return_type="dataframe")
        w = d.loc[X.index, "wt"].values
        pn = d.loc[X.index, "puno"].values == 1
        gap = np.average(y.values.ravel()[pn], weights=w[pn]) - np.average(y.values.ravel()[~pn], weights=w[~pn])
        diff = np.average(X.values[pn], axis=0, weights=w[pn]) - np.average(X.values[~pn], axis=0, weights=w[~pn])
        for coefs in ["pooled (+Puno dummy)", "rest-of-Peru coefficients"]:
            if coefs.startswith("pooled"):
                Xp = X.assign(puno=pn.astype(float))
                beta = np.linalg.lstsq(Xp.values * np.sqrt(w)[:, None], y.values.ravel() * np.sqrt(w), rcond=None)[0][:-1]
            else:
                sel = ~pn if coefs.startswith("rest") else pn
                beta = np.linalg.lstsq(X.values[sel] * np.sqrt(w[sel])[:, None], y.values.ravel()[sel] * np.sqrt(w[sel]),
                                       rcond=None)[0]
            contrib = pd.Series(diff * beta, index=X.columns)
            tot = 0
            for gname, keys in groups.items():
                cols = [c for c in X.columns if any(c == k or c.startswith(k + "[") for k in keys)]
                v = contrib[cols].sum()
                tot += v
                out.append(dict(outcome=outcome, coefs=coefs, component=gname, pp=100 * v))
            out.append(dict(outcome=outcome, coefs=coefs, component="TOTAL explained", pp=100 * tot))
            out.append(dict(outcome=outcome, coefs=coefs, component="unexplained", pp=100 * (gap - tot)))
            out.append(dict(outcome=outcome, coefs=coefs, component="GAP Puno - rest", pp=100 * gap))
    o = pd.DataFrame(out)
    o.to_csv(OUT / "v2_oaxaca_puno.csv", index=False)
    return o


def factorial(g):
    rows = []
    for lab, d in [("2010-2024", g), ("2014-2024", g[g.year >= 2014])]:
        gaps = {}
        for k, v in {"A": "anemia_2001", "B": "anemia_B", "C": "anemia_C", "D": "anemia_2024"}.items():
            s = d.dropna(subset=[v])
            p, r = s[s.puno == 1], s[s.puno == 0]
            gaps[k] = 100 * (np.average(p[v], weights=p.wt) - np.average(r[v], weights=r.wt))
            rows.append(dict(sample=lab, definition=k, var=v, puno=100 * np.average(p[v], weights=p.wt),
                             rest=100 * np.average(r[v], weights=r.wt), gap=gaps[k]))
        phi_alt = ((gaps["C"] - gaps["A"]) + (gaps["D"] - gaps["B"])) / 2
        phi_cut = ((gaps["B"] - gaps["A"]) + (gaps["D"] - gaps["C"])) / 2
        rows.append(dict(sample=lab, definition="Shapley: altitude adjustment", gap=phi_alt))
        rows.append(dict(sample=lab, definition="Shapley: age threshold", gap=phi_cut))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "v2_factorial_shapley.csv", index=False)
    print(t.round(2).to_string())
    print(pd.crosstab(g.puno, g.alt_bin, normalize="index").round(3))


def main():
    pd.set_option("display.width", 230)
    cache = OUT / "_cache_v2_endes.parquet"
    if cache.exists():
        raw = pd.read_parquet(cache)
    else:
        raw = load()
        raw.to_parquet(cache)
    g = build(raw)
    validate(g)
    a = g[g.year >= 2013]
    aud = a.groupby("year").apply(lambda s: pd.Series({
        "adult_answered": s.qs208.isin([1, 2]).mean(), "drank_30d": np.nanmean(s.drank_30d),
        "beer_heavy": np.nanmean(s.beer_heavy), "freq_problem": np.nanmean(s.freq_problem),
        "freq_problem_missing": s.freq_problem.isna().mean()}))
    print(aud.round(3).to_string())
    aud.to_csv(OUT / "v2_alcohol_audit_by_year.csv")
    factorial(g)
    t = regs(g)
    print(t.round(4).to_string())
    o = oaxaca(g)
    print(o.pivot_table(index="component", columns=["outcome", "coefs"], values="pp").round(2).to_string())


if __name__ == "__main__":
    main()
