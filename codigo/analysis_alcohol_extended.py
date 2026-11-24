"""
analysis_alcohol_extended.py - robustness for the alcohol / child-anemia question.

ENDES children 6-35 months, 2010-2024 without 2020 (phone survey). Outcomes:
  anemia  hc57 in 1-3 (INEI headline)       hb_raw  hc53/10 g/dl     hb_adj  hc56/10 g/dl (altitude-adjusted)
Alcohol, two sources:
  ENDES adult health module (2013+): drank_30d (QS210), beer_30d (QS212A), problem (QS713..QS729)
  ENAHO module 601 (2010+), department x year, real annual spending per person:
      beer_pc (code 4404)  other_pc (44xx except 4404)
Specs: year FE + child/household controls + department FE; all years and without 2024.
Inference for the ENAHO department-level regressor: wild cluster restricted bootstrap
(Webb 6-point weights, 1,999 draws, clusters = 25 departments), computed exactly via FWL.
Oaxaca-Blinder (pooled coefficients): Puno vs rest of Peru, anemia and hb_adj.
Engel curves (ENAHO, households with a child under 5, poor and all): heme-iron food share and
beer budget share against log per-capita expenditure; displacement regression of heme share
on beer budget share given a cubic in log per-capita expenditure and dept x year FE.
All associations are descriptive.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
import patsy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import endes_units as eu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "analysis" / "alcohol_anemia"
OUT.mkdir(parents=True, exist_ok=True)
YEARS = [y for y in range(2010, 2025) if y != 2020]
K = ["conglome", "vivienda", "hogar"]
PROBLEM = ["qs713", "qs714", "qs716", "qs717", "qs720", "qs727", "qs728", "qs729"]
RNG = np.random.default_rng(20261124)


# ---------------------------------------------------------------- readers
def rd_dta(path, cols):
    _, m = pyreadstat.read_dta(str(path), metadataonly=True)
    cl = {c.lower(): c for c in m.column_names}
    use = [cl[c] for c in cols if c in cl]
    for enc in (None, "latin1"):
        try:
            df, _ = pyreadstat.read_dta(str(path), usecols=use, encoding=enc)
            break
        except Exception:
            if enc == "latin1":
                raise
    df.columns = [c.lower() for c in df.columns]
    return df


def find_sav(d, need):
    for p in sorted(set(d.rglob("*.sav")) | set(d.rglob("*.SAV"))):
        _, m = pyreadstat.read_sav(str(p), metadataonly=True)
        cl = {c.lower(): c for c in m.column_names}
        if all(c in cl for c in need):
            return p, cl
    return None, None


def rd_sav(d, need, extra=()):
    p, cl = find_sav(d, need)
    if p is None:
        return None
    df, _ = pyreadstat.read_sav(str(p), usecols=[cl[c] for c in list(need) + list(extra) if c in cl])
    df.columns = [c.lower() for c in df.columns]
    df["hhid"] = df["hhid"].astype(str).str.strip()
    return df


# ---------------------------------------------------------------- ENDES
def endes():
    fr = []
    for y in YEARS:
        d = eu.dir_for(y)
        c = rd_sav(d, ["hhid", "hc57", "hc1"], ["hc27", "hc61", "hc53", "hc56"])
        h = rd_sav(d, ["hhid", "hv005", "hv024", "hv025", "hv001"], ["hv040", "hv008"]).drop_duplicates("hhid")
        w = rd_sav(d, ["hhid", "hv270"]).drop_duplicates("hhid")
        s = rd_sav(d, ["hhid", "qs208", "qs210"], ["qs212a", "qssexo", "qs23"] + PROBLEM)
        g = c.merge(h, on="hhid", how="left").merge(w, on="hhid", how="left")
        if s is not None:
            g = g.merge(s.drop_duplicates("hhid"), on="hhid", how="left")
        g = g[eu.true_year_mask(g, y, "hv008")]
        g["year"] = y
        fr.append(g)
        print(f"  ENDES {y}: children {len(g):,}  adult module {'yes' if s is not None else 'no'}", flush=True)
    g = pd.concat(fr, ignore_index=True)
    g = g[g.hc1.between(6, 35)].copy()
    g["anemia"] = g.hc57.isin([1, 2, 3]).astype(float).where(g.hc57.isin([1, 2, 3, 4]))
    g["hb_raw"] = (g.hc53 / 10).where(g.hc53.between(30, 250))
    g["hb_adj"] = (g.hc56 / 10).where(g.hc56.between(30, 250))
    g["wt"] = g.hv005 / 1e6
    ok = g.qs208.isin([1, 2])
    g["drank_30d"] = (g.qs210 == 1).astype(float).where(ok)
    g["beer_30d"] = (g.qs212a == 1).astype(float).where(ok)
    g["problem"] = g[[c for c in PROBLEM if c in g]].eq(1).any(axis=1).astype(float).where(ok)
    g["alt_bin"] = pd.cut(g.hv040, [-1, 500, 1500, 2500, 3500, 6000], labels=False)
    g["meduc"] = g.hc61.where(g.hc61 < 8)
    g["dept"] = g.hv024.astype(int)
    g["cluster"] = g.year.astype(str) + "_" + g.hv001.astype(int).astype(str)
    return g


# ---------------------------------------------------------------- ENAHO
HEME = ("08", "12", "13", "14", "20", "21", "22")


def enaho():
    hh = []
    for y in YEARS:
        f = rd_dta(RAW / "gastos_alimentos" / f"enaho-{y}-07.dta", K + ["p601a", "d601c", "i601c", "i601e"])
        f["code"] = f.p601a.astype(str).str.strip().str.zfill(4)
        dval = f.d601c if "d601c" in f else f.i601c
        f["v"] = pd.to_numeric(dval, errors="coerce").fillna(0) + pd.to_numeric(f.get("i601e", 0), errors="coerce").fillna(0)
        fam = f.code.str[:2]
        f["beer"] = np.where(f.code == "4404", f.v, 0)
        f["oalc"] = np.where((fam == "44") & (f.code != "4404"), f.v, 0)
        f["heme"] = np.where(fam.isin(HEME) | (f.code == "1004"), f.v, 0)
        f["food"] = np.where(fam != "44", f.v, 0)
        for c in K:
            f[c] = f[c].astype(str).str.strip()
        a = f.groupby(K)[["beer", "oalc", "heme", "food"]].sum().reset_index()
        s = rd_dta(RAW / "sumaria" / f"enaho-{y}-34.dta", K + ["ubigeo", "gashog2d", "mieperho", "factor07", "pobreza"])
        m = rd_dta(RAW / "miembros" / f"enaho-{y}-02.dta", K + ["p208a"])
        for d_ in (s, m):
            for c in K:
                d_[c] = d_[c].astype(str).str.strip()
        kids = m[pd.to_numeric(m.p208a, errors="coerce") < 5].groupby(K).size().rename("kids5").reset_index()
        d = s.merge(a, on=K, how="left").merge(kids, on=K, how="left")
        d[["beer", "oalc", "heme", "food", "kids5"]] = d[["beer", "oalc", "heme", "food", "kids5"]].fillna(0)
        d["year"] = y
        hh.append(d)
        print(f"  ENAHO {y}: hh {len(d):,}  beer buyers {100*np.average(d.beer > 0, weights=d.factor07):.1f}%", flush=True)
    d = pd.concat(hh, ignore_index=True)
    d["dept"] = d.ubigeo.astype(str).str.zfill(6).str[:2].astype(int)
    d["pw"] = d.factor07 * d.mieperho
    return d


def dept_year(d):
    rows = []
    for (y, dep), g in d.groupby(["year", "dept"]):
        rows.append(dict(year=y, dept=dep,
                         beer_pc=np.average(g.beer / g.mieperho, weights=g.pw),
                         other_pc=np.average(g.oalc / g.mieperho, weights=g.pw)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- wild cluster restricted bootstrap
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])


def wcr_p(df, y, x, rhs, w, cl, B=1999):
    Z = patsy.dmatrix(rhs, df, return_type="dataframe").values
    sw = np.sqrt(df[w].values)
    ys, xs, Zs = df[y].values * sw, df[x].values * sw, Z * sw[:, None]
    ZtZi = np.linalg.pinv(Zs.T @ Zs)
    xt = xs - Zs @ (ZtZi @ (Zs.T @ xs))          # x residualized on Z
    e = ys - Zs @ (ZtZi @ (Zs.T @ ys))           # restricted residuals (beta_x = 0)
    gid, inv = np.unique(df[cl].values, return_inverse=True)
    G = len(gid)
    s = np.bincount(inv, xt * e, G)              # sum x~ e
    q = np.bincount(inv, xt * xt, G)
    A = np.zeros((G, Zs.shape[1])); Bm = np.zeros((G, Zs.shape[1]))
    np.add.at(A, inv, Zs * e[:, None]); np.add.at(Bm, inv, Zs * xt[:, None])
    Mmat = Bm @ ZtZi @ A.T                       # G x G
    xx = q.sum()

    def tstat(v):
        b = (v * s).sum() / xx
        sc = v * s - Mmat @ v - b * q
        return b / np.sqrt((sc ** 2).sum() / xx ** 2)
    # original t: unrestricted, with v = 1 the score uses e = restricted residuals; recompute directly
    b0 = (xt * ys).sum() / xx
    u0 = ys - Zs @ (ZtZi @ (Zs.T @ ys)) - xt * b0
    sc0 = np.bincount(inv, xt * u0, G)
    t0 = b0 / np.sqrt((sc0 ** 2).sum() / xx ** 2)
    tb = np.array([tstat(RNG.choice(WEBB, G)) for _ in range(B)])
    return b0, t0, (np.abs(tb) >= abs(t0)).mean()


# ---------------------------------------------------------------- analyses
CTRL = "C(hc1) + C(hc27) + C(hv270) + C(hv025) + C(alt_bin) + C(meduc)"


def child_regs(g, dep):
    rows = []
    g = g.merge(dep, on=["year", "dept"], how="left")
    for v in ["beer_pc", "other_pc"]:
        g[v + "_z"] = (g[v] - dep[v].mean()) / dep[v].std()
    base = g.dropna(subset=["wt", "hc27", "hv270", "alt_bin", "meduc"])
    for outcome in ["anemia", "hb_raw", "hb_adj"]:
        for excl, lab in [(False, "2010-2024"), (True, "without 2024")]:
            d0 = base.dropna(subset=[outcome])
            d0 = d0[d0.year < 2024] if excl else d0
            # ENDES adult measures (2013+), many clusters: analytic cluster SE
            for x in ["drank_30d", "beer_30d", "problem"]:
                d = d0.dropna(subset=[x, "qs23", "qssexo"])
                f = f"{outcome} ~ {x} + {CTRL} + C(qssexo) + qs23 + C(year) + C(dept)"
                m = smf.wls(f, data=d, weights=d.wt).fit(cov_type="cluster", cov_kwds={"groups": d.cluster})
                rows.append(dict(source="ENDES adult", outcome=outcome, years=lab, regressor=x,
                                 coef=m.params[x], se=m.bse[x], p=m.pvalues[x], p_wcr=np.nan, n=int(m.nobs)))
            # ENAHO department-year spending: 25 clusters -> wild cluster restricted bootstrap
            for x in ["beer_pc_z", "other_pc_z"]:
                for fe in ["", " + C(dept)"]:
                    rhs = f"{CTRL} + C(year){fe}"
                    m = smf.wls(f"{outcome} ~ {x} + {rhs}", data=d0, weights=d0.wt).fit(
                        cov_type="cluster", cov_kwds={"groups": d0.dept})
                    _, _, pw = wcr_p(d0, outcome, x, rhs, "wt", "dept")
                    rows.append(dict(source="ENAHO dept-year" + (" + dept FE" if fe else ""), outcome=outcome,
                                     years=lab, regressor=x, coef=m.params[x], se=m.bse[x], p=m.pvalues[x],
                                     p_wcr=pw, n=int(m.nobs)))
            print(f"  regs done: {outcome} {lab}", flush=True)
    return pd.DataFrame(rows)


def oaxaca(g):
    out = []
    d = g.dropna(subset=["drank_30d", "problem", "hc27", "hv270", "alt_bin", "meduc", "wt"]).copy()
    d["puno"] = (d.dept == 21).astype(float)
    groups = {"alcohol (adult drank 30d, problem drinking)": ["drank_30d", "problem"],
              "wealth quintile": "C(hv270)", "mother's education": "C(meduc)",
              "urban/rural": "C(hv025)", "altitude": "C(alt_bin)",
              "child age and sex": ["C(hc1)", "C(hc27)"]}
    terms = []
    for v in groups.values():
        terms += v if isinstance(v, list) else [v]
    for outcome in ["anemia", "hb_adj", "hb_raw"]:
        e = d.dropna(subset=[outcome])
        y, X = patsy.dmatrices(f"{outcome} ~ " + " + ".join(terms) + " + C(year) + puno", e, return_type="dataframe")
        w = e.loc[X.index, "wt"].values
        beta = np.linalg.lstsq(X.values * np.sqrt(w)[:, None], y.values.ravel() * np.sqrt(w), rcond=None)[0]
        pn = X.puno.values == 1
        diff = np.average(X.values[pn], axis=0, weights=w[pn]) - np.average(X.values[~pn], axis=0, weights=w[~pn])
        contrib = pd.Series(diff * beta, index=X.columns)
        gap = np.average(y.values.ravel()[pn], weights=w[pn]) - np.average(y.values.ravel()[~pn], weights=w[~pn])
        for gname, v in groups.items():
            keys = v if isinstance(v, list) else [v]
            cols = [c for c in X.columns if any(c == k or c.startswith(k + "[") for k in keys)]
            out.append(dict(outcome=outcome, component=gname, value=contrib[cols].sum()))
        yr = [c for c in X.columns if c.startswith("C(year)")]
        out.append(dict(outcome=outcome, component="survey year mix", value=contrib[yr].sum()))
        expl = sum(r["value"] for r in out if r["outcome"] == outcome)
        out.append(dict(outcome=outcome, component="TOTAL explained", value=expl))
        out.append(dict(outcome=outcome, component="unexplained", value=gap - expl))
        out.append(dict(outcome=outcome, component="GAP Puno - rest", value=gap))
    return pd.DataFrame(out)


def engel(d):
    d = d[(d.kids5 > 0) & (d.food > 0) & (d.gashog2d > 0)].copy()
    d["lpc"] = np.log(d.gashog2d / d.mieperho)
    d["heme_share"] = 100 * d.heme / d.food
    d["beer_bshare"] = 100 * d.beer / d.gashog2d
    d["lsize"] = np.log(d.mieperho)
    d["poor"] = d.pobreza.isin([1, 2])
    d["dy"] = d.dept.astype(str) + "_" + d.year.astype(str)
    rows = []
    for lab, s in {"all households with child <5": d, "poor households with child <5": d[d.poor]}.items():
        for yv in ["heme_share", "beer_bshare"]:
            m = smf.wls(f"{yv} ~ lpc + lsize + C(dy)", data=s, weights=s.factor07).fit(
                cov_type="cluster", cov_kwds={"groups": s.conglome + s.year.astype(str)})
            rows.append(dict(sample=lab, model=f"Engel slope of {yv} on log pc exp", coef=m.params["lpc"],
                             se=m.bse["lpc"], p=m.pvalues["lpc"], n=int(m.nobs),
                             mean=np.average(s[yv], weights=s.factor07)))
        m = smf.wls("heme_share ~ beer_bshare + lpc + I(lpc**2) + I(lpc**3) + lsize + C(dy)", data=s,
                    weights=s.factor07).fit(cov_type="cluster", cov_kwds={"groups": s.conglome + s.year.astype(str)})
        rows.append(dict(sample=lab, model="heme share on beer budget share (pp per pp)", coef=m.params["beer_bshare"],
                         se=m.bse["beer_bshare"], p=m.pvalues["beer_bshare"], n=int(m.nobs),
                         mean=np.average(s.heme_share, weights=s.factor07)))
        s2 = s.assign(any_beer=(s.beer > 0).astype(float))
        m = smf.wls("heme_share ~ any_beer + lpc + I(lpc**2) + I(lpc**3) + lsize + C(dy)", data=s2,
                    weights=s2.factor07).fit(cov_type="cluster", cov_kwds={"groups": s2.conglome + s2.year.astype(str)})
        rows.append(dict(sample=lab, model="heme share on any beer purchase (pp)", coef=m.params["any_beer"],
                         se=m.bse["any_beer"], p=m.pvalues["any_beer"], n=int(m.nobs),
                         mean=np.average(s2.any_beer, weights=s2.factor07)))
    # binned Engel curves for the figure/table
    d["dec"] = d.groupby("year").lpc.transform(lambda v: pd.qcut(v, 10, labels=False))
    bins = d.groupby("dec").apply(lambda s: pd.Series({
        "heme_share": np.average(s.heme_share, weights=s.factor07),
        "beer_bshare": np.average(s.beer_bshare, weights=s.factor07),
        "beer_buyers": 100 * np.average(s.beer > 0, weights=s.factor07)}))
    return pd.DataFrame(rows), bins


def main():
    pd.set_option("display.width", 220)
    cg, cd = OUT / "_cache_endes_children.parquet", OUT / "_cache_enaho_households.parquet"
    if cg.exists():
        g = pd.read_parquet(cg)
    else:
        g = endes()
        g.to_parquet(cg)
    for y in [2010, 2015, 2019, 2023, 2024]:
        s = g[(g.year == y) & g.anemia.notna()]
        print(f"  check anemia {y}: {100*np.average(s.anemia, weights=s.wt):.1f}%")
    if cd.exists():
        d = pd.read_parquet(cd)
    else:
        d = enaho()
        d.to_parquet(cd)
    dep = dept_year(d)
    dep.to_csv(OUT / "ext_enaho_beer_dept_year.csv", index=False)
    r = child_regs(g, dep)
    r.to_csv(OUT / "ext_child_regressions.csv", index=False)
    print(r.round(4).to_string())
    o = oaxaca(g)
    o.to_csv(OUT / "ext_oaxaca_puno.csv", index=False)
    print(o.round(4).to_string())
    e, bins = engel(d)
    e.to_csv(OUT / "ext_engel.csv", index=False)
    bins.to_csv(OUT / "ext_engel_bins.csv")
    print(e.round(4).to_string())
    print(bins.round(3).to_string())


if __name__ == "__main__":
    main()
