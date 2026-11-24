"""
analysis_alcohol_engel.py - Engel curves and budget displacement, light version.

Uses the ENAHO household cache written by analysis_alcohol_extended.py (2010-2024, no 2020).
Households with a child under 5 (all, and poor). Department x year fixed effects are
absorbed by weighted within-group demeaning (FWL), then WLS with SE clustered by
conglomerado x year. Outcomes in percent:
  heme_share   heme-iron foods / non-alcohol food spending
  beer_bshare  beer / total household expenditure
Models: Engel slope on log per-capita expenditure (+ log household size);
heme share on beer budget share and on any beer purchase, given a cubic in log pc exp.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "analysis" / "alcohol_anemia"


def demean(df, cols, g, w):
    out = {}
    for c in cols:
        m = (df[c] * df[w]).groupby(df[g]).transform("sum") / df[w].groupby(df[g]).transform("sum")
        out[c] = df[c] - m
    return pd.DataFrame(out)


def wls_cluster(y, X, w, cl):
    sw = np.sqrt(w)
    Xs, ys = X * sw[:, None], y * sw
    XtXi = np.linalg.inv(Xs.T @ Xs)
    b = XtXi @ Xs.T @ ys
    u = ys - Xs @ b
    _, inv = np.unique(cl, return_inverse=True)
    S = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(S, inv, Xs * u[:, None])
    V = XtXi @ (S.T @ S) @ XtXi
    G = inv.max() + 1
    V *= G / (G - 1)
    return b, np.sqrt(np.diag(V))


def main():
    d = pd.read_parquet(OUT / "_cache_enaho_households.parquet")
    d = d[(d.kids5 > 0) & (d.food > 0) & (d.gashog2d > 0)].copy()
    d["lpc"] = np.log(d.gashog2d / d.mieperho)
    d["lpc2"], d["lpc3"] = d.lpc ** 2, d.lpc ** 3
    d["heme_share"] = 100 * d.heme / d.food
    d["beer_bshare"] = 100 * d.beer / d.gashog2d
    d["any_beer"] = (d.beer > 0).astype(float)
    d["lsize"] = np.log(d.mieperho)
    d["poor"] = d.pobreza.isin([1, 2])
    d["dy"] = d.dept.astype(str) + "_" + d.year.astype(str)
    d["cl"] = d.conglome.astype(str) + "_" + d.year.astype(str)
    rows = []
    for lab, s in {"all households with child <5": d, "poor households with child <5": d[d.poor]}.items():
        s = s.reset_index(drop=True)
        cols = ["heme_share", "beer_bshare", "any_beer", "lpc", "lpc2", "lpc3", "lsize"]
        z = demean(s, cols, "dy", "factor07")
        w, cl = s.factor07.values, s.cl.values
        specs = [("Engel slope: heme share on log pc exp", "heme_share", ["lpc", "lsize"], "lpc"),
                 ("Engel slope: beer budget share on log pc exp", "beer_bshare", ["lpc", "lsize"], "lpc"),
                 ("heme share on beer budget share (pp per pp)", "heme_share", ["beer_bshare", "lpc", "lpc2", "lpc3", "lsize"], "beer_bshare"),
                 ("heme share on any beer purchase (pp)", "heme_share", ["any_beer", "lpc", "lpc2", "lpc3", "lsize"], "any_beer")]
        for name, y, xs, key in specs:
            b, se = wls_cluster(z[y].values, z[xs].values, w, cl)
            k = xs.index(key)
            rows.append(dict(sample=lab, model=name, coef=b[k], se=se[k], t=b[k] / se[k], n=len(s),
                             mean_y=np.average(s[y], weights=w)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "ext_engel.csv", index=False)
    d["dec"] = d.groupby("year").lpc.transform(lambda v: pd.qcut(v, 10, labels=False))
    bins = d.groupby("dec").apply(lambda s: pd.Series({
        "heme_share": np.average(s.heme_share, weights=s.factor07),
        "beer_budget_share": np.average(s.beer_bshare, weights=s.factor07),
        "pct_buying_beer": 100 * np.average(s.any_beer, weights=s.factor07)}))
    bins.to_csv(OUT / "ext_engel_bins.csv")
    pd.set_option("display.width", 200)
    print(t.round(4).to_string())
    print(bins.round(3).to_string())


if __name__ == "__main__":
    main()
