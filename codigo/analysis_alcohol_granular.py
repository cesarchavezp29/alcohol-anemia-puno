"""
analysis_alcohol_granular.py - finer time, geography and individual heterogeneity
(suggestion from a reader of the thread; design discussed with ChatGPT, rounds 8-9).

Builds on _predictors_features.parquet (ENDES 2014-2024, children 6-35 months linked to the
mother, adult alcohol module available; 87,952 children). Adds:
  district   UBIGEO from RECH0 (2019+), kept only if it exists in the INEI 2025 polygons; for all
             years, the cluster's (displaced) GPS point is assigned to the polygon that contains it,
             or to the nearest polygon within 5 km that belongs to the survey department. Validated
             against UBIGEO by child, by cluster-year and by cluster-year in Puno. Analysis uses
             UBIGEO when present, else GPS; children without a district are dropped.
  date       exact day, month and year of the hemoglobin measurement (RECH6 hc17, hc18, hc19);
             day of year on a common 365-day calendar. Rows without an exact date are excluded
             from date-based designs (never imputed).
  respondent QSNUMERO; respondent is the child's mother when QSNUMERO == hc60.
Parts
  1. Fixed-effect ladder on one common sample (singletons of the finest FE removed first), with
     the basic controls and with the full M2 controls of analysis_anemia_predictors.py
     (mother's hemoglobin, morbidity, breastfeeding, height-for-age, birth weight, etc.).
  2. Heterogeneity (district + year-month FE in both samples), Holm-adjusted interaction p-values;
     hemoglobin thresholds and continuous hemoglobin.
  3. Monthly profiles; Candelaria: exposure = days of the fiesta (2-16 February, INEI directory)
     inside the adult's 30-day recall window, proxied by the hemoglobin measurement date
     (the adult health module has no interview day in most years); lagged window = measured
     31-90 days after the fiesta ended. Puno vs other areas at 3,000 m or more, district +
     year-month FE, the window main effects included. Inference by cluster and by district;
     leave-one-year-out; the same January-April contrast for each other highland department
     (placebo); Puno-specific bimonthly profile for drinking, anemia, hemoglobin, morbidity and diet.
  4. ENAHO: households acquiring beer for the home (purchases plus other acquisitions) and beer
     per person (ratio of weighted totals), by survey month.
All associations are descriptive.
"""
from __future__ import annotations
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import pyfixest as pf
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import endes_units as eu  # noqa: E402
from analysis_alcohol_extended import rd_sav  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "alcohol_anemia"
SHP = Path(os.environ.get("INEI_DISTRICT_SHP", str(Path(__file__).resolve().parents[1] / "raw" / "geo" / "Limite_Distrital_INEI_2025_CPV.shp")))
YEARS = [y for y in range(2014, 2025) if y != 2020]
ALC = ["freq_problem", "heavy_L", "drank_30d", "occasions"]
CTRL = "C(hc1) + C(hc27) + C(hv270) + C(hv025) + C(alt_bin) + C(meduc) + C(qssexo) + qs23"
M2VARS = ["m_hb_adj", "diarrhea", "fever", "cough", "breastfed_now", "haz", "bw_kg", "small", "m_height", "m_age",
          "hhsize", "piped_water", "sewer", "biomass", "short_int"]
FIT = dict(fixef_maxiter=200000)
HIGH_DEPTS = {3: "Apurímac", 5: "Ayacucho", 8: "Cusco", 9: "Huancavelica", 12: "Junín", 19: "Pasco"}
FIESTA = (33, 47)  # 2 to 16 February, day of year on a 365-day calendar


def extra():
    fr = []
    for y in YEARS:
        d = eu.dir_for(y)
        h = rd_sav(d, ["hhid", "hv005", "hv024"], ["ubigeo", "longitudx", "latitudy", "long_ccpp", "lat_ccpp"]).drop_duplicates("hhid")
        if "latitudy" not in h and "lat_ccpp" in h:   # 2017 names
            h = h.rename(columns={"lat_ccpp": "latitudy", "long_ccpp": "longitudx"})
        h = h.drop(columns=[c for c in ["lat_ccpp", "long_ccpp"] if c in h])
        c = rd_sav(d, ["hhid", "hc0", "hc53"], ["hc17", "hc18", "hc19"]).drop_duplicates(["hhid", "hc0"])
        s = rd_sav(d, ["hhid", "qs208"], ["qsnumero"]).drop_duplicates("hhid")
        g = c.merge(h.drop(columns=["hv005", "hv024"]), on="hhid", how="left") \
             .merge(s.drop(columns="qs208"), on="hhid", how="left")
        g["year"] = y
        for col in ["ubigeo", "longitudx", "latitudy", "hc17", "hc18", "hc19", "qsnumero"]:
            if col not in g:
                g[col] = np.nan
        g["ubigeo"] = g["ubigeo"].astype("string")
        fr.append(g.drop(columns="hc53"))
    return pd.concat(fr, ignore_index=True)


def measure_dates():
    """Day/month/year of the child's measurement from REC44 (caseid + hwidx), all years."""
    from analysis_anemia_predictors import rd_case
    fr = []
    for y in YEARS:
        t = rd_case(eu.dir_for(y), ["caseid", "hwidx", "hw17"], ["hw18", "hw19"])
        if t is None:
            continue
        t = t.rename(columns={"mline": "hc60", "hwidx": "bidx"}).drop_duplicates(["hhid", "hc60", "bidx"])
        t["year"] = y
        fr.append(t)
    return pd.concat(fr, ignore_index=True)


def districts(g):
    geo = gpd.read_file(SHP)[["UBIGEO", "geometry"]]
    valid = set(geo.UBIGEO)
    pts = g[["latitudy", "longitudx"]].dropna()
    pts = pts[(pts.latitudy.between(-19, 0.5)) & (pts.longitudx.between(-82, -68))]
    gp = gpd.GeoDataFrame(index=pts.index, geometry=gpd.points_from_xy(pts.longitudx, pts.latitudy), crs="EPSG:4326")
    gp = gp.to_crs(geo.crs)
    j = gpd.sjoin(gp, geo, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    miss = j.UBIGEO.isna()
    if miss.any():  # outside every polygon (coast, lake): nearest within 5 km, checked against the department below
        near = gpd.sjoin_nearest(gp.loc[miss[miss].index].to_crs(32718), geo.to_crs(32718), how="left",
                                 max_distance=5000)
        near = near[~near.index.duplicated()]
        j.loc[near.index, "UBIGEO"] = near.UBIGEO
    gps = j.UBIGEO.reindex(g.index).astype("string").str.zfill(6)
    dep_code = g.hv024.astype(int).astype(str).str.zfill(2)
    g["ubigeo_gps"] = gps.where(gps.str[:2] == dep_code)
    off = g["ubigeo"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    g["ubigeo_off"] = off.where(off.isin(valid))
    both = g.ubigeo_off.notna() & g.ubigeo_gps.notna()
    eq = (g.ubigeo_off == g.ubigeo_gps)
    cy = g.year.astype(str) + "_" + g.hv001.astype(int).astype(str)
    byc = pd.DataFrame({"cy": cy[both], "same": eq[both], "puno": g.puno[both]}).drop_duplicates("cy")
    val = dict(n_children=int(both.sum()), district_children=float(eq[both].mean()),
               province_children=float((g.ubigeo_off[both].str[:4] == g.ubigeo_gps[both].str[:4]).mean()),
               n_clusters=len(byc), district_clusters=float(byc.same.mean()),
               n_clusters_puno=int((byc.puno == 1).sum()), district_clusters_puno=float(byc[byc.puno == 1].same.mean()),
               gps_dropped_wrong_dept=int((gps.notna() & g.ubigeo_gps.isna()).sum()))
    for lab, m in [("urban", g.hv025 == 1), ("rural", g.hv025 == 2)]:
        b = both & m
        val[f"district_children_{lab}"] = float(eq[b].mean())
    g["dist"] = g.ubigeo_off.fillna(g.ubigeo_gps).astype("string")
    return g, val


def prep():
    d = pd.read_parquet(OUT / "_predictors_features.parquet")
    cache = OUT / "_cache_granular_extra.parquet"
    if cache.exists():
        x = pd.read_parquet(cache)
    else:
        x = extra()
        x.to_parquet(cache)
    d = d.merge(x, on=["hhid", "hc0", "year"], how="left", validate="one_to_one")
    c44 = OUT / "_cache_rec44_dates.parquet"
    if c44.exists():
        m44 = pd.read_parquet(c44)
    else:
        m44 = measure_dates()
        m44.to_parquet(c44)
    d = d.merge(m44, on=["hhid", "hc60", "bidx", "year"], how="left", validate="many_to_one")
    for a_, b_ in [("hc17", "hw17"), ("hc18", "hw18"), ("hc19", "hw19")]:
        d[a_] = d[a_].where(d[a_].notna(), d[b_])
    d, val = districts(d)
    n0 = len(d)
    d = d[d.dist.notna()].copy()
    val["dropped_no_district"] = n0 - len(d)
    d["prov"] = d.dist.str[:4]
    # dates
    cmc_m = ((d.hv008 - 1) % 12 + 1)
    d["month"] = d.hc18.where(d.hc18.between(1, 12), cmc_m).astype(int)
    d["myear"] = d.hc19.where(d.hc19.between(2013, 2025), d.year).astype(int)
    d["day"] = d.hc17.where(d.hc17.between(1, 31))
    day365 = d.day.where(d.month != 2, d.day.clip(upper=28))          # 29 February -> 28 February
    dt = pd.to_datetime(dict(year=2013, month=d.month, day=day365.fillna(1)), errors="coerce")
    d["exact"] = d.day.notna() & d.hc18.between(1, 12) & dt.notna()
    d["doy"] = dt.dt.dayofyear.where(d.exact)
    assert d.loc[d.exact, "doy"].notna().all()
    d["ym"] = d.year.astype(str) + "_" + d.month.astype(str).str.zfill(2)
    d["dist_year"] = d.dist.astype(str) + "_" + d.year.astype(str)
    d["poor"] = (d.hv270 <= 2).astype(float)          # two lowest wealth-index quintiles
    d["resp_female"] = (d.qssexo == 2).astype(float)
    d["resp_mother"] = (d.qsnumero == d.hc60).astype(float).where(d.qsnumero.notna())
    d["young"] = (d.hc1 <= 23).astype(float)
    d["area3"] = np.select([d.puno == 1, d.hv040 >= 3000], ["Puno", "Otras zonas sobre 3.000 m"], "Resto del Perú")
    # Candelaria exposure: fiesta days inside the 30-day recall window ending on the measurement date
    lo, hi = FIESTA
    t = d.doy
    d["fdays"] = (np.minimum(t, hi) - np.maximum(t - 29, lo) + 1).clip(lower=0).where(d.exact)
    d["fd15"] = d.fdays / (hi - lo + 1)
    d["lagwin"] = t.between(hi + 31, hi + 90).astype(float).where(d.exact)
    # M2 controls: mean-filled with missing flags
    m2 = []
    for v in M2VARS:
        d[f"f_{v}"] = d[v].fillna(d[v].mean())
        m2.append(f"f_{v}")
        if d[v].isna().any():
            d[f"mis_{v}"] = d[v].isna().astype(float)
            m2.append(f"mis_{v}")
    d["bord_c"] = d.bord_c.fillna(-1)
    global CTRL_M2
    CTRL_M2 = CTRL + " + C(bord_c) + " + " + ".join(m2)
    for c in ["dept", "prov", "dist", "dist_year", "ym", "hc1", "hc27", "hv270", "hv025", "alt_bin", "meduc",
              "qssexo", "month", "bord_c"]:
        d[c] = d[c].astype(str)
    return d, val


def drop_singletons(s, fes):
    while True:
        keep = np.ones(len(s), bool)
        for f in fes:
            keep &= s.groupby(f)[f].transform("size").values > 1
        if keep.all():
            return s
        s = s[keep]


def coef_puno(m, x):
    b, V = m.coef(), m._vcov
    names = list(b.index)
    i, j = names.index(x), names.index("x_puno")
    return b[x], m.se()[x], b[x] + b["x_puno"], np.sqrt(V[i, i] + V[j, j] + 2 * V[i, j])


def fe_ladder(d):
    ladder = [("Departamento + año", "dept + year"), ("Provincia + año", "prov + year"),
              ("Distrito + año", "dist + year"), ("Distrito + año-mes", "dist + ym"),
              ("Distrito-año + año-mes", "dist_year + ym")]
    rows = []
    for ctrl_lab, ctrl in [("básicos", CTRL), ("M2", CTRL_M2)]:
        for x in ALC:
            s = d.dropna(subset=[x, "anemia_2024", "qs23"]).copy()
            s = drop_singletons(s, ["dist_year", "ym"])    # one common sample for the whole ladder
            s["x_puno"] = s[x] * s.puno
            for lab, fe in ladder:
                m = pf.feols(f"anemia_2024 ~ {x} + x_puno + {ctrl} | {fe}", data=s, weights="wt",
                             vcov={"CRV1": "cluster"}, **FIT)
                b, se, bp, sep = coef_puno(m, x)
                rows.append(dict(controls=ctrl_lab, regressor=x, fe=lab, rest=100 * b, rest_se=100 * se,
                                 puno=100 * bp, puno_se=100 * sep, n=m._N))
            print(f"  ladder {ctrl_lab} {x} n={len(s)}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "gran_fe_ladder.csv", index=False)
    return t


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def heterogeneity(d):
    rows = []
    for x in ["freq_problem", "heavy_L", "drank_30d"]:
        for mod in ["poor", "resp_female", "resp_mother", "young", "rural"]:
            s = d.dropna(subset=[x, mod, "anemia_2024", "qs23"]).copy()
            s["xm"] = s[x] * s[mod]
            for lab, sub in [("Perú", s), ("Puno", s[s.puno == 1])]:
                try:
                    m = pf.feols(f"anemia_2024 ~ {x} + xm + {mod} + {CTRL} | dist + ym", data=sub,
                                 weights="wt", vcov={"CRV1": "cluster"}, **FIT)
                except Exception as e:
                    print("   skip", x, mod, lab, e)
                    continue
                b, V = m.coef(), m._vcov
                names = list(b.index)
                i, j = names.index(x), names.index("xm")
                rows.append(dict(regressor=x, moderator=mod, sample=lab, b0=100 * b[x], se0=100 * m.se()[x],
                                 b1=100 * (b[x] + b["xm"]), se1=100 * np.sqrt(V[i, i] + V[j, j] + 2 * V[i, j]),
                                 diff=100 * b["xm"], p_diff=m.pvalue()["xm"], n=m._N,
                                 share_mod=float(np.average(sub[mod], weights=sub.wt))))
    t = pd.DataFrame(rows)
    for lab in ["Perú", "Puno"]:
        k = t["sample"] == lab
        t.loc[k, "p_holm"] = holm(t.loc[k, "p_diff"])
    t.to_csv(OUT / "gran_heterogeneity.csv", index=False)
    rows = []
    for c in [9.0, 10.0, 10.5, 11.0, 11.5]:
        d[f"below_{c}"] = (d.hb_adj24 < c).astype(float)
    for x in ["freq_problem", "heavy_L", "drank_30d"]:
        s = d.dropna(subset=[x, "hb_adj24", "qs23"]).copy()
        s["x_puno"] = s[x] * s.puno
        for yv in [f"below_{c}" for c in [9.0, 10.0, 10.5, 11.0, 11.5]] + ["hb_adj24"]:
            m = pf.feols(f"{yv} ~ {x} + x_puno + {CTRL} | dist + ym", data=s, weights="wt",
                         vcov={"CRV1": "cluster"}, **FIT)
            b, se, bp, sep = coef_puno(m, x)
            k = 1 if yv == "hb_adj24" else 100
            rows.append(dict(regressor=x, outcome=yv, rest=k * b, rest_se=k * se, puno=k * bp, puno_se=k * sep,
                             mean=float(np.average(s[yv], weights=s.wt)) * k))
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / "gran_hb_thresholds.csv", index=False)
    return t, t2


def K(yv):
    return 100 if yv in ("drank_30d", "beer_heavy", "anemia_2024", "diarrhea", "fever", "cough", "heme_food",
                         "d_dairy", "diet_div") else 1


def monthly(d):
    rows = []
    for area, s in d.groupby("area3"):
        for mth, q in s.groupby(s.month.astype(int)):
            r = dict(area=area, month=mth, n=len(q))
            for v in ["drank_30d", "occasions", "heavy_L", "beer_heavy", "anemia_2024", "hb_adj24"]:
                z = q.dropna(subset=[v])
                mu = np.average(z[v], weights=z.wt)
                r[v], r[v + "_se"] = K(v) * mu, K(v) * np.sqrt(np.average((z[v] - mu) ** 2, weights=z.wt) / len(z))
            rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "gran_monthly_profiles.csv", index=False)

    hi = d[(d.area3 != "Resto del Perú") & d.exact].copy()
    hi["fd_p"], hi["lag_p"] = hi.fd15 * hi.puno, hi.lagwin * hi.puno
    outs = ["drank_30d", "occasions", "heavy_L", "beer_heavy", "anemia_2024", "hb_adj24"]
    rows = []
    for yv in outs:
        s = hi.dropna(subset=[yv])
        for vc_lab, vc in [("conglomerado", {"CRV1": "cluster"}), ("distrito", {"CRV1": "dist"})]:
            m = pf.feols(f"{yv} ~ fd15 + fd_p + lagwin + lag_p | dist + ym + hc1 + hv270", data=s, weights="wt",
                         vcov=vc, **FIT)
            rows.append(dict(outcome=yv, cluster=vc_lab, fiesta=K(yv) * m.coef()["fd_p"], fiesta_se=K(yv) * m.se()["fd_p"],
                             lag=K(yv) * m.coef()["lag_p"], lag_se=K(yv) * m.se()["lag_p"], n=m._N,
                             n_puno_exposed=int((s.fdays > 0).mul(s.puno).sum()), n_puno_lag=int(s.lag_p.sum())))
        loo = []
        for y in sorted(s.year.unique()):
            q = s[s.year != y]
            m = pf.feols(f"{yv} ~ fd15 + fd_p + lagwin + lag_p | dist + ym + hc1 + hv270", data=q, weights="wt",
                         vcov={"CRV1": "cluster"}, **FIT)
            loo.append(K(yv) * m.coef()["fd_p"])
        rows[-1]["loo_min"], rows[-1]["loo_max"] = min(loo), max(loo)
        rows[-2]["loo_min"], rows[-2]["loo_max"] = min(loo), max(loo)
    w = pd.DataFrame(rows)
    w.to_csv(OUT / "gran_candelaria_windows.csv", index=False)

    # bimonthly Puno profile (ref jul-ago), district + year-month FE; also morbidity and diet
    hm = d[d.area3 != "Resto del Perú"].copy()
    hm["bim"] = ((hm.month.astype(int) - 1) // 2).astype(int)
    labs = ["ene-feb", "mar-abr", "may-jun", "jul-ago", "sep-oct", "nov-dic"]
    for k in range(6):
        hm[f"bp{k}"] = ((hm.bim == k) & (hm.puno == 1)).astype(float)
    rows = []
    for yv in ["drank_30d", "beer_heavy", "anemia_2024", "hb_adj24", "m_hb_adj", "diarrhea", "fever", "cough",
               "heme_food", "d_dairy", "diet_div"]:
        s = hm.dropna(subset=[yv])
        m = pf.feols(f"{yv} ~ " + " + ".join(f"bp{k}" for k in range(6) if k != 3) + " | dist + ym + hc1 + hv270",
                     data=s, weights="wt", vcov={"CRV1": "cluster"}, **FIT)
        for k in range(6):
            c, se = (0.0, 0.0) if k == 3 else (K(yv) * m.coef()[f"bp{k}"], K(yv) * m.se()[f"bp{k}"])
            rows.append(dict(outcome=yv, window=labs[k], coef=c, se=se, n=m._N))
    b = pd.DataFrame(rows)
    b.to_csv(OUT / "gran_puno_bimonthly.csv", index=False)

    # placebo: the same January-April contrast for each other highland department (vs highland areas outside Puno)
    rows = []
    hm["janapr"] = hm.month.astype(int).between(1, 4).astype(float)
    for code, name in [(21, "Puno")] + list(HIGH_DEPTS.items()):
        s0 = hm if code == 21 else hm[hm.puno == 0]
        s0 = s0.assign(tr=(s0.dept.astype(int) == code).astype(float))
        s0["tr_ja"] = s0.tr * s0.janapr
        for yv in ["anemia_2024", "hb_adj24", "drank_30d"]:
            s = s0.dropna(subset=[yv])
            m = pf.feols(f"{yv} ~ tr_ja | dist + ym + hc1 + hv270", data=s, weights="wt", vcov={"CRV1": "cluster"}, **FIT)
            rows.append(dict(dept=name, outcome=yv, coef=K(yv) * m.coef()["tr_ja"], se=K(yv) * m.se()["tr_ja"],
                             n_dept=int(s.tr.sum())))
    p = pd.DataFrame(rows)
    p.to_csv(OUT / "gran_janapr_placebo.csv", index=False)
    return t, w, b, p


def candelaria_days(d):
    """Days of the district's own Candelaria (INEI directory) inside the 30-day window ending on the measurement date."""
    f = pd.read_csv(ROOT / "raw" / "festividades" / "festividades_distritales_2013.csv").dropna(subset=["UBIGEO"])
    f["UBIGEO"] = f.UBIGEO.astype(int).astype(str).str.zfill(6)
    f = f[f.fiesta.str.upper().str.contains("CANDELARIA", na=False)].copy()
    day = f.day.astype(int).where(f.month != 2, f.day.astype(int).clip(upper=28))
    f["s"] = pd.to_datetime(dict(year=2013, month=f.month.astype(int), day=day)).dt.dayofyear
    f["L"] = f.days.fillna(1).clip(lower=1).astype(int)
    f = f.sort_values("L", ascending=False).drop_duplicates("UBIGEO").set_index("UBIGEO")
    s_ = d.dist.map(f.s)
    L_ = d.dist.map(f.L)
    t = d.doy
    ov = sum((((t - k - s_) % 365) < L_).astype(float) for k in range(30))
    d["cand_days"] = ov.where(s_.notna(), 0.0).where(d.exact)
    since_end = (t - (s_ + L_ - 1)) % 365
    d["cand_lag"] = since_end.between(31, 90).astype(float).where(s_.notna(), 0.0).where(d.exact)
    d["cand_len"] = L_.fillna(0)
    return d


def fe_support(s, x, fe):
    """Share of the variance of x left after the fixed effects (how much variation the coefficient uses)."""
    r = pf.feols(f"{x} ~ 1 | {fe}", data=s, weights="wt", **FIT)
    res = np.asarray(r.resid()).ravel()
    w = np.asarray(r._weights).ravel()            # effective sample (singletons dropped by pyfixest)
    xv = np.asarray(r._Y).ravel()
    return float(np.average(res ** 2, weights=w) / np.average((xv - np.average(xv, weights=w)) ** 2, weights=w))


def season_tests(d):
    rows = []
    outs = ["drank_30d", "beer_heavy", "anemia_2024", "hb_adj24"]
    # 1. Puno vs other highland areas, adding Puno x calendar-month FE
    hi = d[(d.area3 != "Resto del Perú") & d.exact].copy()
    hi["fd_p"], hi["lag_p"] = hi.fd15 * hi.puno, hi.lagwin * hi.puno
    hi["puno_m"] = hi.puno.astype(int).astype(str) + "_" + hi.month.astype(str)
    for spec, fe in [("año-mes", "dist + ym + hc1 + hv270"),
                     ("+ Puno x mes calendario", "dist + ym + puno_m + hc1 + hv270")]:
        sup = fe_support(hi.dropna(subset=["anemia_2024"]), "fd_p", fe)
        for yv in outs:
            s = hi.dropna(subset=[yv])
            m = pf.feols(f"{yv} ~ fd15 + fd_p + lagwin + lag_p | {fe}", data=s, weights="wt",
                         vcov={"CRV1": "dist"}, **FIT)
            rows.append(dict(test="Puno vs zonas altas", spec=spec, outcome=yv, coef=K(yv) * m.coef()["fd_p"],
                             se=K(yv) * m.se()["fd_p"], support=sup, n=m._N))
    # 2. within Puno: districts differ in whether and how long they celebrate the Candelaria
    p = candelaria_days(d[(d.puno == 1) & d.exact].copy())
    p["cand15"] = p.cand_days / 15
    grp = p.drop_duplicates("dist").cand_len
    for yv in outs:
        s = p.dropna(subset=[yv])
        m = pf.feols(f"{yv} ~ cand15 + cand_lag | dist + ym + hc1 + hv270", data=s, weights="wt",
                     vcov={"CRV1": "dist"}, **FIT)
        rows.append(dict(test="Dentro de Puno", spec="distrito + año-mes", outcome=yv,
                         coef=K(yv) * m.coef()["cand15"], se=K(yv) * m.se()["cand15"],
                         support=fe_support(s, "cand15", "dist + ym + hc1 + hv270"), n=m._N,
                         n_exposed=int((s.cand_days > 0).sum()), districts_long=int((grp >= 7).sum()),
                         districts_short=int(((grp > 0) & (grp < 7)).sum()), districts_none=int((grp == 0).sum())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "gran_candelaria_tests.csv", index=False)
    # 3. Gelbach decomposition of the Puno x January-April coefficient (highland areas, district + year-month FE)
    hm = d[d.area3 != "Resto del Perú"].copy()
    hm["pj"] = (hm.puno * hm.month.astype(int).between(1, 4)).astype(float)
    for v in ["heme_food", "d_dairy", "diet_div"]:
        hm[f"f_{v}"] = hm[v].fillna(hm[v].mean())
        hm[f"mis_{v}"] = hm[v].isna().astype(float)
    blocks = {"Edad, sexo y orden de nacimiento": ["hc1", "hc27", "bord_c"], "Altitud y zona": ["alt_bin", "hv025"],
              "Riqueza y hogar": ["hv270", "f_hhsize", "f_piped_water", "f_sewer", "f_biomass"],
              "Diarrea, fiebre y tos": ["f_diarrhea", "f_fever", "f_cough", "mis_diarrhea", "mis_fever", "mis_cough"],
              "Alimentación de ayer": ["f_heme_food", "f_d_dairy", "f_diet_div", "mis_heme_food", "mis_d_dairy",
                                       "mis_diet_div"],
              "Hemoglobina de la madre": ["f_m_hb_adj", "mis_m_hb_adj"],
              "Alcohol del adulto": ["drank_30d", "occasions", "heavy_L", "freq_problem"]}
    X, owner = pd.DataFrame(index=hm.index), {}
    for bname, vs in blocks.items():
        for v in vs:
            if v not in hm:
                continue
            dm = pd.get_dummies(hm[v], prefix=v, drop_first=True, dtype=float) if hm[v].dtype == object \
                else hm[[v]].astype(float)
            for c in dm.columns:
                cc = "z_" + "".join(ch if ch.isalnum() else "_" for ch in c)
                X[cc] = dm[c].values
                owner[cc] = bname
    X = X.loc[:, X.std() > 0]
    hm = pd.concat([hm, X], axis=1)
    zc = list(X.columns)
    rows = []
    for yv in ["anemia_2024", "hb_adj24"]:
        s = hm.dropna(subset=[yv] + zc)
        base = pf.feols(f"{yv} ~ pj | dist + ym", data=s, weights="wt", vcov={"CRV1": "dist"}, **FIT)
        full = pf.feols(f"{yv} ~ pj + {' + '.join(zc)} | dist + ym", data=s, weights="wt", vcov={"CRV1": "dist"}, **FIT)
        bfull = full.coef()
        contrib = {}
        for c in zc:
            if c not in bfull:
                continue
            gam = pf.feols(f"{c} ~ pj | dist + ym", data=s, weights="wt", **FIT).coef()["pj"]
            contrib[owner[c]] = contrib.get(owner[c], 0.0) + gam * bfull[c]
        for bname, v in contrib.items():
            rows.append(dict(outcome=yv, component=bname, value=K(yv) * v))
        rows.append(dict(outcome=yv, component="Exceso enero-abril observado", value=K(yv) * base.coef()["pj"],
                         se=K(yv) * base.se()["pj"]))
        rows.append(dict(outcome=yv, component="Exceso que queda sin explicar", value=K(yv) * bfull["pj"],
                         se=K(yv) * full.se()["pj"]))
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "gran_janapr_gelbach.csv", index=False)
    return t, g


def enaho_monthly():
    import pyreadstat
    e = pd.read_parquet(OUT / "_cache_enaho_households.parquet")
    fr = []
    for y in sorted(e.year.unique()):
        p = ROOT / "raw" / "sumaria" / f"enaho-{int(y)}-34.dta"
        if not p.exists():
            continue
        try:
            s, _ = pyreadstat.read_dta(str(p), usecols=["conglome", "vivienda", "hogar", "mes"])
        except Exception:
            s, _ = pyreadstat.read_dta(str(p), usecols=["conglome", "vivienda", "hogar", "mes"], encoding="latin1")
        s["year"] = y
        fr.append(s)
    s = pd.concat(fr)
    for c in ["conglome", "vivienda", "hogar"]:
        s[c] = s[c].astype(str).str.strip()
        e[c] = e[c].astype(str).str.strip()
    e = e.merge(s, on=["conglome", "vivienda", "hogar", "year"], how="left")
    e["mes"] = pd.to_numeric(e.mes, errors="coerce")
    e["acq_beer"] = (e.beer > 0).astype(float)   # purchases plus other acquisitions of beer for the home
    e["puno"] = (e.dept.astype(int) == 21)
    rows = []
    for (pn, m), q in e.dropna(subset=["mes"]).groupby(["puno", "mes"]):
        w = q.factor07
        rows.append(dict(area="Puno" if pn else "Resto del Perú", month=int(m), n=len(q),
                         acquires_beer=100 * np.average(q.acq_beer, weights=w),
                         beer_per_person=(q.beer * w).sum() / (q.mieperho * w).sum()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "gran_enaho_monthly_beer.csv", index=False)
    return t


def main():
    pd.set_option("display.width", 220)
    import warnings
    warnings.filterwarnings("ignore")
    if "--season" in sys.argv:
        d, _ = prep()
        t2, g2 = season_tests(d)
        print(t2.round(3).to_string())
        print(g2.round(2).to_string())
        return
    d, val = prep()
    print("district validation:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in val.items()})
    pd.Series(val).to_csv(OUT / "gran_district_validation.csv")
    print("districts:", d.dist.nunique(), " Puno districts:", d[d.puno == 1].dist.nunique(),
          " exact dates:", round(d.exact.mean(), 3), " resp is mother:", round(d.resp_mother.mean(), 3), flush=True)
    d.to_parquet(OUT / "_granular_features.parquet")
    print(fe_ladder(d).round(2).to_string(), flush=True)
    h, ht = heterogeneity(d)
    print(h.round(3).to_string())
    print(ht.round(3).to_string(), flush=True)
    t, w, b, p = monthly(d)
    print(w.round(2).to_string())
    print(b.round(2).to_string())
    print(p.round(2).to_string(), flush=True)
    print(enaho_monthly().round(2).to_string())
    t2, g2 = season_tests(d)
    print(t2.round(3).to_string())
    print(g2.round(2).to_string())


if __name__ == "__main__":
    main()
