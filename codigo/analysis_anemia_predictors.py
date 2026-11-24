"""
analysis_anemia_predictors.py - what predicts child anemia, and what explains the Puno gap.

Sample: ENDES 2014-2024 (2020 excluded, phone survey), children 6-35 months with measured
hemoglobin. Anemia rebuilt with the WHO 2024 rule (and the 2001 rule as a check) exactly as
in analysis_alcohol_v2.build().

Module links (verified on 2014, 2019 and 2024 files):
  RECH6  hhid, hc0 (child's line), hc60 (mother's line)
  REC21  caseid = hhid + mother's line (3 chars); b16 = child's line -> bidx, bord, b11
  REC41  caseid + midx = bidx   birth weight m19, size m18, breastfeeding m4, prenatal iron m45
  REC43  caseid + hidx = bidx   diarrhea h11, fever h22, cough h31, deworming h43
  REC95  caseid + idx95 = bidx  iron in last 7 days s465ea-ed
  RECH5  hhid + ha0 = hc60      mother's hemoglobin ha53, pregnant ha54, height ha3, age ha1
  RECH0  hv009 household size;  RECH23  hv201 water, hv205 toilet, hv226 cooking fuel
All associations are descriptive.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import endes_units as eu  # noqa: E402
from analysis_alcohol_extended import rd_sav, find_sav  # noqa: E402
from analysis_alcohol_v2 import build, one_per_hh, HARM, BEV  # noqa: E402
import pyreadstat  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "analysis" / "alcohol_anemia"
YEARS = [y for y in range(2014, 2025) if y != 2020]


def rd_case(d, need, extra=()):
    p, cl = find_sav(d, need)
    if p is None:
        return None
    df, _ = pyreadstat.read_sav(str(p), usecols=[cl[c] for c in list(need) + list(extra) if c in cl])
    df.columns = [c.lower() for c in df.columns]
    s = df["caseid"].astype(str)
    df["hhid"] = s.str[:-3].str.strip()
    df["mline"] = pd.to_numeric(s.str[-3:], errors="coerce")
    return df.drop(columns="caseid")


def load():
    fr = []
    for y in YEARS:
        d = eu.dir_for(y)
        c = rd_sav(d, ["hhid", "hc1", "hc53"], ["hc0", "hc60", "hc27", "hc61", "hc57", "hc70"])
        h = one_per_hh(rd_sav(d, ["hhid", "hv005", "hv024", "hv025", "hv001", "hv040"], ["hv008", "hv009"]), "RECH0")
        w = one_per_hh(rd_sav(d, ["hhid", "hv270"]), "RECH23")
        ws = one_per_hh(rd_sav(d, ["hhid", "hv201", "hv205"], ["hv226"]), "water")
        w = w.merge(ws, on="hhid", how="left", validate="one_to_one")
        qv = ["qs209", "qs210", "qs211c", "qs212a", "qssexo", "qs23"] + HARM + \
             [f"qs212{b}v" for b in BEV] + [f"qs212{b}b" for b in BEV]
        s = rd_sav(d, ["hhid", "qs208"], qv).drop_duplicates("hhid")
        g = c.merge(h, on="hhid", how="left", validate="many_to_one") \
             .merge(w, on="hhid", how="left", validate="many_to_one") \
             .merge(s, on="hhid", how="left", validate="many_to_one")
        g = g[eu.true_year_mask(g, y, "hv008")].copy()
        g["year"] = y
        g = g[g.hc1.between(6, 35) & g.hc53.between(30, 250)].copy()
        n0 = len(g)
        # mother
        m = rd_sav(d, ["hhid", "ha0", "ha53"], ["ha54", "ha3", "ha1"]).rename(columns={"ha0": "hc60"})
        m = m.drop_duplicates(["hhid", "hc60"])
        g = g.merge(m, on=["hhid", "hc60"], how="left", validate="many_to_one")
        # birth history link
        b = rd_case(d, ["caseid", "b16", "bidx"], ["bord", "b11"]).rename(columns={"b16": "hc0", "mline": "hc60"})
        b = b[b.hc0 > 0].drop_duplicates(["hhid", "hc60", "hc0"])
        g = g.merge(b, on=["hhid", "hc60", "hc0"], how="left", validate="many_to_one")
        for need, extra, idx in [(["caseid", "midx"], ["m19", "m18", "m4", "m45", "m14"], "midx"),
                                 (["caseid", "hidx", "h11"], ["h22", "h31", "h43"], "hidx"),
                                 (["caseid", "idx95"], ["s465ea", "s465eb", "s465ec", "s465ed"], "idx95")]:
            t = rd_case(d, need, extra)
            if t is None:
                print(f"    {y}: missing {need}")
                continue
            t = t.rename(columns={idx: "bidx", "mline": "hc60"}).drop_duplicates(["hhid", "hc60", "bidx"])
            g = g.merge(t, on=["hhid", "hc60", "bidx"], how="left", validate="many_to_one")
        # diet of the youngest child under 3 living with the mother (REC42, woman level): bidx == 1 only
        diet = ["v414e", "v414f", "v414g", "v414h", "v414h1", "v414h2", "v414i", "v414j", "v414k", "v414l",
                "v414m", "v414n", "v414o", "v414p", "v411a", "v412"]
        t = rd_case(d, ["caseid", "v414g"], diet).drop_duplicates(["hhid", "mline"])
        t = t.rename(columns={"mline": "hc60"})
        t["bidx"] = 1.0
        g = g.merge(t, on=["hhid", "hc60", "bidx"], how="left", validate="many_to_one")
        print(f"  {y}: children {n0:,}  mother Hb {g.ha53.notna().mean():.2f}  bidx {g.bidx.notna().mean():.2f}  "
              f"m19 {g.m19.notna().mean() if 'm19' in g else 0:.2f}  h11 {g.h11.notna().mean() if 'h11' in g else 0:.2f}  "
              f"s465 {g.s465ea.notna().mean() if 's465ea' in g else 0:.2f}", flush=True)
        fr.append(g)
    return pd.concat(fr, ignore_index=True)


def yes(s, yes_codes=(1, 2), no_codes=(0,)):
    return s.isin(yes_codes).astype(float).where(s.isin(list(yes_codes) + list(no_codes)))


def features(g):
    g = build(g)
    alt = g.hv040.where(g.hv040.between(0, 5500))
    # mother anemia, WHO 2024 rule (same altitude correction as the child; 12 g/dL, 11 if pregnant)
    mhb = (g.ha53 / 10).where(g.ha53.between(30, 250))
    madj = mhb - (0.0056384 * alt + 0.0000003 * alt ** 2) / 10
    preg = g.ha54 == 1
    g["m_anemia"] = (madj < np.where(preg, 11.0, 12.0)).astype(float).where(madj.notna())
    g["m_height"] = (g.ha3 / 10).where(g.ha3.between(1200, 2000))
    g["m_short"] = (g.m_height < 145).astype(float).where(g.m_height.notna())
    g["m_age"] = g.ha1.where(g.ha1.between(12, 49))
    g["m_teen"] = (g.m_age < 20).astype(float).where(g.m_age.notna())
    # child
    g["female"] = (g.hc27 == 2).astype(float)
    g["bord_c"] = g.bord.clip(upper=5)
    g["short_int"] = (g.b11 < 24).astype(float).where(g.b11.notna() | (g.bord == 1))
    g.loc[g.bord == 1, "short_int"] = 0.0
    g["lbw"] = (g.m19 < 2500).astype(float).where(g.m19.between(500, 6000))
    g["small"] = g.m18.isin([4, 5]).astype(float).where(g.m18.isin([1, 2, 3, 4, 5]))
    g["stunted"] = (g.hc70 < -200).astype(float).where(g.hc70.between(-600, 600))
    # morbidity
    g["diarrhea"] = yes(g.h11)
    g["fever"] = yes(g.h22, (1,))
    g["cough"] = yes(g.h31)
    # care / feeding
    g["breastfed_now"] = (g.m4 == 95).astype(float).where(g.m4.notna() & (g.m4 != 98))
    g["prenatal_iron"] = yes(g.m45, (1,))
    g["deworm"] = yes(g.h43, (1,))
    ir = g[[c for c in ["s465ea", "s465eb", "s465ec", "s465ed"] if c in g]]
    g["iron_7d"] = ir.eq(1).any(axis=1).astype(float).where(ir.isin([1, 2]).any(axis=1))
    # diet yesterday (WHO groups, 6-23 months, youngest child): 0 no, 1 yes, 8 dk
    def grp(cols):
        cols = [c for c in cols if c in g]
        x = g[cols]
        return x.eq(1).any(axis=1).astype(float).where(x.isin([0, 1]).any(axis=1))
    groups = {"d_grains": ["v414e", "v414f"], "d_legumes": ["v414o"], "d_dairy": ["v414p", "v411a", "v412"],
              "d_flesh": ["v414h", "v414h1", "v414h2", "v414m", "v414n"], "d_eggs": ["v414g"],
              "d_vita": ["v414i", "v414j", "v414k"], "d_other_fv": ["v414l"]}
    for k, v in groups.items():
        g[k] = grp(v)
    bm = (g.m4 == 95).astype(float).where(g.m4.notna() & (g.m4 != 98))
    dd = pd.concat([g[k] for k in groups] + [bm], axis=1)
    valid = dd.iloc[:, :-1].notna().all(axis=1)
    g["diet_div"] = (dd.fillna(0).sum(axis=1) >= 5).astype(float).where(valid)
    g["heme_food"] = g.d_flesh
    # household
    g["piped_water"] = g.hv201.isin([11, 12]).astype(float).where(g.hv201.notna())
    g["sewer"] = g.hv205.isin([11, 12]).astype(float).where(g.hv205.notna())
    g["biomass"] = g.hv226.isin([6, 7, 8, 9, 10, 11]).astype(float).where(g.hv226.notna() & (g.hv226 != 96))
    g["hhsize"] = g.hv009.where(g.hv009.between(1, 30))
    g["rural"] = (g.hv025 == 2).astype(float)
    g["alt_500"] = pd.cut(alt, [-1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 6000], labels=False)
    return g


# ------------------------------------------------------------------ design
ALC = ["drank_30d", "occasions", "heavy_L", "freq_problem"]


def dummies(s, prefix, drop_first=True):
    d = pd.get_dummies(s.astype("Int64"), prefix=prefix, dtype=float)
    return d.iloc[:, 1:] if drop_first else d


def fill_flag(d, col):
    x = d[col].astype(float)
    out = pd.DataFrame({col: x.fillna(x.mean())}, index=d.index)
    if x.isna().any():
        out[col + "_miss"] = x.isna().astype(float)
    return out


def design(d, blocks):
    parts, cols = [], {}
    for name, spec in blocks.items():
        fr = []
        for kind, col in spec:
            if kind == "cat":
                fr.append(dummies(d[col], col))
            elif kind == "ff":
                fr.append(fill_flag(d, col))
            else:
                fr.append(d[[col]].astype(float))
        b = pd.concat(fr, axis=1)
        b = b.loc[:, b.std() > 0]
        cols[name] = list(b.columns)
        parts.append(b)
    return pd.concat(parts, axis=1), cols


BLOCKS_M1 = {
    "Altitud y zona": [("cat", "alt_500"), ("num", "rural")],
    "Edad y sexo del niño": [("cat", "age6"), ("num", "female")],
    "Orden de nacimiento": [("cat", "bord_c"), ("ff", "short_int")],
    "Peso y tamaño al nacer": [("ff", "bw_kg"), ("ff", "small")],
    "Madre: educación, edad y talla": [("cat", "meduc"), ("ff", "m_age"), ("ff", "m_height")],
    "Riqueza y tamaño del hogar": [("cat", "hv270"), ("ff", "hhsize")],
    "Agua, desagüe y combustible": [("ff", "piped_water"), ("ff", "sewer"), ("ff", "biomass")],
    "Alcohol del adulto": [("num", c) for c in ALC],
}
BLOCKS_M2 = dict(BLOCKS_M1, **{
    "Hemoglobina de la madre": [("ff", "m_hb_adj")],
    "Diarrea, fiebre y tos": [("ff", "diarrhea"), ("ff", "fever"), ("ff", "cough")],
    "Lactancia actual": [("ff", "breastfed_now")],
    "Talla para la edad del niño": [("ff", "haz")],
})
BLOCKS_M3 = dict(BLOCKS_M2, **{
    "Hierro, antiparasitario y hierro prenatal": [("ff", "iron_7d"), ("ff", "deworm"), ("ff", "prenatal_iron")],
})
DIET = {"Alimentación de ayer": [("ff", "heme_food"), ("ff", "d_eggs"), ("ff", "d_legumes"), ("ff", "d_dairy"),
                                 ("ff", "diet_div")]}


def prep(g):
    g = g.copy()
    g["age6"] = pd.cut(g.hc1, [5, 11, 17, 23, 29, 35], labels=False)
    g["bw_kg"] = (g.m19 / 1000).where(g.m19.between(500, 6000))
    alt = g.hv040.where(g.hv040.between(0, 5500))
    g["m_hb_raw"] = (g.ha53 / 10).where(g.ha53.between(30, 250))
    g["m_hb_adj"] = g.m_hb_raw - (0.0056384 * alt + 0.0000003 * alt ** 2) / 10
    g["haz"] = (g.hc70 / 100).where(g.hc70.between(-600, 600))
    g["high_alt"] = (g.hv040 >= 3000).astype(float)
    return g


# ------------------------------------------------------------------ weighted algebra
def partial(M, base, w):
    """Residualize the columns of M on base (weighted)."""
    if base is None or base.shape[1] == 0:
        mu = (M * w[:, None]).sum(0) / w.sum()
        return M - mu
    sw = np.sqrt(w)[:, None]
    B = np.linalg.lstsq(base * sw, M * sw, rcond=None)[0]
    return M - base @ B


def wls(X, y, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def year_fe(d):
    return pd.get_dummies(d.year, dtype=float).values


def shapley_r2(d, blocks, y="anemia_2024"):
    X, cols = design(d, blocks)
    w = d.wt.values
    Y = partial(d[[y]].values.astype(float), year_fe(d), w)[:, 0]
    Xr = partial(X.values, year_fe(d), w)
    W = Xr * w[:, None]
    XtX, Xty, tss = Xr.T @ W, W.T @ Y, (w * Y ** 2).sum()
    tss_raw = (w * (d[y].values - np.average(d[y].values, weights=w)) ** 2).sum()
    names = list(blocks)
    idx = {n: [X.columns.get_loc(c) for c in cols[n]] for n in names}
    K = len(names)
    from math import factorial as f
    memo = {}

    def v(mask):
        if mask in memo:
            return memo[mask]
        ii = sum((idx[names[k]] for k in range(K) if mask >> k & 1), [])
        if not ii:
            memo[mask] = 0.0
            return 0.0
        A, b = XtX[np.ix_(ii, ii)], Xty[ii]
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
        memo[mask] = float(beta @ b) / tss_raw
        return memo[mask]

    phi = np.zeros(K)
    for mask in range(1 << K):
        s = bin(mask).count("1")
        for k in range(K):
            if not mask >> k & 1:
                phi[k] += f(s) * f(K - s - 1) / f(K) * (v(mask | 1 << k) - v(mask))
    full = v((1 << K) - 1)
    fe_only = 1 - tss / tss_raw
    return pd.DataFrame({"block": names, "r2_pp": 100 * phi, "share": phi / full}), 100 * full, 100 * fe_only


def cv_r2(d, blocks, y="anemia_2024", k=5, seed=1):
    X, _ = design(d, blocks)
    rng = np.random.default_rng(seed)
    cl = d.cluster.unique()
    fold = pd.Series(rng.integers(0, k, len(cl)), index=cl).reindex(d.cluster).values
    w, Y = d.wt.values, d[y].values.astype(float)
    F = year_fe(d)
    res = {"FE": 0.0, "full": 0.0}
    tot = 0.0
    for j in range(k):
        tr, te = fold != j, fold == j
        for name, M in [("FE", F), ("full", np.hstack([F, X.values]))]:
            b = wls(M[tr], Y[tr], w[tr])
            res[name] += (w[te] * (Y[te] - M[te] @ b) ** 2).sum()
        tot += (w[te] * (Y[te] - np.average(Y[tr], weights=w[tr])) ** 2).sum()
    return 100 * (1 - res["FE"] / tot), 100 * (1 - res["full"] / tot)


def gelbach(d, blocks, group, base_fn, y="anemia_2024", B=199, seed=7):
    """Gelbach decomposition of the coefficient on `group` when the blocks are added.
    base_fn(d) -> base controls (array) that stay in both models. With base = year FE and the
    full model, the explained part equals a pooled Oaxaca-Blinder with a group dummy."""
    X, cols = design(d, blocks)
    names = list(blocks)
    w0 = d.wt.values
    cl_codes, cl_idx = np.unique(d.cluster.values, return_inverse=True)
    rng = np.random.default_rng(seed)

    def once(w):
        base = base_fn(d)
        M = partial(np.column_stack([d[y].values.astype(float), d[group].values.astype(float), X.values]), base, w)
        Y, P, Xr = M[:, 0], M[:, 1], M[:, 2:]
        d_base = (w * P * Y).sum() / (w * P * P).sum()
        b = wls(np.column_stack([P, Xr]), Y, w)
        gam = (Xr * (w * P)[:, None]).sum(0) / (w * P * P).sum()
        contrib = gam * b[1:]
        out = {n: contrib[[X.columns.get_loc(c) for c in cols[n]]].sum() for n in names}
        out["_base"], out["_full"] = d_base, b[0]
        return out

    point = once(w0)
    draws = []
    for _ in range(B):
        m = np.bincount(rng.integers(0, len(cl_codes), len(cl_codes)), minlength=len(cl_codes))
        draws.append(once(w0 * m[cl_idx]))
    dr = pd.DataFrame(draws)
    rows = [dict(component=k, pp=100 * v, lo=100 * dr[k].quantile(0.025), hi=100 * dr[k].quantile(0.975))
            for k, v in point.items()]
    return pd.DataFrame(rows)


def lpm_contrasts(d, blocks, label):
    import statsmodels.api as sm
    X, cols = design(d, blocks)
    M = sm.add_constant(np.column_stack([year_fe(d)[:, 1:], X.values]))
    names = ["const"] + [f"yr{i}" for i in range(year_fe(d).shape[1] - 1)] + list(X.columns)
    m = sm.WLS(d.anemia_2024.values, M, weights=d.wt.values).fit(cov_type="cluster", cov_kwds={"groups": d.cluster.values})
    b, se = pd.Series(m.params, index=names), pd.Series(m.bse, index=names)
    want = {"Más pobre (quintil 1) frente a más rico (quintil 5)": ("-", "hv270_5"),
            "6 a 11 meses frente a 30 a 35 meses": ("-", "age6_4"),
            "Madre con 1 g/dL menos de hemoglobina": ("-", "m_hb_adj"),
            "Madre sin educación frente a superior": ("-", "meduc_3"),
            "Con desagüe en la vivienda": ("+", "sewer"),
            "Agua por red pública": ("+", "piped_water"),
            "Cocina con leña o bosta": ("+", "biomass"),
            "1 kg más de peso al nacer": ("+", "bw_kg"),
            "Diarrea en las últimas 2 semanas": ("+", "diarrhea"),
            "Fiebre en las últimas 2 semanas": ("+", "fever"),
            "1 desviación estándar más de talla para la edad": ("+", "haz"),
            "Adulto bebedor frecuente con daños": ("+", "freq_problem"),
            "Litros en el día de mayor consumo": ("+", "heavy_L"),
            "Niña frente a niño": ("+", "female")}
    rows = []
    for lab, (sg, c) in want.items():
        if c in b:
            s = -1 if sg == "-" else 1
            rows.append(dict(sample=label, contrast=lab, pp=100 * s * b[c], se=100 * se[c]))
    return pd.DataFrame(rows)


def sample_table(g):
    rows = []

    def add(lab, s):
        p, r = s[s.puno == 1], s[s.puno == 0]
        rows.append(dict(step=lab, n=len(s), n_puno=len(p), puno=100 * np.average(p.anemia_2024, weights=p.wt),
                         rest=100 * np.average(r.anemia_2024, weights=r.wt),
                         gap=100 * (np.average(p.anemia_2024, weights=p.wt) - np.average(r.anemia_2024, weights=r.wt))))
    s = g.dropna(subset=["anemia_2024"])
    add("Hemoglobina válida, 6-35 meses", s)
    s = s[s.hc60 > 0]
    add("Madre vive en el hogar", s)
    s = s[s.bidx.notna()]
    add("Enlazado a la historia de nacimientos", s)
    s = s.dropna(subset=["hv270", "meduc", "hv040"])
    add("Riqueza, educación y altitud disponibles", s)
    s = s.dropna(subset=ALC)
    add("Alcohol del adulto disponible (muestra principal)", s)
    t = pd.DataFrame(rows)
    return t, s


def mother_hb_check(d):
    import statsmodels.formula.api as smf
    rows = []
    for lab, s in [("todas las altitudes", d), ("1.000 m o menos", d[d.hv040 <= 1000]),
                   ("3.500 a 4.000 m", d[d.hv040.between(3500, 4000)])]:
        s = s.dropna(subset=["m_hb_raw", "hb_raw"])
        for yv, xv in [("hb_raw", "m_hb_raw"), ("hb_adj24", "m_hb_adj")]:
            m = smf.wls(f"{yv} ~ {xv} + C(alt_500) + C(age6) + C(year)", data=s, weights=s.wt).fit(
                cov_type="cluster", cov_kwds={"groups": s.cluster})
            rows.append(dict(sample=lab, outcome=yv, coef=m.params[xv], se=m.bse[xv], n=int(m.nobs)))
    return pd.DataFrame(rows)


def robustness():
    """Checks asked by ChatGPT (round 6)."""
    import statsmodels.formula.api as smf
    d = pd.read_parquet(OUT / "_predictors_features.parquet")
    # 1. the earlier one-at-a-time alcohol regression, on the M2 sample
    from analysis_alcohol_v2 import CTRL
    rows = []
    s = d.dropna(subset=["hc27", "hv270", "alt_bin", "meduc", "qssexo", "qs23"])
    for x in ["freq_problem", "heavy_L", "drank_30d"]:
        m = smf.wls(f"anemia_2024 ~ {x} + {x}:puno + {CTRL} + C(year) + C(dept)", data=s, weights=s.wt).fit(
            cov_type="cluster", cov_kwds={"groups": s.cluster})
        rows.append(dict(check="previous spec on M2 sample", term=x, pp=100 * m.params[x], se=100 * m.bse[x], n=int(m.nobs)))
    # 2. biomass audit
    specs = [("bruta", "biomass + C(year)"),
             ("+ altitud", "biomass + C(alt_500) + C(year)"),
             ("+ riqueza y zona", "biomass + C(alt_500) + C(hv270) + rural + C(year)"),
             ("+ departamento x año", "biomass + C(alt_500) + C(hv270) + rural + C(dept):C(year)")]
    for lab, rhs in specs:
        for yv, sub in [("anemia_2024", d), ("hb_raw", d), ("anemia_2024", d[d.hv040.between(3500, 4000)])]:
            q = sub.dropna(subset=["biomass", yv])
            m = smf.wls(f"{yv} ~ {rhs}", data=q, weights=q.wt).fit(cov_type="cluster", cov_kwds={"groups": q.cluster})
            k = 100 if yv.startswith("anemia") else 1
            rows.append(dict(check=f"biomass {lab}" + (" 3500-4000 m" if len(sub) < len(d) else ""), term=yv,
                             pp=k * m.params["biomass"], se=k * m.bse["biomass"], n=int(m.nobs)))
    pd.DataFrame(rows).to_csv(OUT / "pred_robust_checks.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)
    # 3. Shapley with department x year FE (national)
    dy = lambda s: pd.get_dummies(s.dept.astype(str) + "_" + s.year.astype(str), dtype=float).values  # noqa: E731
    global year_fe
    orig = year_fe
    year_fe = dy
    t, full, fe = shapley_r2(d, BLOCKS_M2)
    year_fe = orig
    t["r2_full_pp"], t["r2_fe_pp"] = full, fe
    t.to_csv(OUT / "pred_shapley_deptyear.csv", index=False)
    print("Shapley dept x year FE", round(full, 2), round(fe, 2))
    print(t.round(3).to_string(), flush=True)
    # 4. final bootstrap 1,999 for the main decompositions, and the 3,500-4,000 m band
    out = []
    band = d[d.hv040.between(3500, 4000)]
    hi = d[(d.puno == 1) | (d.high_alt == 1)]
    for comp, s in [("Puno frente al resto del Perú", d), ("Puno frente a otras zonas sobre 3.000 m", hi),
                    ("Puno frente a otras zonas entre 3.500 y 4.000 m", band)]:
        t = gelbach(s, BLOCKS_M2, "puno", year_fe, B=1999)
        t["comparison"], t["n"], t["n_puno"] = comp, len(s), int(s.puno.sum())
        out.append(t)
        print(comp, len(s), int(s.puno.sum()))
        print(t.round(2).to_string(), flush=True)
    pd.concat(out).to_csv(OUT / "pred_gelbach_final.csv", index=False)


def main():
    if "--robust" in sys.argv:
        robustness()
        return
    pd.set_option("display.width", 220)
    cache = OUT / "_cache_predictors.parquet"
    if cache.exists():
        raw = pd.read_parquet(cache)
    else:
        raw = load()
        raw.to_parquet(cache)
    g = prep(features(raw))
    st, d = sample_table(g)
    print(st.round(1).to_string())
    st.to_csv(OUT / "pred_sample_steps.csv", index=False)

    # Shapley of R2 and out-of-sample R2, national and Puno
    sh = []
    for lab, s in [("Perú", d), ("Puno", d[d.puno == 1])]:
        for mname, bl in [("M1", BLOCKS_M1), ("M2", BLOCKS_M2), ("M3", BLOCKS_M3)]:
            t, full, fe = shapley_r2(s, bl)
            cv_fe, cv_full = cv_r2(s, bl)
            t["sample"], t["model"], t["r2_full_pp"], t["r2_fe_pp"] = lab, mname, full, fe
            t["cv_r2_fe"], t["cv_r2_full"] = cv_fe, cv_full
            sh.append(t)
            print(f"  Shapley {lab} {mname}: R2 incremental {full:.2f}%  (year FE {fe:.2f}%)  CV {cv_fe:.2f} -> {cv_full:.2f}", flush=True)
    sh = pd.concat(sh)
    sh.to_csv(OUT / "pred_shapley_r2.csv", index=False)
    print(sh[sh.model == "M2"].round(3).to_string())

    # contrasts
    ct = pd.concat([lpm_contrasts(d, BLOCKS_M2, "Perú"), lpm_contrasts(d[d.puno == 1], BLOCKS_M2, "Puno")])
    ct.to_csv(OUT / "pred_contrasts.csv", index=False)
    print(ct.round(2).to_string())

    # Gelbach / Oaxaca of the Puno gap
    gb = []
    yfe = year_fe
    yas = lambda s: np.column_stack([year_fe(s), dummies(s.age6, "a").values, s.female.values])  # noqa: E731
    hi = d[(d.puno == 1) | (d.high_alt == 1)]
    for comp, s in [("Puno frente al resto del Perú", d), ("Puno frente a otras zonas sobre 3.000 m", hi)]:
        for outcome in ["anemia_2024", "anemia_2001"]:
            for bname, bf in [("año", yfe), ("año, edad y sexo", yas)]:
                bl = {k: v for k, v in BLOCKS_M2.items() if not (bname != "año" and k == "Edad y sexo del niño")}
                t = gelbach(s, bl, "puno", bf, y=outcome)
                t["comparison"], t["outcome"], t["base"] = comp, outcome, bname
                gb.append(t)
                print(f"  Gelbach {comp} | {outcome} | base {bname}: gap {t.set_index('component').pp['_base']:.1f} "
                      f"-> {t.set_index('component').pp['_full']:.1f}", flush=True)
    gb = pd.concat(gb)
    gb.to_csv(OUT / "pred_gelbach.csv", index=False)
    print(gb[(gb.outcome == "anemia_2024") & (gb.base == "año")].round(2).to_string())

    # diet extension, 6-23 months, youngest child
    dd = d[(d.hc1 <= 23) & (d.bidx == 1) & d.d_eggs.notna()]
    bl = dict(BLOCKS_M2, **DIET)
    t, full, fe = shapley_r2(dd, bl)
    t["sample"], t["r2_full_pp"] = "Perú 6-23 meses", full
    t.to_csv(OUT / "pred_shapley_diet.csv", index=False)
    gd = gelbach(dd, bl, "puno", year_fe)
    gd.to_csv(OUT / "pred_gelbach_diet.csv", index=False)
    print("  diet subsample", len(dd), "R2", round(full, 2))
    print(t.round(3).to_string())
    print(gd.round(2).to_string())
    print(dd.groupby("puno")[["heme_food", "d_eggs", "d_legumes", "d_dairy", "diet_div"]].mean().round(3))

    mh = mother_hb_check(d)
    mh.to_csv(OUT / "pred_mother_hb.csv", index=False)
    print(mh.round(3).to_string())
    d.to_parquet(OUT / "_predictors_features.parquet")


if __name__ == "__main__":
    main()
