"""
analysis_fiestas.py - local festivals, drinking and child hemoglobin (event-time design).

Festival calendar: INEI (2013), Directorio Nacional de Principales Festividades a Nivel Distrital,
parsed by parse_festividades_inei.py. Days of year on a common 365-day calendar for festivals and
measurement dates (no leap-year drift); only children with an exact measurement date.
Exposures (the adult's 30-day recall window is proxied by the 30 days ending on the hemoglobin
measurement date):
  main_days   days of the district's MAIN festival (longest duration, excluding anniversaries,
              Christmas, New Year, Holy Week, Fiestas Patrias; ties: earliest) inside the window
  any_days    days of ANY festival of the district inside the window (festival intensity)
  since_any   days since the start of the most recent festival of any kind
Event bins on r = days from the start of the main festival (-182..182):
  -90..-61, -60..-31, -30..-1, 0..30, 31..60, 61..90, 91..120; reference = all other days.
  Joint Wald tests for the three pre bins and for the four post bins.
Estimation within district: district + year-month FE (and + department x calendar month);
standard errors by cluster and, as a check, by district. Puno: 15-day bins after the festival.
Descriptive; festivals also change food, income, travel and infections.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "alcohol_anemia"
FEST = ROOT / "raw" / "festividades" / "festividades_distritales_2013.csv"
BINS = [(-90, -61), (-60, -31), (-30, -1), (0, 30), (31, 60), (61, 90), (91, 120)]
PUNO_BINS = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
CTRL = "C(hc1) + C(hc27) + C(hv270) + C(hv025) + C(alt_bin) + C(meduc)"
FIT = dict(fixef_maxiter=200000)
EXCL = r"ANIVERSARIO|NAVIDAD|ANO NUEVO|AÑO NUEVO|SEMANA SANTA|CREACI|FIESTAS PATRIAS"
OUTS = ["drank_30d", "occasions", "heavy_L", "beer_heavy", "anemia_2024", "hb_adj24"]


def K(yv):
    return 100 if yv in ("drank_30d", "beer_heavy", "anemia_2024") else 1


def calendar():
    f = pd.read_csv(FEST).dropna(subset=["UBIGEO", "month", "day"]).copy()
    f["UBIGEO"] = f.UBIGEO.astype(int).astype(str).str.zfill(6)
    day = f.day.astype(int).where(f.month != 2, f.day.astype(int).clip(upper=28))
    f["doy"] = pd.to_datetime(dict(year=2013, month=f.month.astype(int), day=day), errors="coerce").dt.dayofyear
    f = f.dropna(subset=["doy"])
    f["L"] = f.days.fillna(1).clip(lower=1).astype(int)
    keep = ~f.fiesta.str.upper().str.contains(EXCL, regex=True, na=False)
    main = f[keep].sort_values(["UBIGEO", "L", "doy"], ascending=[True, False, True]).drop_duplicates("UBIGEO")
    return main[["UBIGEO", "doy", "L", "fiesta"]].rename(columns={"doy": "f_doy", "L": "f_len"}), f


def overlap(t, s, L):
    """Days of a festival starting on s and lasting L days inside the 30 days ending on t (365-day circle)."""
    t, s, L = np.asarray(t), np.asarray(s), np.asarray(L)
    return sum((((t - k - s) % 365) < L).astype(int) for k in range(30))


def rel_day(doy, fdoy):
    r = (np.asarray(doy) - np.asarray(fdoy)) % 365
    return np.where(r > 182, r - 365, r)


def add_bins(d, col, bins, prefix):
    names = []
    for lo, hi in bins:
        n = f"{prefix}{lo}_{hi}".replace("-", "m")
        d[n] = d[col].between(lo, hi).astype(float)
        names.append(n)
    return names


def wald(m, names):
    b = m.coef()[names].values
    idx = [list(m.coef().index).index(n) for n in names]
    V = m._vcov[np.ix_(idx, idx)]
    chi = float(b @ np.linalg.solve(V, b))
    return chi, float(stats.chi2.sf(chi, len(names)))


def main():
    import warnings
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 220)
    d = pd.read_parquet(OUT / "_granular_features.parquet")
    d = d[d.exact].copy()
    main_f, allf = calendar()
    byd = {u: (g.doy.values, g.L.values) for u, g in allf.groupby("UBIGEO")}
    d["has_any"] = d.dist.isin(byd.keys())
    # any-festival exposures, computed by district
    d["any_days"] = np.nan
    d["since_any"] = np.nan
    for u, idx in d[d.has_any].groupby("dist").groups.items():
        s_arr, L_arr = byd[u]
        t = d.loc[idx, "doy"].values
        d.loc[idx, "any_days"] = sum(overlap(t, s, L) for s, L in zip(s_arr, L_arr))      # festival-days (overlaps add up)
        d.loc[idx, "any_uniq"] = sum(np.any([((t - k - s) % 365) < L for s, L in zip(s_arr, L_arr)], axis=0).astype(int)
                                     for k in range(30))                                      # days with at least one festival
        d.loc[idx, "since_any"] = np.min([(t - s) % 365 for s in s_arr], axis=0)
    d = d.merge(main_f, left_on="dist", right_on="UBIGEO", how="left")
    d["has_main"] = d.f_doy.notna()
    d["r_main"] = np.where(d.has_main, rel_day(d.doy, d.f_doy.fillna(0)), np.nan)
    d["main_days"] = np.where(d.has_main, overlap(d.doy, d.f_doy.fillna(0), d.f_len.fillna(1)), np.nan)
    print(f"exact-date children {len(d):,}; with any festival {d.has_any.mean():.3f}; with main festival {d.has_main.mean():.3f}")
    dm = d[d.has_main].copy()
    bins = add_bins(dm, "r_main", BINS, "b")
    dm["dept_m"] = dm.dept.astype(str) + "_" + dm.month.astype(str)
    rows, tests = [], []
    for spec, fe in [("distrito + año-mes", "dist + ym"), ("+ departamento x mes", "dist + ym + dept_m")]:
        for yv in OUTS:
            s = dm.dropna(subset=[yv])
            for vc_lab, vc in [("conglomerado", {"CRV1": "cluster"}), ("distrito", {"CRV1": "dist"})]:
                if spec != "distrito + año-mes" and vc_lab == "distrito":
                    continue
                m = pf.feols(f"{yv} ~ {' + '.join(bins)} + {CTRL} | {fe}", data=s, weights="wt", vcov=vc, **FIT)
                for bn in bins:
                    rows.append(dict(spec=spec, cluster=vc_lab, outcome=yv, bin=bn, coef=K(yv) * m.coef()[bn],
                                     se=K(yv) * m.se()[bn], n_bin=int(s[bn].sum()), n=m._N))
                cp, pp = wald(m, bins[:3])
                cq, pq = wald(m, bins[3:])
                tests.append(dict(spec=spec, cluster=vc_lab, outcome=yv, chi2_pre=cp, p_pre=pp, chi2_post=cq, p_post=pq))
        print(f"  event bins {spec} done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "fiestas_event_bins.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "fiestas_joint_tests.csv", index=False)

    # exposure intensity: festival days inside the recall window
    rows = []
    da = d[d.has_any].copy()
    da["any_days10"] = da.any_days / 10
    da["any_uniq10"] = da.any_uniq / 10
    dm["main_days10"] = dm.main_days / 10
    for lab, sel in [("Perú", None), ("Sobre 3.000 m", "hv040 >= 3000"), ("Puno", "puno == 1")]:
        for xname, base in [("main_days10", dm), ("any_days10", da), ("any_uniq10", da)]:
            s0 = base if sel is None else base.query(sel)
            fe = "dist + ym" if lab != "Puno" else "dist + year + month"
            for yv in OUTS:
                s = s0.dropna(subset=[yv])
                m = pf.feols(f"{yv} ~ {xname} + {CTRL} | {fe}", data=s, weights="wt", vcov={"CRV1": "dist"}, **FIT)
                rows.append(dict(sample=lab, exposure=xname, outcome=yv, coef=K(yv) * m.coef()[xname],
                                 se=K(yv) * m.se()[xname], share_exposed=float((s[xname] > 0).mean()), n=m._N))
    w = pd.DataFrame(rows)
    w.to_csv(OUT / "fiestas_intensity.csv", index=False)

    # Puno, 15-day bins after the main festival
    p = dm[dm.puno == 1].copy()
    pb = add_bins(p, "r_main", PUNO_BINS, "p")
    rows = []
    for yv in OUTS:
        s = p.dropna(subset=[yv])
        m = pf.feols(f"{yv} ~ {' + '.join(pb)} + {CTRL} | dist + year + month", data=s, weights="wt",
                     vcov={"CRV1": "dist"}, **FIT)
        for bn in pb:
            rows.append(dict(outcome=yv, bin=bn, coef=K(yv) * m.coef()[bn], se=K(yv) * m.se()[bn], n_bin=int(s[bn].sum())))
    pd.DataFrame(rows).to_csv(OUT / "fiestas_puno_15day.csv", index=False)

    mf = main_f.assign(month=pd.to_datetime(main_f.f_doy - 1, unit="D", origin="2013-01-01").dt.month)
    puno_codes = set(d.loc[d.puno == 1, "dist"])
    diag = pd.DataFrame({"all": mf.month.value_counts().sort_index(),
                         "puno": mf[mf.UBIGEO.isin(puno_codes)].month.value_counts().sort_index()}).fillna(0).astype(int)
    diag.to_csv(OUT / "fiestas_main_by_month.csv")
    print(pd.DataFrame(tests).round(3).to_string())
    print(t[(t.spec == "distrito + año-mes") & (t.cluster == "conglomerado")].round(2).to_string())
    print(w.round(2).to_string())
    print(pd.DataFrame(rows).round(2).to_string())
    print(diag.T.to_string())


if __name__ == "__main__":
    main()
