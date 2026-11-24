"""
fig_alcohol_nota.py - figures for the note on alcohol and child anemia in Puno.
Vector PDFs, Times, no in-image titles, legend under the axes, estimates in navy.
Output: analysis/alcohol_anemia/nota/figuras/
"""
from __future__ import annotations
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_alcohol_v2 as V2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis" / "alcohol_anemia"
F = A / "nota" / "figuras"
F.mkdir(parents=True, exist_ok=True)
SHP = Path(os.environ.get("INEI_DISTRICT_SHP", str(Path(__file__).resolve().parents[1] / "raw" / "geo" / "Limite_Distrital_INEI_2025_CPV.shp")))
NAVY, CRAN, GREY, TEAL = "#1f3b57", "#a4133c", "#8c8c8c", "#2a7f7a"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "legend.frameon": False})
DEP = {1: "Amazonas", 2: "Áncash", 3: "Apurímac", 4: "Arequipa", 5: "Ayacucho", 6: "Cajamarca", 7: "Callao",
       8: "Cusco", 9: "Huancavelica", 10: "Huánuco", 11: "Ica", 12: "Junín", 13: "La Libertad", 14: "Lambayeque",
       15: "Lima", 16: "Loreto", 17: "Madre de Dios", 18: "Moquegua", 19: "Pasco", 20: "Piura", 21: "Puno",
       22: "San Martín", 23: "Tacna", 24: "Tumbes", 25: "Ucayali"}


def departments():
    g = gpd.read_file(SHP)
    g["dept"] = g["UBIGEO"].astype(str).str.zfill(6).str[:2].astype(int)
    return g.dissolve(by="dept").reset_index()[["dept", "geometry"]]


def choropleth(geo, values, path, label, vmin, vmax, cmap="YlOrRd"):
    fig, ax = plt.subplots(figsize=(3.6, 4.6))
    m = geo.merge(values.rename("v"), left_on="dept", right_index=True, how="left")
    norm = Normalize(vmin, vmax)
    m.plot(column="v", cmap=cmap, norm=norm, ax=ax, edgecolor="white", linewidth=0.4,
           missing_kwds={"color": "#eeeeee"})
    m[m.dept == 21].boundary.plot(ax=ax, color="black", linewidth=1.1)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.8)
    cb.set_label(label)
    cb.outline.set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    geo = departments()
    g = V2.build(pd.read_parquet(A / "_cache_v2_endes.parquet"))
    g = g[g.year >= 2014]
    an = {}
    for v in ["anemia_2001", "anemia_2024"]:
        s = g.dropna(subset=[v])
        an[v] = s.groupby("dept").apply(lambda x: 100 * np.average(x[v], weights=x.wt))
    # maps of anemia, common scale
    lo, hi = 20, 75
    choropleth(geo, an["anemia_2001"], F / "mapa_anemia_2001.pdf", "Anemia, criterio OMS 2001 (%)", lo, hi)
    choropleth(geo, an["anemia_2024"], F / "mapa_anemia_2024.pdf", "Anemia, criterio OMS 2024 (%)", lo, hi)
    # map of beer liters (ENAPREF)
    e = pd.read_csv(A / "enapref2020_beer_liters.csv")
    e = e[e.area.str.startswith("dept")].copy()
    e["dept"] = e.area.str[-2:].astype(int)
    lit = e.set_index("dept").C_plus_price_nat
    choropleth(geo, lit, F / "mapa_cerveza_enapref.pdf", "Litros de cerveza por persona al año", 0, 30, cmap="Blues")
    tab = pd.DataFrame({"anemia_2001": an["anemia_2001"], "anemia_2024": an["anemia_2024"], "litros": lit})
    tab.index = tab.index.map(DEP)
    tab.round(1).to_csv(A / "nota" / "tabla_departamentos.csv")

    # Puno vs rest of Peru: actual values, one small panel per measure
    a = g.copy()

    def wavg(x, v):
        x = x.dropna(subset=[v])
        return np.average(x[v], weights=x.wt)
    panels = []
    s = a.dropna(subset=["drank_30d"])
    panels.append(("Adultos que bebieron\nen el último mes (%)\nENDES", 100 * wavg(s[s.puno == 0], "drank_30d"), 100 * wavg(s[s.puno == 1], "drank_30d"), "{:.0f}"))
    s = a[a.drank_30d == 1]
    panels.append(("Veces que bebió en el mes,\nentre quienes bebieron\nENDES", wavg(s[s.puno == 0], "occasions"), wavg(s[s.puno == 1], "occasions"), "{:.1f}"))
    vd = pd.read_csv(A / "cons_part2_violence_desc.csv")
    sh = vd[vd.categoria.str.startswith(("2", "3"))].groupby("area").share.sum()
    panels.append(("Mujeres cuya pareja se embriaga\na veces o con frecuencia (%)\nENDES", sh["Resto del Peru"], sh["Puno"], "{:.0f}"))
    av = a.qs212av.where(a.qs212av.between(0, 97)).fillna(0)
    ab = a.qs212ab.where(a.qs212ab.between(0, 97)).fillna(0)
    a["beer_L_heavy"] = (av * 0.25 + ab * 0.62).where(a.beer_heavy == 1)
    s = a[a.beer_heavy == 1]
    panels.append(("Litros de cerveza en la ocasión\nde mayor consumo del mes,\nentre quienes tomaron cerveza\nENDES", wavg(s[s.puno == 0], "beer_L_heavy"), wavg(s[s.puno == 1], "beer_L_heavy"), "{:.1f}"))
    lp = e.set_index("dept").C_plus_price_nat
    rest = pd.read_csv(A / "enapref2020_beer_liters.csv").set_index("area")
    panels.append(("Litros de cerveza adquiridos\npor persona al año,\ndentro y fuera de casa\nENAPREF", rest.loc["Peru sin Puno", "C_plus_price_nat"], lp[21], "{:.1f}"))
    ena = pd.read_csv(A / "ext_enaho_beer_dept_year.csv")
    ena = ena[ena.year >= 2014]
    panels.append(("Soles gastados en cerveza\npara el hogar por persona\nal año\nENAHO", ena[ena.dept != 21].beer_pc.mean(), ena[ena.dept == 21].beer_pc.mean(), "{:.1f}"))
    pd.DataFrame(panels, columns=["medida", "resto", "puno", "fmt"]).drop(columns="fmt").round(2).to_csv(A / "nota" / "tabla_puno_vs_resto.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 5.2))
    for ax, (lab, vr, vp, fmt) in zip(axes.ravel(), panels):
        ax.bar([0, 1], [vr, vp], color=[NAVY, CRAN], width=0.6)
        top = max(vr, vp)
        for xx, val in zip([0, 1], [vr, vp]):
            ax.text(xx, val + 0.03 * top, fmt.format(val).replace(".", ","), ha="center", fontsize=9)
        ax.set_xticks([0, 1], ["Resto\ndel Perú", "Puno"], fontsize=9)
        ax.set_ylim(0, top * 1.18)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(lab, fontsize=8.5)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(F / "puno_vs_resto_alcohol.pdf", bbox_inches="tight")
    plt.close(fig)
    print(pd.DataFrame(panels).iloc[:, :3].round(2).to_string())

    # factorial decomposition
    fs = pd.read_csv(A / "v2_factorial_shapley.csv")
    fs = fs[(fs["sample"] == "2014-2024") & fs.definition.isin(["A", "B", "C", "D"])]
    fs = fs.set_index("definition").loc[["A", "B", "C", "D"]]
    labs = ["Regla de 2001", "Regla de 2001\ncon el nuevo\numbral por edad", "Regla de 2001\ncon la nueva\ncorrección por altitud", "Regla de 2024"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(4)
    w = 0.36
    ax.bar(x - w / 2, fs.rest, width=w, color=NAVY, label="Resto del Perú")
    ax.bar(x + w / 2, fs.puno, width=w, color=CRAN, label="Puno")
    for i, (r_, p_) in enumerate(zip(fs.rest, fs.puno)):
        ax.text(i - w / 2, r_ + 1, f"{r_:.1f}".replace(".", ","), ha="center", fontsize=8.5)
        ax.text(i + w / 2, p_ + 1, f"{p_:.1f}".replace(".", ","), ha="center", fontsize=8.5)
        top = max(r_, p_) + 9
        ax.plot([i - w / 2, i - w / 2, i + w / 2, i + w / 2], [top - 1.5, top, top, top - 1.5], color="black", lw=0.7)
        ax.text(i, top + 1, f"diferencia\n{p_ - r_:.1f} pp".replace(".", ","), ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x, labs, fontsize=9)
    ax.set_ylabel("Niños de 6 a 35 meses\ncon anemia (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.savefig(F / "brecha_por_criterio.pdf", bbox_inches="tight")
    plt.close(fig)

    # coefficients, anemia 2024, rest vs Puno
    rg = pd.read_csv(A / "v2_regressions_puno.csv")
    rg = rg[(rg.outcome == "anemia_2024") & (rg.years == "2014-2024")]
    names = {"drank_30d": "Bebió en 30 días", "occasions": "Una ocasión más en 30 días",
             "heavy_L": "Un litro más en la ocasión máxima", "beer_heavy": "Cerveza en la ocasión máxima",
             "freq_problem": "Bebedor frecuente con daños"}
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    y = np.arange(len(rg))
    ax.axvline(0, color=GREY, lw=0.8)
    ax.errorbar(100 * rg.rest, y + 0.15, xerr=196 * rg.rest_se, fmt="o", color=NAVY, capsize=2, label="Resto del Perú")
    ax.errorbar(100 * rg.puno, y - 0.15, xerr=196 * rg.puno_se, fmt="s", color=CRAN, capsize=2, label="Puno")
    ax.set_yticks(y, [names[x] for x in rg.regressor])
    ax.invert_yaxis()
    ax.set_xlabel("Asociación con la probabilidad de anemia (pp), criterio OMS 2024, IC 95%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.savefig(F / "coeficientes_anemia.pdf", bbox_inches="tight")
    plt.close(fig)

    # counterfactual
    cf = pd.read_csv(A / "cons_part1_counterfactual.csv")
    cf = cf[(cf.outcome == "anemia_2024") & cf.part.str.startswith("1a")].reset_index(drop=True)
    lab = []
    for _, r_ in cf.iterrows():
        if r_.part.startswith("1a"):
            lab.append(f"{r_.area.replace('Peru', 'Perú')}, el adulto entrevistado no bebe\n(coeficientes {'de Puno' if r_.coefficients.startswith('Puno') else 'del resto del Perú'})")
        else:
            lab.append(f"{r_.area.replace('Peru', 'Perú')}, el gasto en alcohol\npasa a gasto del hogar")
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    y = np.arange(len(cf))
    ax.axvline(0, color=GREY, lw=0.8)
    err = np.vstack([cf.change_pp - cf.ci_low, cf.ci_high - cf.change_pp]).astype(float)
    err = np.nan_to_num(err)
    ax.errorbar(cf.change_pp, y, xerr=err, fmt="o", color=NAVY, capsize=2)
    ax.set_yticks(y, lab)
    ax.invert_yaxis()
    ax.set_xlabel("Cambio simulado en la anemia infantil (pp), criterio OMS 2024, IC 95%")
    fig.savefig(F / "contrafactual.pdf", bbox_inches="tight")
    plt.close(fig)

    # violence by partner drinking
    vd2 = A / "cons_part2_violence_desc.csv"
    if vd2.exists():
        v = pd.read_csv(vd2)
        order = ["0 no bebe", "1 bebe, nunca se embriaga", "2 se embriaga a veces", "3 se embriaga con frecuencia"]
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        w = 0.38
        for k, (area, col) in enumerate([("Resto del Peru", NAVY), ("Puno", CRAN)]):
            s = v[v.area == area].set_index("categoria").reindex(order)
            xs = np.arange(4) + (k - 0.5) * w
            ax.bar(xs, s.physsex, width=w, color=col, label=area.replace("Peru", "Perú"),
                   yerr=1.96 * s.physsex_se, error_kw=dict(ecolor="black", lw=0.8, capsize=2))
            for xx, val, se in zip(xs, s.physsex, s.physsex_se):
                ax.text(xx, val + 1.96 * se + 0.6, f"{val:.0f}", ha="center", fontsize=8)
        ax.set_xticks(range(4), ["La pareja\nno bebe", "Bebe, nunca\nse embriaga", "Se embriaga\na veces", "Se embriaga\ncon frecuencia"])
        ax.set_ylabel("Violencia física o sexual\nen los últimos 12 meses (%)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        fig.savefig(F / "violencia_por_consumo_pareja.pdf", bbox_inches="tight")
        plt.close(fig)
    print(tab.round(1).sort_values("anemia_2001", ascending=False).to_string())


if __name__ == "__main__":
    main()
