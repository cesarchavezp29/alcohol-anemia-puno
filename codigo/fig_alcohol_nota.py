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


SHORT = {"Altitud y zona": "Altitud y zona rural", "Edad y sexo del niño": "Edad y sexo del niño",
         "Orden de nacimiento": "Orden de nacimiento", "Peso y tamaño al nacer": "Peso al nacer",
         "Madre: educación, edad y talla": "Educación, edad y talla de la madre",
         "Riqueza y tamaño del hogar": "Riqueza del hogar", "Agua, desagüe y combustible": "Agua, desagüe y leña",
         "Alcohol del adulto": "Alcohol del adulto", "Hemoglobina de la madre": "Hemoglobina de la madre",
         "Diarrea, fiebre y tos": "Diarrea, fiebre y tos", "Lactancia actual": "Lactancia",
         "Talla para la edad del niño": "Talla del niño para su edad"}


def predictors_figure():
    sh = pd.read_csv(A / "pred_shapley_r2.csv")
    gb = pd.read_csv(A / "pred_gelbach.csv")
    sh = sh[sh.model == "M2"]
    nat = sh[sh["sample"] == "Perú"].set_index("block").r2_pp
    pun = sh[sh["sample"] == "Puno"].set_index("block").r2_pp
    order = nat.sort_values(ascending=True).index
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 8.6), gridspec_kw=dict(height_ratios=[1, 1.05], hspace=0.42))
    y = np.arange(len(order))
    h = 0.38
    a1.barh(y + h / 2, nat[order], height=h, color=NAVY, label="Todo el Perú")
    a1.barh(y - h / 2, pun[order], height=h, color=CRAN, label="Solo Puno")
    for yy, v1, v2 in zip(y, nat[order], pun[order]):
        a1.text(v1 + 0.03, yy + h / 2, f"{v1:.2f}".replace(".", ","), va="center", fontsize=7.5, color=NAVY)
        a1.text(v2 + 0.03, yy - h / 2, f"{v2:.2f}".replace(".", ","), va="center", fontsize=7.5, color=CRAN)
    a1.set_yticks(y, [SHORT[b] for b in order], fontsize=8.5)
    a1.set_xlim(0, 2.75)
    tot_n = sh[sh["sample"] == "Perú"].r2_full_pp.iloc[0]
    tot_p = sh[sh["sample"] == "Puno"].r2_full_pp.iloc[0]
    a1.set_xlabel(f"Puntos de R², cuánto de la variación de la anemia entre niños explica cada grupo\n"
                  f"(todos juntos explican {tot_n:.1f}% en el Perú y {tot_p:.1f}% en Puno)".replace(".", ",", 2), fontsize=8.5)
    a1.legend(loc="lower right", fontsize=8.5)
    a1.set_title("A. ¿Qué predice que un niño tenga anemia?", loc="left", fontsize=10.5)

    fin = A / "pred_gelbach_final.csv"
    if fin.exists():
        gf = pd.read_csv(fin)
        t = gf[gf.comparison.str.contains("resto")].set_index("component")
    else:
        t = gb[(gb.outcome == "anemia_2024") & (gb.base == "año") & gb.comparison.str.contains("resto")].set_index("component")
    blocks = t.drop(index=["_base", "_full"]).sort_values("pp")
    labs = ["Brecha observada"] + [SHORT[b] for b in blocks.index] + ["Brecha que queda sin explicar"]
    vals = [t.loc["_base", "pp"]] + list(blocks.pp) + [t.loc["_full", "pp"]]
    lo = [t.loc["_base", "lo"]] + list(blocks.lo) + [t.loc["_full", "lo"]]
    hi = [t.loc["_base", "hi"]] + list(blocks.hi) + [t.loc["_full", "hi"]]
    cols = [GREY] + [NAVY for b in blocks.index] + [GREY]
    y2 = np.arange(len(labs))[::-1]
    a2.axvline(0, color="black", lw=0.6)
    a2.barh(y2, vals, color=cols, height=0.62)
    a2.errorbar(vals, y2, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none",
                ecolor="black", lw=0.7, capsize=2)
    for yy, v, h_, l_ in zip(y2, vals, hi, lo):
        x = (h_ + 0.25) if v >= 0 else (l_ - 0.25)
        a2.text(x, yy, (f"{v:+.1f}" if abs(v) >= 0.05 else "0,0").replace(".", ","), va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    a2.set_yticks(y2, labs, fontsize=8.5)
    for lbl in a2.get_yticklabels():
        if lbl.get_text().startswith("Brecha"):
            lbl.set_fontweight("bold")
    a2.set_xlim(-6, 13.5)
    a2.set_xlabel("Puntos porcentuales de la diferencia de anemia entre Puno y el resto del Perú, regla OMS 2024\n"
                  "(positivo: ese grupo de factores hace que Puno tenga más anemia)", fontsize=8.5)
    a2.set_title("B. ¿Qué explica que Puno tenga más anemia que el resto del Perú?", loc="left", fontsize=10.5)
    fig.savefig(F / "predictores_anemia.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    predictors_figure()
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
    names = {"drank_30d": "El adulto bebió alcohol\nen el último mes (sí o no)",
             "occasions": "Cada vez más que bebió\nen el último mes",
             "heavy_L": "Cada litro más que tomó en\nsu día de mayor consumo",
             "beer_heavy": "Tomó cerveza en su día\nde mayor consumo (sí o no)",
             "freq_problem": "Bebe 12 o más días al año\ny reporta daños por el alcohol"}
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    y = np.arange(len(rg))
    ax.axvline(0, color=GREY, lw=0.9)
    for k, (col, mk, lab, b, se, dy) in enumerate([(NAVY, "o", "Resto del Perú", rg.rest, rg.rest_se, -0.17),
                                                   (CRAN, "s", "Puno", rg.puno, rg.puno_se, 0.17)]):
        b, se = 100 * b.values, 100 * se.values
        sig = np.abs(b) > 1.96 * se
        ax.errorbar(b, y + dy, xerr=1.96 * se, fmt="none", ecolor=col, elinewidth=1.4, capsize=2.5)
        ax.scatter(b[sig], (y + dy)[sig], marker=mk, s=42, color=col, zorder=3)
        ax.scatter(b[~sig], (y + dy)[~sig], marker=mk, s=42, facecolor="white", edgecolor=col, linewidth=1.4, zorder=3)
        for yy, bb, ss in zip(y + dy, b, se):
            ax.text(bb + 1.96 * ss + 0.4, yy, f"{bb:+.1f}".replace(".", ","), ha="left", va="center", fontsize=8.5, color=col)
        ax.scatter([], [], marker=mk, s=42, color=col, label=lab)
    ax.scatter([], [], marker="o", s=42, facecolor="white", edgecolor="black", label="Marcador vacío, diferencia no clara (el margen cruza el cero)")
    ax.set_yticks(y, [names[x] for x in rg.regressor], fontsize=9)
    ax.set_xlim(-12, 18)
    ax.set_ylim(len(rg) - 0.5, -0.6)
    ax.set_xlabel("Puntos porcentuales más de anemia en sus niños frente a niños parecidos\n(criterio OMS 2024, margen de error al 95%)")
    ax.text(0, 1.0, "← menos anemia  ", transform=ax.get_xaxis_transform(), ha="right", va="bottom", fontsize=9, color=GREY)
    ax.text(0, 1.0, "  más anemia →", transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=9, color=GREY)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2, fontsize=9)
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
    lab = []
    for _, r_ in cf.iterrows():
        who = "Niños de Puno" if r_.area == "Puno" else "Niños de todo el Perú"
        coef = "lo estimado para Puno" if r_.coefficients.startswith("Puno") else "lo estimado para\nel resto del país"
        lab.append(f"{who},\nusando {coef}")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(cf))
    ax.axvline(0, color=GREY, lw=0.9)
    err = np.vstack([cf.change_pp - cf.ci_low, cf.ci_high - cf.change_pp]).astype(float)
    ax.errorbar(cf.change_pp, y, xerr=err, fmt="o", color=CRAN, ms=7, capsize=3, elinewidth=1.4)
    for yy, (_, r_) in zip(y, cf.iterrows()):
        txt = f"{r_.change_pp:.1f} puntos (de {r_.ci_low:.1f} a {r_.ci_high:+.1f})".replace(".", ",")
        ax.text(r_.change_pp, yy - 0.22, txt, ha="center", va="bottom", fontsize=8.5, color=CRAN)
    ax.set_yticks(y, lab, fontsize=9)
    ax.set_ylim(len(cf) - 0.5, -0.8)
    ax.set_xlim(-2.6, 0.6)
    ax.text(0, 1.0, "← menos anemia  ", transform=ax.get_xaxis_transform(), ha="right", va="bottom", fontsize=9, color=GREY)
    ax.set_xlabel("Cambio simulado en la anemia infantil (puntos porcentuales) si el adulto entrevistado\nde cada hogar dejara de beber, criterio OMS 2024, margen de error al 95%")
    fig.savefig(F / "contrafactual.pdf", bbox_inches="tight")
    plt.close(fig)

    # violence by partner drinking
    vd2 = A / "cons_part2_violence_desc.csv"
    if vd2.exists():
        v = pd.read_csv(vd2)
        order = ["0 no bebe", "1 bebe, nunca se embriaga", "2 se embriaga a veces", "3 se embriaga con frecuencia"]
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        w = 0.38
        shares = {}
        for k, (area, col) in enumerate([("Resto del Peru", NAVY), ("Puno", CRAN)]):
            s = v[v.area == area].set_index("categoria").reindex(order)
            shares[area] = s.share.values
            xs = np.arange(4) + (k - 0.5) * w
            ax.bar(xs, s.physsex, width=w, color=col, label=area.replace("Peru", "Perú"),
                   yerr=1.96 * s.physsex_se, error_kw=dict(ecolor="black", lw=0.8, capsize=2))
            for xx, val, se in zip(xs, s.physsex, s.physsex_se):
                ax.text(xx, val + 1.96 * se + 0.6, f"{val:.1f}".replace(".", ","), ha="center", fontsize=8)
        cats = ["La pareja\nno bebe", "Bebe, nunca\nse embriaga", "Se embriaga\na veces", "Se embriaga\ncon frecuencia"]
        labs = [f"{c}\n\nmujeres en esta\ncategoría\nresto {r:.1f}%, Puno {p:.1f}%".replace(".", ",")
                for c, r, p in zip(cats, shares["Resto del Peru"], shares["Puno"])]
        ax.set_xticks(range(4), labs, fontsize=8.5)
        ax.set_ylabel("Mujeres que sufrieron violencia física\no sexual de su pareja en el último año (%)", fontsize=9)
        ax.set_ylim(0, 39)
        ax.legend(loc="upper left", ncol=1, fontsize=9)
        fig.savefig(F / "violencia_por_consumo_pareja.pdf", bbox_inches="tight")
        plt.close(fig)
    print(tab.round(1).sort_values("anemia_2001", ascending=False).to_string())


if __name__ == "__main__":
    main()
