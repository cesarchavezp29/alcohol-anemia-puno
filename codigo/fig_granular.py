"""fig_granular.py - reply to a reader: Puno seasonality, the Candelaria window and local festivals."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = Path(__file__).resolve().parents[1] / "analysis" / "alcohol_anemia"
NAVY, CRAN, GREY = "#1f3b57", "#a4133c", "#8c8c8c"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def pp(v, nd=1):
    return f"{v:+.{nd}f}".replace(".", ",")


b = pd.read_csv(A / "gran_puno_bimonthly.csv")
c = pd.read_csv(A / "gran_candelaria_tests.csv")
e = pd.read_csv(A / "fiestas_event_bins.csv")
e = e[(e.spec == "distrito + año-mes") & (e.cluster == "distrito")]
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.2, 11.2), gridspec_kw=dict(hspace=0.48, height_ratios=[1.1, 0.75, 1]))

# A. seasonality
labs = ["ene-feb", "mar-abr", "may-jun", "jul-ago", "sep-oct", "nov-dic"]
x = np.arange(6)
for k, (yv, col, lab) in enumerate([("anemia_2024", CRAN, "Anemia infantil"),
                                    ("beer_heavy", NAVY, "Adulto tomó cerveza en su día de mayor consumo")]):
    s = b[b.outcome == yv].set_index("window").reindex(labs)
    xx = x + (k - 0.5) * 0.18
    ref = s.index == "jul-ago"
    a1.errorbar(xx[~ref], s.coef[~ref], yerr=1.96 * s.se[~ref], fmt="o", color=col, capsize=3, label=lab)
    a1.scatter(xx[ref], s.coef[ref], marker="D", facecolor="white", edgecolor=col, zorder=3)
a1.axhline(0, color=GREY, lw=0.8)
a1.axvspan(-0.4, 1.4, color="#f2e6ea", zorder=0)
a1.text(0.5, 27, "Temporada de lluvias;\nCandelaria en febrero", ha="center", va="top", fontsize=8.5, color=CRAN)
a1.text(3, -4, "referencia", ha="center", fontsize=8, color=GREY)
a1.set_xticks(x, labs)
a1.set_ylim(-12, 30)
a1.set_ylabel("Diferencia Puno menos otras zonas\naltas, respecto de jul-ago (pp)")
a1.legend(loc="upper right", fontsize=8.5)
a1.set_title("A. El exceso de anemia de Puno se concentra entre enero y abril", loc="left", fontsize=11)

# B. Candelaria window
cc = c[c.test == "Puno vs zonas altas"]
rows = [("beer_heavy", "Cerveza en su día\nde mayor consumo", NAVY), ("anemia_2024", "Anemia infantil", CRAN)]
ylab, ypos = [], []
y0 = 0
for yv, lab, col in rows:
    for j, (spec, mk, name) in enumerate([("año-mes", "o", "comparando con otras zonas altas el mismo mes"),
                                          ("+ Puno x mes calendario", "s", "y además con Puno en otros años, mismo mes")]):
        r = cc[(cc.outcome == yv) & (cc.spec == spec)].iloc[0]
        yy = y0 - j * 0.35
        a2.errorbar(r.coef, yy, xerr=1.96 * r.se, fmt=mk, color=col, capsize=3, mfc=col if j == 0 else "white")
        a2.text(r.coef + 1.96 * r.se + 1, yy, pp(r.coef), va="center", fontsize=8, color=col)
    ylab.append(lab)
    ypos.append(y0 - 0.175)
    y0 -= 1.2
a2.axvline(0, color=GREY, lw=0.8)
a2.set_yticks(ypos, ylab, fontsize=9)
a2.set_ylim(y0 + 0.6, 0.9)
a2.set_xlim(-25, 52)
a2.set_xlabel("Asociación con una exposición de 15 días a la Candelaria (puntos porcentuales)")
from matplotlib.lines import Line2D
a2.legend(handles=[Line2D([], [], marker="o", ls="", color="black", label="Efectos fijos de distrito y año-mes"),
                   Line2D([], [], marker="s", ls="", mfc="white", mec="black", label="Además, Puno x mes calendario")],
          loc="upper right", fontsize=8)
a2.set_title("B. La asociación con la cerveza persiste; la de la anemia se atenúa", loc="left", fontsize=11)

# C. local festivals, whole country
order = ["bm90_m61", "bm60_m31", "bm30_m1", "b0_30", "b31_60", "b61_90", "b91_120"]
mid = np.array([-75, -45, -15, 15, 45, 75, 105])
for k, (yv, col, lab) in enumerate([("anemia_2024", CRAN, "Anemia infantil"), ("drank_30d", NAVY, "Adulto bebió en el último mes")]):
    s = e[e.outcome == yv].set_index("bin").reindex(order)
    a3.errorbar(mid + (k - 0.5) * 5, s.coef, yerr=1.96 * s.se, fmt="o", color=col, capsize=3, label=lab)
a3.axhline(0, color=GREY, lw=0.8)
a3.axvline(0, color="black", lw=0.8, ls=":")
a3.text(2, 4.3, "inicio de la fiesta\nprincipal del distrito", fontsize=8.5, va="top")
a3.set_ylim(-5, 5)
a3.set_xlabel("Días entre la fiesta principal del distrito y la medición de la hemoglobina")
a3.set_ylabel("Puntos frente a niños del mismo\ndistrito (referencia: más de 90 días\nantes o 120 después)")
a3.legend(loc="lower left", fontsize=8.5)
a3.set_title("C. Las fiestas locales no muestran un aumento sistemático del alcohol ni de la anemia", loc="left", fontsize=11)

n_b = int(e[e.outcome == "anemia_2024"].n.iloc[0])
fig.text(0.01, 0.0, "ENDES 2014-2024 sin 2020, niños de 6 a 35 meses, regla OMS 2024, fecha exacta de medición, IC al 95% (errores agrupados por conglomerado en A y por distrito en B y C).\n"
         "A y B: Puno frente a zonas sobre 3.000 m de otros departamentos, efectos fijos de distrito, año-mes, edad y quintil. C: "
         + f"{n_b:,}".replace(",", ".") + " niños de todo el Perú,\n"
         "efectos fijos de distrito y año-mes. Fiestas del Directorio Nacional de Festividades a Nivel Distrital del INEI (2013). Asociaciones, no efectos causales.",
         fontsize=7.3, color="#444444")
fig.savefig(A / "nota" / "figuras" / "granularidad.pdf", bbox_inches="tight")
fig.savefig(A / "hilo" / "imagenes_redes" / "10_granularidad_respuesta.png", dpi=220, bbox_inches="tight")
