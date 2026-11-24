"""
fig_alcohol_redes.py - social-media versions of the note's figures. Each image carries the
equation behind the number and a note with the source and how it is built. The two anemia
maps are combined into one figure with a shared color scale.
Output: analysis/alcohol_anemia/hilo/imagenes_redes/
"""
from __future__ import annotations
from pathlib import Path
import sys
import textwrap
import numpy as np
import pandas as pd
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_alcohol_nota import departments, DEP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "analysis" / "alcohol_anemia"
FIG = A / "nota" / "figuras"
OUT = A / "hilo" / "imagenes_redes"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "pdf.fonttype": 42,
                     "mathtext.fontset": "stix"})
FIRMA = "Elaboración propia con microdatos del INEI."

EQ_ANEMIA = [r"$\mathrm{Anemia}_i = 1\,[\,Hb_i - c(h_i) < u(\mathrm{edad}_i)\,]$,   $Hb$ hemoglobina medida (g/dL), $h$ altitud (m)",
             r"Regla 2001   $c(h) = -0{,}032\,a + 0{,}022\,a^2$ con $a = 3{,}3\,h/1000$,   $u = 11$",
             r"Regla 2024   $c(h) = (0{,}0056384\,h + 0{,}0000003\,h^2)/10$,   $u = 10{,}5$ si edad $< 24$ meses y $u = 11$ si no"]


def note_block(fig, eq, text, y0, wrap, size_eq=10.5, size=9.5):
    """Draw equation lines and a wrapped note, bottom-up starting at y0 (figure fraction)."""
    body = textwrap.fill(text + " " + FIRMA, wrap)
    fig.text(0.03, y0, body, ha="left", va="bottom", fontsize=size, color="#333333")
    return body.count("\n") + 1


def compose(img, eq, text, width_in=7.5, wrap=105, max_h=None):
    lines = textwrap.fill(text + " " + FIRMA, wrap).count("\n") + 1
    h_img = width_in * img.shape[0] / img.shape[1]
    if max_h:
        h_img = min(h_img, max_h)
    h_eq = 0.30 * len(eq) + (0.15 if eq else 0)
    h_txt = 0.19 * lines + 0.30
    H = h_img + h_eq + h_txt
    fig = plt.figure(figsize=(width_in, H))
    ax = fig.add_axes([0, (h_eq + h_txt) / H, 1, h_img / H])
    ax.imshow(img)
    ax.set_axis_off()
    for k, line in enumerate(eq):
        fig.text(0.03, (h_txt + h_eq - 0.30 * (k + 1)) / H, line, ha="left", va="bottom", fontsize=10.5)
    fig.text(0.03, 0.15 / H, textwrap.fill(text + " " + FIRMA, wrap), ha="left", va="bottom", fontsize=9.5, color="#333333")
    return fig


def raster(pdf_name):
    pix = fitz.open(str(FIG / f"{pdf_name}.pdf"))[0].get_pixmap(dpi=220)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]


def maps():
    geo = departments()
    t = pd.read_csv(A / "nota" / "tabla_departamentos.csv")
    inv = {v: k for k, v in DEP.items()}
    t["dept"] = t.iloc[:, 0].map(inv)
    fig, axes = plt.subplots(1, 2, figsize=(9, 6.6))
    norm = Normalize(20, 75)
    for ax, col, lab in [(axes[0], "anemia_2001", "Regla de la OMS de 2001"), (axes[1], "anemia_2024", "Regla de la OMS de 2024")]:
        m = geo.merge(t[["dept", col]], on="dept", how="left")
        m.plot(column=col, cmap="YlOrRd", norm=norm, ax=ax, edgecolor="white", linewidth=0.4)
        m[m.dept == 21].boundary.plot(ax=ax, color="black", linewidth=1.2)
        val = t.loc[t.dept == 21, col].iloc[0]
        c = m[m.dept == 21].geometry.iloc[0].centroid
        ax.annotate(f"Puno\n{val:.1f}%".replace(".", ","), xy=(c.x, c.y), xytext=(c.x + 2.2, c.y - 0.6),
                    fontsize=11, ha="left", va="center", arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
        rank = int(t[col].rank(ascending=False, method="min")[t.dept == 21].iloc[0])
        ax.set_title(f"{lab}\nPuno ocupa el puesto {rank} de 25", fontsize=13)
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd")
    cax = fig.add_axes([0.25, 0.08, 0.5, 0.025])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Niños de 6 a 35 meses con anemia (%)", fontsize=11)
    cb.outline.set_visible(False)
    fig.subplots_adjust(bottom=0.14, top=0.9, wspace=0.05)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    out = compose(img, EQ_ANEMIA,
                  "Promedio 2014-2024 de la ENDES. Un niño tiene anemia si su hemoglobina, menos la corrección por altitud, "
                  "queda por debajo del umbral. Los dos mapas usan a los mismos niños y solo cambia la regla. En la altura "
                  "la regla de 2001 resta más que la de 2024, a 3.800 m unos 3,1 g/dL frente a 2,6. El borde negro marca Puno.",
                  width_in=9, wrap=130)
    out.savefig(OUT / "01_mapas_anemia_2001_2024.png", dpi=220)
    plt.close(out)


def with_note(pdf_name, out_name, eq, note, width_in=7.5, wrap=105, max_h=None):
    fig = compose(raster(pdf_name), eq, note, width_in, wrap, max_h)
    fig.savefig(OUT / out_name, dpi=220)
    plt.close(fig)


def main():
    maps()
    with_note("mapa_cerveza_enapref", "02_mapa_cerveza_enapref.png",
              [r"$L_{pc} = 52 \times \sum_h w_h\,L_h \,/\, \sum_h w_h\,N_h$",
               r"$L_h$ litros de cerveza del hogar en 7 días, $N_h$ miembros, $w_h$ factor de expansión"],
              "Litros de cerveza adquiridos por persona al año, dentro y fuera del hogar, ENAPREF 2019-2020. Los diarios "
              "de 7 días registran las compras del hogar y de cada uno de sus miembros, incluida la cerveza en bares y "
              "fiestas. Los litros salen del volumen declarado o del tamaño del envase. La ENAPREF representa a las "
              "ciudades y no a todo el departamento. El borde negro marca Puno.", width_in=7.5, wrap=105, max_h=6.2)
    with_note("puno_vs_resto_alcohol", "03_puno_vs_resto_alcohol.png",
              [r"Cada barra es un promedio ponderado   $\bar y_g = \sum_{i \in g} w_i\,y_i \,/\, \sum_{i \in g} w_i$,   $g$ = Puno o resto del Perú",
               r"Litros en el día de mayor consumo   $0{,}25 \times \mathrm{vasos\ o\ latas} + 0{,}62 \times \mathrm{botellas}$"],
              "Fila de arriba, cuánta gente bebe y con qué frecuencia. Fila de abajo, cuánto se bebe o se compra. ENDES "
              "2014-2024 para el adulto de 15 años o más entrevistado en hogares con niños de 6 a 35 meses, y ENDES "
              "2010-2022 para las mujeres con pareja. En la ENDES la cantidad se refiere al día en que más bebió en el "
              "último mes. ENAPREF 2019-2020. ENAHO 2014-2024, que solo registra la cerveza comprada para el hogar.")
    with_note("brecha_por_criterio", "04_brecha_por_criterio.png",
              [r"$\mathrm{Diferencia} = \overline{\mathrm{Anemia}}_{Puno} - \overline{\mathrm{Anemia}}_{resto}$,   "
               r"$\mathrm{Anemia}_i = 1\,[\,Hb_i - c(h_i) < u(\mathrm{edad}_i)\,]$",
               r"Las reglas del centro combinan $c$ de una regla con $u$ de la otra"],
              "Porcentaje de niños de 6 a 35 meses con anemia en Puno y en el resto del Perú según cuatro reglas, ENDES "
              "2014-2024. La regla de 2024 cambió dos cosas respecto de la de 2001, la corrección por altitud c y el "
              "umbral u para menores de 2 años, que bajó de 11 a 10,5 g/dL. Las dos reglas del centro aplican solo uno "
              "de esos cambios. La diferencia casi desaparece cuando se cambia la corrección por altitud.")
    with_note("coeficientes_anemia", "05_coeficientes_anemia.png",
              [r"$\mathrm{Anemia}_i = \alpha + \beta\,Alc_h + \theta\,(Alc_h \times Puno_i) + X_i'\gamma + \mu_{dep} + \lambda_{año} + \varepsilon_i$",
               r"Punto azul (resto del Perú) $= \hat\beta$,   punto rojo (Puno) $= \hat\beta + \hat\theta$"],
              "Cada fila es una regresión distinta, con una medida de consumo de alcohol Alc del adulto entrevistado "
              "del hogar. X incluye edad y sexo del niño, quintil de riqueza, área, altitud, educación de la madre y "
              "edad y sexo del adulto, más efectos fijos de departamento y año. Anemia con la regla de 2024. Las "
              "líneas son el margen de error al 95%. ENDES 2014-2024. Son asociaciones, no efectos causales.")
    with_note("contrafactual", "06_contrafactual.png",
              [r"$\Delta\,\mathrm{Anemia} = -\sum_k b_k\,\overline{Alc}_k$,   $b_k = \hat\beta_k + \hat\theta_k$ con coeficientes de Puno, $b_k = \hat\beta_k$ con los del resto",
               r"$k$ = bebió en el mes, veces que bebió, litros en su día de mayor consumo, bebe con frecuencia y reporta daños"],
              "Cambio predicho en la anemia infantil, con la regla de 2024, si las cuatro medidas de consumo del adulto "
              "entrevistado pasaran a cero y todo lo demás quedara igual. Las cuatro medidas se estiman juntas en una "
              "sola regresión. Las líneas son el margen de error al 95%. ENDES 2014-2024. Es una simulación con un "
              "modelo estadístico y supone que toda la asociación es causal, por eso es un máximo.")
    with_note("violencia_por_consumo_pareja", "07_violencia_por_consumo_pareja.png",
              [r"$\%\,\mathrm{violencia}_g = 100 \times \sum_{i \in g} w_i\,V_i \,/\, \sum_{i \in g} w_i$",
               r"$V_i = 1$ si la mujer sufrió violencia física o sexual de su pareja en los últimos 12 meses"],
              "Porcentaje de mujeres con pareja que sufrieron violencia física o sexual de su pareja, según cuántas veces "
              "se embriaga la pareja, en Puno y en el resto del Perú. Las líneas son el margen de error al 95%. ENDES "
              "2010-2022, módulo de violencia doméstica. Desde 2023 la pregunta cambió y esos años no se incluyen.")
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
