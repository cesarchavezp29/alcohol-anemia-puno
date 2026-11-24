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


def trim(img, pad=12, thr=245):
    """Cut white margins around the plot."""
    mask = (img < thr).any(axis=2)
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, img.shape[1])
    return img[r0:r1, c0:c1]


def compose(img, eq, text, width_in=7.5, wrap=105, max_h=None):
    img = trim(img)
    body = textwrap.fill(text + " " + FIRMA, wrap)
    lines = body.count("\n") + 1
    h_img = width_in * img.shape[0] / img.shape[1]
    if max_h:
        h_img = min(h_img, max_h)
    gap, h_line_eq, h_line_txt = 0.12, 0.27, 0.175
    h_eq = h_line_eq * len(eq)
    h_txt = h_line_txt * lines
    H = h_img + gap + h_eq + gap + h_txt + 0.12
    fig = plt.figure(figsize=(width_in, H))
    ax = fig.add_axes([0, 1 - h_img / H, 1, h_img / H])
    ax.imshow(img)
    ax.set_axis_off()
    y = h_img + gap                      # inches from the top
    for line in eq:
        fig.text(0.03, 1 - y / H, line, ha="left", va="top", fontsize=10.5)
        y += h_line_eq
    y += gap
    fig.text(0.03, 1 - y / H, body, ha="left", va="top", fontsize=9.5, color="#333333", linespacing=1.25)
    return fig


def raster(pdf_name):
    pix = fitz.open(str(FIG / f"{pdf_name}.pdf"))[0].get_pixmap(dpi=220)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]


def maps():
    geo = departments()
    t = pd.read_csv(A / "nota" / "tabla_departamentos.csv")
    inv = {v: k for k, v in DEP.items()}
    t["dept"] = t.iloc[:, 0].map(inv)
    fig, axes = plt.subplots(1, 2, figsize=(9, 6.6), dpi=220)
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


def beer_map():
    geo = departments()
    t = pd.read_csv(A / "nota" / "tabla_departamentos.csv")
    inv = {v: k for k, v in DEP.items()}
    t["dept"] = t.iloc[:, 0].map(inv)
    fig, ax = plt.subplots(figsize=(5.6, 7.2), dpi=220)
    norm = Normalize(0, 30)
    m = geo.merge(t[["dept", "litros"]], on="dept", how="left")
    m.plot(column="litros", cmap="Blues", norm=norm, ax=ax, edgecolor="white", linewidth=0.4)
    m[m.dept == 21].boundary.plot(ax=ax, color="black", linewidth=1.2)
    val = t.loc[t.dept == 21, "litros"].iloc[0]
    c = m[m.dept == 21].geometry.iloc[0].centroid
    ax.annotate(f"Puno\n{val:.1f} litros".replace(".", ","), xy=(c.x, c.y), xytext=(c.x + 2.2, c.y - 0.6),
                fontsize=11, ha="left", va="center", arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    rank = int(t["litros"].rank(ascending=False, method="min")[t.dept == 21].iloc[0])
    rest = pd.read_csv(A / "enapref2020_beer_liters.csv").set_index("area").loc["Peru sin Puno", "C_plus_price_nat"]
    ax.set_title(f"Cerveza adquirida por persona al año\nPuno ocupa el puesto {rank} de 25 "
                 f"(resto del Perú, {rest:.1f} litros)".replace(".", ","), fontsize=12)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap="Blues")
    cax = fig.add_axes([0.2, 0.08, 0.6, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Litros de cerveza por persona al año", fontsize=11)
    cb.outline.set_visible(False)
    fig.subplots_adjust(bottom=0.14, top=0.9)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    out = compose(img,
                  [r"$L_{pc} = 52 \times (\Sigma_h\, w_h\,L_h) \,/\, (\Sigma_h\, w_h\,N_h)$",
                   r"$L_h$ litros de cerveza del hogar en 7 días, $N_h$ miembros, $w_h$ factor de expansión"],
                  "Litros de cerveza adquiridos por persona al año, dentro y fuera del hogar, ENAPREF 2019-2020. Los "
                  "diarios de 7 días registran las compras del hogar y de cada uno de sus miembros, incluida la cerveza "
                  "en bares y fiestas. Los litros salen del volumen declarado o del tamaño del envase. La ENAPREF "
                  "representa a las ciudades y no a todo el departamento. El borde negro marca Puno.",
                  width_in=7.0, wrap=98)
    out.savefig(OUT / "02_mapa_cerveza_enapref.png", dpi=220)
    plt.close(out)


def with_note(pdf_name, out_name, eq, note, width_in=7.5, wrap=105, max_h=None):
    fig = compose(raster(pdf_name), eq, note, width_in, wrap, max_h)
    fig.savefig(OUT / out_name, dpi=220)
    plt.close(fig)


def main():
    maps()
    beer_map()
    with_note("puno_vs_resto_alcohol", "03_puno_vs_resto_alcohol.png",
              [r"Cada barra es un promedio ponderado   $\bar y_g = (\Sigma_{i \in g}\, w_i\,y_i) \,/\, (\Sigma_{i \in g}\, w_i)$,   $g$ = Puno o resto del Perú",
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
              [r"$\Delta\,\mathrm{Anemia} = -\Sigma_k\, b_k\,\overline{Alc}_k$,   $b_k = \hat\beta_k + \hat\theta_k$ con coeficientes de Puno, $b_k = \hat\beta_k$ con los del resto",
               r"$k$ = bebió en el mes, veces que bebió, litros en su día de mayor consumo, bebe con frecuencia y reporta daños"],
              "Cambio predicho en la anemia infantil, con la regla de 2024, si las cuatro medidas de consumo del adulto "
              "entrevistado pasaran a cero y todo lo demás quedara igual. Las cuatro medidas se estiman juntas en una "
              "sola regresión. Las líneas son el margen de error al 95%. ENDES 2014-2024. Es una simulación con un "
              "modelo estadístico y supone que toda la asociación es causal, por eso es un máximo.")
    with_note("violencia_por_consumo_pareja", "07_violencia_por_consumo_pareja.png",
              [r"$\%\,\mathrm{violencia}_g = 100 \times (\Sigma_{i \in g}\, w_i\,V_i) \,/\, (\Sigma_{i \in g}\, w_i)$",
               r"$V_i = 1$ si la mujer sufrió violencia física o sexual de su pareja en los últimos 12 meses"],
              "Porcentaje de mujeres con pareja que sufrieron violencia física o sexual de su pareja, según cuántas veces "
              "se embriaga la pareja, en Puno y en el resto del Perú. Las líneas son el margen de error al 95%. ENDES "
              "2010-2022, módulo de violencia doméstica. Desde 2023 la pregunta cambió y esos años no se incluyen.")
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
