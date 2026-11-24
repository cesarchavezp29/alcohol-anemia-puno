"""
fig_alcohol_redes.py - social-media versions of the note's figures: each image carries a
note with the source and how the number is built. The two anemia maps are combined into a
single figure with a shared color scale. Output: analysis/alcohol_anemia/hilo/imagenes_redes/
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
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "pdf.fonttype": 42})
FIRMA = "Elaboración propia con microdatos del INEI."


def add_note(fig, text, y=0.01, width=118, size=9.5):
    fig.text(0.03, y, textwrap.fill(text + " " + FIRMA, width), ha="left", va="bottom", fontsize=size, color="#333333")


def maps():
    geo = departments()
    t = pd.read_csv(A / "nota" / "tabla_departamentos.csv")
    inv = {v: k for k, v in DEP.items()}
    t["dept"] = t.iloc[:, 0].map(inv)
    fig, axes = plt.subplots(1, 2, figsize=(9, 7.6))
    norm = Normalize(20, 75)
    for ax, col, lab in [(axes[0], "anemia_2001", "Criterio OMS 2001"), (axes[1], "anemia_2024", "Criterio OMS 2024")]:
        m = geo.merge(t[["dept", col]], on="dept", how="left")
        m.plot(column=col, cmap="YlOrRd", norm=norm, ax=ax, edgecolor="white", linewidth=0.4)
        m[m.dept == 21].boundary.plot(ax=ax, color="black", linewidth=1.2)
        val = t.loc[t.dept == 21, col].iloc[0]
        c = m[m.dept == 21].geometry.iloc[0].centroid
        ax.annotate(f"Puno\n{val:.1f}%".replace(".", ","), xy=(c.x, c.y), xytext=(c.x + 2.2, c.y - 0.6),
                    fontsize=11, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
        rank = int(t[col].rank(ascending=False, method="min")[t.dept == 21].iloc[0])
        ax.set_title(f"{lab}\nPuno ocupa el puesto {rank} de 25", fontsize=13)
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd")
    cax = fig.add_axes([0.25, 0.20, 0.5, 0.02])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Niños de 6 a 35 meses con anemia (%)", fontsize=11)
    cb.outline.set_visible(False)
    fig.subplots_adjust(bottom=0.25, top=0.9, wspace=0.05)
    add_note(fig, "Promedio 2014-2024 de la ENDES. Para cada niño tomo la hemoglobina medida, le resto la corrección "
             "por altitud de cada criterio de la OMS y la comparo con su umbral, 11 g/dL, o 10,5 g/dL para menores de "
             "2 años con el criterio de 2024. Los dos mapas usan a los mismos niños, solo cambia la regla. El borde "
             "negro marca Puno.", y=0.015, width=125)
    fig.savefig(OUT / "01_mapas_anemia_2001_2024.png", dpi=220)
    plt.close(fig)


def with_note(pdf_name, out_name, note, width_in=7.5, wrap=105):
    pix = fitz.open(str(FIG / f"{pdf_name}.pdf"))[0].get_pixmap(dpi=220)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    lines = textwrap.fill(note + " " + FIRMA, wrap).count("\n") + 1
    h_img = width_in * pix.height / pix.width
    h_txt = 0.2 * lines + 0.35
    fig = plt.figure(figsize=(width_in, h_img + h_txt))
    ax = fig.add_axes([0, h_txt / (h_img + h_txt), 1, h_img / (h_img + h_txt)])
    ax.imshow(img)
    ax.set_axis_off()
    add_note(fig, note, y=0.12 * 0.35 / (h_img + h_txt), width=wrap, size=10)
    fig.savefig(OUT / out_name, dpi=220)
    plt.close(fig)


def main():
    maps()
    with_note("mapa_cerveza_enapref", "02_mapa_cerveza_enapref.png",
              "Litros de cerveza adquiridos por persona al año, dentro y fuera del hogar. Datos de los diarios de "
              "gasto de 7 días de la ENAPREF 2019-2020, que registran las compras del hogar y de cada uno de sus "
              "miembros, incluida la cerveza en bares y fiestas. Los litros salen del volumen declarado o del tamaño "
              "del envase, y anualizo multiplicando por 52. La ENAPREF representa a las ciudades y no a todo el "
              "departamento. El borde negro marca Puno.", width_in=5.2, wrap=72)
    with_note("puno_vs_resto_alcohol", "03_puno_vs_resto_alcohol.png",
              "Fila de arriba, cuánta gente bebe y con qué frecuencia. Fila de abajo, cuánto se bebe o se compra. "
              "ENDES 2014-2024 para el adulto de 15 años o más entrevistado en hogares con niños de 6 a 35 meses, y "
              "ENDES 2010-2022 para las mujeres con pareja. En la ENDES la cantidad se refiere a la vez que más bebió "
              "en el último mes, contando 0,25 litros por vaso o lata y 0,62 por botella. ENAPREF 2019-2020. ENAHO "
              "2014-2024, que solo registra la cerveza comprada para consumir en el hogar.")
    with_note("brecha_por_criterio", "04_brecha_por_criterio.png",
              "Porcentaje de niños de 6 a 35 meses con anemia en Puno y en el resto del Perú según cuatro reglas, "
              "ENDES 2014-2024. La regla de la OMS de 2024 cambió dos cosas respecto de la de 2001, la corrección por "
              "altitud y el umbral para menores de 2 años, que bajó de 11 a 10,5 g/dL. Las dos reglas del centro "
              "aplican solo uno de esos cambios. La diferencia entre Puno y el resto casi desaparece cuando se cambia "
              "la corrección por altitud, y casi no se mueve cuando solo se cambia el umbral.")
    with_note("coeficientes_anemia", "05_coeficientes_anemia.png",
              "Diferencia en la probabilidad de anemia (criterio OMS 2024) asociada a cada medida de consumo de "
              "alcohol del adulto entrevistado, comparando niños con la misma edad, sexo, quintil de riqueza, área, "
              "altitud, educación de la madre, departamento y año. Cada medida se estima por separado. Las líneas "
              "son intervalos de confianza al 95%. ENDES 2014-2024. Son asociaciones, no efectos causales.")
    with_note("contrafactual", "06_contrafactual.png",
              "Cambio predicho en la anemia infantil (criterio OMS 2024) si el adulto entrevistado del hogar no "
              "bebiera, manteniendo igual todo lo demás. Uso los coeficientes estimados para Puno o los del resto "
              "del Perú. Las líneas son intervalos de confianza al 95%. ENDES 2014-2024. Es una simulación con un "
              "modelo estadístico, no una estimación causal.")
    with_note("violencia_por_consumo_pareja", "07_violencia_por_consumo_pareja.png",
              "Porcentaje de mujeres con pareja que sufrieron violencia física o sexual de su pareja en los últimos "
              "12 meses, según cuántas veces se embriaga la pareja, en Puno y en el resto del Perú. Las líneas son "
              "intervalos de confianza al 95%. ENDES 2010-2022, módulo de violencia doméstica. Desde 2023 la "
              "pregunta cambió y esos años no se incluyen.")
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
