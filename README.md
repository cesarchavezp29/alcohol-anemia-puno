# ¿Explica la cerveza la anemia infantil en Puno?

Código, nota y resultados de un análisis con microdatos del INEI sobre la relación entre el consumo de alcohol y la anemia infantil en Puno, Perú.

La nota completa está en [`nota/nota_alcohol_anemia_puno.pdf`](nota/nota_alcohol_anemia_puno.pdf).

## Resultados principales

- **Cerveza.** Según los diarios de gasto de la ENAPREF 2019-2020, en Puno se adquieren 21,2 litros de cerveza por persona al año, frente a 15,6 en el resto del Perú, y el departamento ocupa el cuarto lugar entre 25. En la ENDES, sin embargo, en Puno bebe menos gente y con menos frecuencia, aunque en mayor cantidad cuando bebe.
- **Medición de la anemia.** Con el criterio de la OMS de 2001, la anemia de los niños de 6 a 35 meses en Puno fue de 72,8% en 2014-2024. Con el criterio de 2024 fue de 44,5%. La brecha con el resto del país pasa de 30,9 a 8,0 puntos, casi toda por el cambio en la corrección por altitud.
- **Alcohol y anemia.** El consumo de alcohol medido explica prácticamente nada de la brecha (descomposición de Oaxaca-Blinder). Si los adultos entrevistados en Puno dejaran de beber, el modelo predice una reducción de 1,1 puntos (IC 95% entre -2,3 y +0,2).
- **Violencia contra la mujer.** Cuando la pareja se embriaga con frecuencia, el 30% de las mujeres sufrió violencia física o sexual en el último año, frente al 7% cuando la pareja no bebe.

Todos los resultados son asociaciones descriptivas condicionadas en características observables, no efectos causales.

## Contenido

| Carpeta | Contenido |
|---|---|
| `nota/` | Nota en PDF y su fuente en LaTeX |
| `figuras/` | Figuras en PDF para la nota, y en `figuras/redes/` versiones PNG con nota de fuente y construcción para redes sociales |
| `hilo_twitter.txt` | Hilo de nueve tuits que resume los resultados |
| `resultados/` | Tablas de resultados agregados en CSV |
| `codigo/` | Programas en Python |

## Datos

Los microdatos son públicos y se descargan del portal del INEI (https://proyectos.inei.gob.pe/microdatos/). Este repositorio no incluye microdatos. Los programas esperan esta estructura dentro de la carpeta del repositorio.

```
raw/endes/<año>_<código>/        ENDES 2010-2024 (RECH0, RECH6, RECH23, CSALUD01, REC84DV, REC0111)
raw/gastos_alimentos/            ENAHO módulo 07 (601), enaho-<año>-07.dta
raw/sumaria/                     ENAHO sumaria, enaho-<año>-34.dta
raw/miembros/                    ENAHO módulo 02, enaho-<año>-02.dta
raw/educacion/                   ENAHO módulo 03, enaho-<año>-03.dta
raw/panel/<release>_<código>/    ENAHO panel 2011-2015, 2015-2019, 2019-2023 (sumaria)
raw/enapref/2020/                ENAPREF 2019-2020 (encuesta 753), CAP 401-404, CAP 500 y SUMARIA
```

Dos insumos se indican con variables de entorno.

- `INEI_DISTRICT_SHP`, el shapefile de límites distritales del INEI, para los mapas.
- `ENDES_WOMEN_CSV`, un archivo de mujeres de la ENDES con los indicadores de violencia (`physsex_12m`, `emo_12m`, `wt_dv`, `dv_sample`, `psu`, `caseid`, `survey_year`), construido a partir del módulo de violencia doméstica.

## Cómo reproducir

```
python codigo/analysis_alcohol_extended.py      # arma las bases de ENDES y ENAHO (2010-2024)
python codigo/analysis_alcohol_v2.py            # anemia OMS 2001 y 2024, regresiones, Oaxaca, descomposición factorial
python codigo/analysis_beer_enapref.py          # litros de cerveza en la ENAPREF
python codigo/analysis_alcohol_engel.py         # curvas de Engel
python codigo/analysis_alcohol_consequences.py  # contrafactual, violencia, escolaridad, panel
python codigo/fig_alcohol_nota.py               # figuras
```

Las salidas se escriben en `analysis/alcohol_anemia/`. Requiere Python 3.11 con pandas, numpy, pyreadstat, statsmodels, patsy, geopandas, matplotlib y PyMuPDF.

## Notas de método

- La anemia se reconstruye con la hemoglobina medida (`hc53`) y la altitud del conglomerado. La corrección de 2001 es la ecuación del CDC y la de 2024 la de la guía de la OMS. Las prevalencias nacionales difieren en 0,5 puntos o menos de las de Campos-Sánchez et al. (2025), *Rev Peru Med Exp Salud Publica* 42(2).
- En la ENDES, la pregunta QS212 se refiere a la ocasión de **mayor** consumo del último mes. Las preguntas sobre daños por alcohol (QS713 a QS717) solo se hacen a quienes beben 12 o más días al año.
- La pregunta sobre embriaguez de la pareja (D114) cambió de redacción en 2023, por lo que el análisis principal de violencia usa 2010-2022.

## Autor y licencia

Carlos Chávez. El código está bajo licencia MIT. La nota, las figuras y los resultados están bajo CC BY 4.0.
