# Velocidad del viento en Medellín con Naive Bayes

Proyecto final de Minería de Datos. Clasifica la velocidad del viento en Medellín en tres niveles
(bajo, medio y alto) con un modelo Naive Bayes.

**Página:** https://devharlemhm.github.io/NAIVE_BAYES_VIENTO/

## Datos

- Fuente: [Velocidad del Viento, datos.gov.co](https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Velocidad-del-Viento/sgfv-3yp8), filtrado por el municipio de Medellín.
- 1.986.848 registros originales de cuatro estaciones: Olaya Herrera, Pajarito, Villanueva y Santa Elena.
- Mediciones desde junio de 2005 hasta noviembre de 2026.

## Limpieza

| Paso | Filas que quedan | Eliminadas |
| --- | --- | --- |
| Datos originales | 1.986.848 | |
| Quitar nulos | 1.986.848 | 0 |
| Quitar duplicados exactos | 1.866.351 | 120.497 |
| Una medición por estación y momento | 1.770.694 | 95.657 |
| Quitar valores atípicos (negativos o mayores a 30 m/s) | 1.770.632 | 62 |

Además se pasaron los textos a mayúsculas (había "ANTIOQUIA" y "Antioquia") y se unificó el nombre
de cada estación con su código, porque una misma estación aparecía hasta con tres nombres.

## Modelo

- Variables: hora del día, mes y estación.
- Clases según la escala de Beaufort: BAJO (hasta 1,5 m/s), MEDIO (hasta 3,3 m/s) y ALTO (más de 3,3 m/s).
- `CategoricalNB` de scikit-learn, 70 % para entrenar y 30 % para probar.
- Precisión: 51,5 %. Si siempre se predijera BAJO, que es la clase más común, se acertaría el 44,0 %.
- El día de la semana no mejora el modelo (51,5 %), por eso no se usa.

El conjunto de datos solo trae la fecha, la estación y la velocidad. El viento también depende de la
temperatura, la presión y la humedad, que no están en estos datos, y por eso la exactitud es moderada.

## Archivos

- `index.html`: página con las gráficas y el formulario para probar el modelo.
- `modelo.js`: tablas de probabilidad que exporta el código de Python.
- `VIENTO_MEDELLIN_NAIVE_BAYES.ipynb`: código con la limpieza, las gráficas y el modelo.
