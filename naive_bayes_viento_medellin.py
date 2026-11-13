# PROYECTO FINAL - MINERÍA DE DATOS
# Clasificación de la velocidad del viento en Medellín con Naive Bayes
# Fuente: datos.gov.co, conjunto "Velocidad del Viento", municipio de Medellín
# https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Velocidad-del-Viento/sgfv-3yp8

# 1. Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 2. Cargar el archivo de Excel
# Tiene dos hojas porque Excel no permite más de 1.048.576 filas por hoja.
# Los códigos se leen como texto para no perder los ceros de la izquierda.
ruta = "/content/viento_medellin.xlsx"
hoja1 = pd.read_excel(ruta, sheet_name="Medellin_1", dtype={"codigoestacion": str, "codigosensor": str})
hoja2 = pd.read_excel(ruta, sheet_name="Medellin_2", dtype={"codigoestacion": str, "codigosensor": str})
datos = pd.concat([hoja1, hoja2], ignore_index=True)
print(f"Registros cargados: {len(datos)}")

# 3. Revisar qué tan sucios están los datos
print(datos.isnull().sum())
print("Filas duplicadas:", datos.duplicated().sum())
print(datos["departamento"].value_counts())
print(datos.groupby(["codigoestacion", "nombreestacion"]).size())
print(datos["valorobservado"].describe())

# 4. Limpieza de los datos
df = datos.copy()

# 4.1 Unificar los textos (había "ANTIOQUIA" y "Antioquia", "MEDELLÍN" y "Medellín")
df["departamento"] = df["departamento"].str.strip().str.upper()
df["municipio"] = df["municipio"].str.strip().str.upper()

# 4.2 Unificar el nombre de cada estación usando su código
# (una misma estación aparecía hasta con tres nombres distintos)
nombres = {
    "0027015330": "OLAYA HERRERA",
    "0027015290": "PAJARITO",
    "0027015300": "VILLANUEVA",
    "0027010810": "SANTA ELENA"
}
df["nombreestacion"] = df["codigoestacion"].map(nombres)

# 4.3 Convertir la fecha y la velocidad al tipo de dato correcto
df["fechaobservacion"] = pd.to_datetime(df["fechaobservacion"], errors="coerce")
df["valorobservado"] = pd.to_numeric(df["valorobservado"], errors="coerce")

# 4.4 Eliminar filas sin fecha, sin velocidad o sin estación
df = df.dropna(subset=["fechaobservacion", "valorobservado", "nombreestacion"])
print(f"Después de quitar nulos: {len(df)}")

# 4.5 Eliminar registros duplicados
df = df.drop_duplicates()
print(f"Después de quitar duplicados: {len(df)}")

# 4.6 Dejar una sola medición por estación y momento
# (Olaya Herrera tiene dos sensores que a veces miden al mismo tiempo)
df = df.drop_duplicates(subset=["codigoestacion", "fechaobservacion"])
print(f"Después de quitar mediciones repetidas: {len(df)}")

# 4.7 Eliminar valores imposibles
# La velocidad no puede ser negativa y más de 30 m/s (108 km/h) sería un huracán
df = df[(df["valorobservado"] >= 0) & (df["valorobservado"] <= 30)].reset_index(drop=True)
print(f"Registros limpios: {len(df)}")

# 5. Crear las variables para el modelo a partir de la fecha y la estación
codigos = {"OLAYA HERRERA": 0, "PAJARITO": 1, "VILLANUEVA": 2, "SANTA ELENA": 3}
df["hora"] = df["fechaobservacion"].dt.hour
df["mes"] = df["fechaobservacion"].dt.month
df["estacion"] = df["nombreestacion"].map(codigos)

# 6. Crear la variable objetivo con la escala de Beaufort
# 0 = Bajo (hasta 1.5 m/s), 1 = Medio (hasta 3.3 m/s), 2 = Alto (más de 3.3 m/s)
df["nivel"] = np.where(df["valorobservado"] <= 1.5, 0, np.where(df["valorobservado"] <= 3.3, 1, 2))
clases = {0: "Bajo", 1: "Medio", 2: "Alto"}
print(df["nivel"].value_counts())

# 7. Gráficas de los datos limpios
df.groupby("hora")["valorobservado"].mean().plot(marker="o", figsize=(8, 4))
plt.title("Velocidad promedio del viento por hora")
plt.xlabel("Hora del día")
plt.ylabel("Velocidad (m/s)")
plt.grid(True)
plt.show()

df.groupby("mes")["valorobservado"].mean().plot(kind="bar", figsize=(8, 4))
plt.title("Velocidad promedio del viento por mes")
plt.xlabel("Mes")
plt.ylabel("Velocidad (m/s)")
plt.show()

df["nivel"].map(clases).value_counts().plot(kind="bar", figsize=(6, 4))
plt.title("Registros por nivel de viento")
plt.ylabel("Cantidad")
plt.show()

# 8. Preparar los datos para el modelo
features = df[["hora", "mes", "estacion"]].values
labels = df["nivel"].values

# 9. Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

# 10. Crear y entrenar el modelo Naive Bayes
# Se usa CategoricalNB porque la hora, el mes y la estación son categorías
classifier = CategoricalNB()
classifier = classifier.fit(X_train, y_train)

# 11. Evaluar el modelo
y_pred = classifier.predict(X_test)
print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.4f}")

# Comparar con responder siempre el nivel más común (Bajo)
print(f"Precisión si siempre se responde Bajo: {(y_test == 0).mean():.4f}")

# Tabla de aciertos y errores (filas: nivel real, columnas: nivel predicho)
print(pd.crosstab(pd.Series(y_test).map(clases), pd.Series(y_pred).map(clases)))

# 12. Crear datos nuevos para probar el modelo: [hora, mes, estación]
nuevos_valores = [
    [7, 8, 0],
    [15, 8, 0],
    [14, 11, 1],
    [20, 2, 3]
]

# 13. Realizar predicciones y mostrar los resultados
predicciones = classifier.predict(nuevos_valores)
probabilidades = classifier.predict_proba(nuevos_valores)
for valor, pred, prob in zip(nuevos_valores, predicciones, probabilidades):
    print(
        f"Hora: {valor[0]}, Mes: {valor[1]}, Estación: {valor[2]} "
        f"-> Nivel de viento: {clases[pred]} ({prob.max() * 100:.1f}%)"
    )

# 14. Predecir con cualquier valor que se quiera probar
def predecir_viento(hora, mes, estacion):
    pred = classifier.predict([[hora, mes, codigos[estacion]]])[0]
    print(f"Hora: {hora}, Mes: {mes}, Estación: {estacion} -> Nivel de viento: {clases[pred]}")
    return clases[pred]

predecir_viento(16, 7, "PAJARITO")
