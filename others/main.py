# Importar librerías
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

# Configuración de la página
st.set_page_config( page_title="Streamlit local", page_icon="📁", layout="wide" )
st.title("💻 Probar Streamlit de manera local")
st.write("""
Esta aplicación demuestra cómo cargar y explorar un conjunto de datos localmente usando Streamlit.
El conjunto de datos utilizado es un conjunto de datos ficticio sobre diabetes.
""")
st.markdown("---")

# Importar conjunto de datos desde un csv local
st.subheader("1. Cargar conjunto de datos localmente")
uploaded_file = Path(__file__).parent / "diabetes_dataset.csv"

df = pd.read_csv(uploaded_file)
st.success(" ✅ Archivo cargado correctamente.")

# Imprimir las primeras filas del conjunto de datos y las filas y columnas
st.write("-> Vista previa usando df.head()")
st.dataframe(df.head())

st.write("Filas y columnas:", df.shape)

st.markdown("---")

# Análisis del conjunto de datos
st.subheader("2. Analisis del conjunto de datos")
# Definir rangos de edad y crear columna age_group 
bins = [17, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 150]
labels = ['18–24', '25–29', '30–34', '35–39', '40–44', '45–49', 
          '50–54', '55–59', '60–64', '65–69', '70–74', '75–79', '80+']

df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

summary_age = (
    df.groupby("age_group", observed=True)["diagnosed_diabetes"]
      .agg(["count", "sum"])
      .rename(columns={"count": "total", "sum": "casos"})
      .assign(porcentaje_positivo=lambda d: d["casos"] / d["total"])
      .reset_index()
)

# Crear columna phys_activity_week: Se considera una persona que realiza actividad 
# Si tiene >= 150 minutos en el campo physical_activity_minutes_per_week (30 minutos diarios por 5 días)
df["phys_activity_week"] = df["physical_activity_minutes_per_week"].apply(lambda x: True if x > 150 else False)
activity_summary = (
    df.groupby("phys_activity_week", observed=True)["diagnosed_diabetes"]
          .mean()
          .reset_index()
          .rename(columns={"diagnosed_diabetes": "proporcion_diabetes"})
)

st.write("-> Presentación del conjunto de datos")
#Imprimir las primeras n filas, se usa slider para seleccionar n columnas de manera dinámica
n = st.slider("Número de filas", 5, 100, 5)
st.dataframe(df.head(n))

st.write("-> Dimensiones del conjunto de datos")
st.write(f"Filas: {df.shape[0]}")
st.write(f"Columnas: {df.shape[1]}")

st.write("->Tipos de datos")
st.dataframe(df.dtypes.astype(str).rename("tipo").to_frame())

st.write("-> Estadísticas descriptivas")
st.dataframe(df.describe())

st.write("-> Grupos por edad")
st.table(summary_age.head())

st.write("-> Actividad física por semana")
st.table(activity_summary.head())

st.markdown("---")

# Visualizaciones de datos
st.subheader("3. Presentación de gráficos para análisis exploratorio de datos (EDA)")

st.write("⭐ ¿Cuál es la distribución de los diferentes estados de fumador dentro de los grupos de personas con y sin diabetes?")

# SUNBURST
fig = px.sunburst(
    df,
    path=["diagnosed_diabetes", "smoking_status"],
    title="Relación jerárquica entre diabetes y hábito de fumar",
)
st.plotly_chart(fig, use_container_width=True)

# TREEMAP
fig = px.treemap(
    df,
    path=["diagnosed_diabetes", "smoking_status"],
    title="Treemap jerárquico: Diabetes → Fumador"
)
st.plotly_chart(fig, use_container_width=True)

# PIE CHARTS
col1, col2 = st.columns(2)
with col1:
    st.subheader("Diabéticos")
    diab = df[df["diagnosed_diabetes"] == 1]["smoking_status"].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5,5))
    ax1.pie(diab, labels=diab.index, autopct="%1.1f%%")
    ax1.set_title("Fumadores (Diabéticos)")
    st.pyplot(fig1)

with col2:
    st.subheader("No Diabéticos")
    nodiab = df[df["diagnosed_diabetes"] == 0]["smoking_status"].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5,5))
    ax2.pie(nodiab, labels=nodiab.index, autopct="%1.1f%%")
    ax2.set_title("Fumadores (No Diabéticos)")
    st.pyplot(fig2)


st.write("⭐ ¿Cuál es la proporción de personas con diabetes según el grupo etario?")

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=summary_age,
    x="age_group",
    y="porcentaje_positivo",
    color="#FF7789",
    ax=ax
)

ax.set_title("Proporción de personas con diabetes por grupo etario")
ax.set_xlabel("Grupo etario")
ax.set_ylabel("Prevalencia (proporción)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
st.pyplot(fig)


# PIE CHARTS

st.write("⭐ ¿Cuál es la relación entre la proporción de personas diagnosticadas con diabetes y su nivel de actividad física?")

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="diagnosed_diabetes",
    y="bmi",
    ax=ax
)

ax.set_title("Distribución de IMC según diabetes")
ax.set_xlabel("Diagnóstico de diabetes (0 = No, 1 = Sí)")
ax.set_ylabel("IMC")

st.pyplot(fig)

# COUNTPLOT
st.write("⭐ ¿Cuál es la relación entre la proporción de personas diagnosticadas con diabetes y su nivel de actividad física?")

st.write("💪 Proporción de diabetes según nivel de actividad física")

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=activity_summary,
    x="phys_activity_week",
    y="proporcion_diabetes",
    palette="coolwarm",
    ax=ax
)

ax.set_title("Proporción de diabetes según nivel de actividad física")
ax.set_xlabel("Actividad física (0 = No, 1 = Sí)")
ax.set_ylabel("Proporción con diabetes")
ax.set_ylim(0, activity_summary["proporcion_diabetes"].max() * 1.2)

st.pyplot(fig)

## Grafico #02

fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(
    data=df,
    x="diagnosed_diabetes",
    hue="smoking_status",
    ax=ax
)
ax.set_xlabel("Diagnóstico de diabetes (0 = No, 1 = Sí)")
ax.set_ylabel("Cantidad")
ax.set_title("Cantidad de fumadores según diagnóstico de diabetes")

st.pyplot(fig)


st.markdown("---")
st.header("Probar Streamlit de manera local")

st.subheader("3. Presentación de gráficos para análisis exploratorio de datos (EDA)")

st.write("⭐ ¿Cuál es la distribución de los diferentes estados de fumador dentro de los grupos de personas con y sin diabetes?")
