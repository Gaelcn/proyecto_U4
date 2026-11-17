import streamlit as st
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from pathlib import Path

st.set_page_config(page_title="Dashboard Invernadero", layout="wide")
st.title("Optimización inferencial del rendimiento agrícola")

RUTA = Path(__file__).parent / "telemetria.csv"

@st.cache_data
def cargar():
    return duckdb.sql(f"SELECT * FROM '{RUTA.as_posix()}'").df()

df = cargar()

# ---- Filtros dinámicos ----
st.sidebar.header("Filtros")
opciones = sorted(df["tipo_tratamiento"].unique())
trats = st.sidebar.multiselect("Tratamiento", opciones, default=opciones)
rmin, rmax = float(df["volumen_riego"].min()), float(df["volumen_riego"].max())
rango = st.sidebar.slider("Volumen de riego (m³/ha)", rmin, rmax, (rmin, rmax))

f = df[df["tipo_tratamiento"].isin(trats) & df["volumen_riego"].between(*rango)]
if len(f) < 10:
    st.warning("Muy pocos datos con estos filtros. Amplía la selección.")
    st.stop()

# ---- Modelo y prueba t (sobre datos filtrados) ----
X = sm.add_constant(f[["volumen_riego", "fertilizante_kg"]])
modelo = sm.OLS(f["rendimiento_ton"], X).fit()

A = f[f["tipo_tratamiento"] == "Tratamiento_A"]["rendimiento_ton"]
B = f[f["tipo_tratamiento"] == "Tratamiento_B"]["rendimiento_ton"]
if len(A) > 1 and len(B) > 1:
    _, p = stats.ttest_ind(B, A, equal_var=True, alternative="greater")
    p_txt = f"{p:.2e}"
else:
    p_txt = "N/A (requiere A y B)"

# ---- KPIs ----
c1, c2, c3 = st.columns(3)
c1.metric("Rendimiento promedio global", f"{f['rendimiento_ton'].mean():.2f} ton/ha")
c2.metric("R² ajustado del modelo", f"{modelo.rsquared_adj * 100:.1f}%")
c3.metric("P-valor de la prueba t", p_txt)

# ---- Cuadrícula 2x2 ----
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    corr = f[["volumen_riego", "fertilizante_kg", "rendimiento_ton"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Gráfica 1: Mapa de calor de correlación (Pearson)")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots()
    sns.regplot(data=f, x="volumen_riego", y="rendimiento_ton", ci=95, ax=ax,
                scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    ax.set_xlabel("Volumen de riego (m³/ha)")
    ax.set_ylabel("Rendimiento (ton/ha)")
    ax.set_title("Gráfica 2: Dispersión con línea OLS y banda de confianza")
    st.pyplot(fig)
    plt.close(fig)

col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots()
    sns.boxplot(data=f, x="tipo_tratamiento", y="rendimiento_ton", ax=ax)
    ax.set_xlabel("Tipo de tratamiento")
    ax.set_ylabel("Rendimiento (ton/ha)")
    ax.set_title("Gráfica 3: Boxplot de tratamientos")
    st.pyplot(fig)
    plt.close(fig)

with col4:
    fig, ax = plt.subplots()
    sns.histplot(modelo.resid, kde=True, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residuo (Y real - Y predicho)")
    ax.set_title("Gráfica 4: Histograma de residuos")
    st.pyplot(fig)
    plt.close(fig)