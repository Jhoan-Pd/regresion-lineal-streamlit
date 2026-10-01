# =====================================================================
# Interfaz web — Predicciones con Regresión Lineal Múltiple
# Laboratorio de Minería de Datos (CRISP-DM · Fase 6: Despliegue)
# Ejecutar con:  streamlit run app.py
# =====================================================================
import pickle                      # Carga de los modelos serializados (.pkl)
from pathlib import Path           # Construcción de rutas independiente del sistema operativo

import pandas as pd                # Arma la fila de entrada con nombres de columnas
import streamlit as st             # Framework para crear la interfaz web

# Configuración general de la página (título de pestaña, ícono y ancho)
st.set_page_config(page_title="Predicciones · Regresión Lineal", page_icon="📈", layout="centered")

# Carpeta donde el notebook guardó los modelos exportados
CARPETA_MODELOS = Path(__file__).parent / "modelos"

# Definición de cada escenario: archivo del modelo, unidades y cómo pedir cada variable
EJERCICIOS = {
    "Dólar": {
        "archivo": "modelo_dolar.pkl",                     # Modelo exportado del ejercicio 1
        "unidad": "COP",                                   # Unidad de la predicción
        "descripcion": "Predice el **precio del dólar** a partir del día, la inflación y la tasa de interés.",
        "campos": {                                        # Configuración de cada campo de entrada
            "Dia": {"label": "Día (número)", "tipo": "int", "min": 1, "step": 1},
            "Inflacion": {"label": "Inflación diaria (ej. 0.02 = 2 %)", "tipo": "float", "min": 0.0, "step": 0.001, "fmt": "%.4f"},
            "Tasa_interes": {"label": "Tasa de interés diaria (%)", "tipo": "float", "min": 0.0, "step": 0.1, "fmt": "%.2f"},
        },
    },
    "Glucosa": {
        "archivo": "modelo_glucosa.pkl",                   # Modelo exportado del ejercicio 2
        "unidad": "mg/dL",
        "descripcion": "Predice el **nivel de glucosa en sangre** a partir de la edad, el IMC y la actividad física.",
        "campos": {
            "Edad": {"label": "Edad (años)", "tipo": "int", "min": 0, "max": 120, "step": 1},
            "IMC": {"label": "Índice de masa corporal (kg/m²)", "tipo": "float", "min": 0.0, "step": 0.1, "fmt": "%.1f"},
            "Actividad_Fisica": {"label": "Actividad física (horas/semana)", "tipo": "float", "min": 0.0, "max": 168.0, "step": 0.5, "fmt": "%.1f"},
        },
    },
    "Energía": {
        "archivo": "modelo_energia.pkl",                   # Modelo exportado del ejercicio 3
        "unidad": "kWh",
        "descripcion": "Predice el **consumo de energía eléctrica** a partir de la temperatura, la hora y el día de la semana.",
        "campos": {
            "Temperatura": {"label": "Temperatura (°C)", "tipo": "float", "step": 0.5, "fmt": "%.1f"},
            "Hora": {"label": "Hora del día (1–24)", "tipo": "int", "min": 1, "max": 24, "step": 1},
            "Dia_Semana": {"label": "Día de la semana (1=Lun … 7=Dom)", "tipo": "int", "min": 1, "max": 7, "step": 1},
        },
    },
}


@st.cache_resource                                          # Carga cada modelo una sola vez y lo reutiliza
def cargar_modelo(nombre_archivo: str) -> dict:
    """Lee el paquete (modelo + metadatos) guardado con pickle."""
    with open(CARPETA_MODELOS / nombre_archivo, "rb") as f:  # Abre el archivo en modo binario de lectura
        return pickle.load(f)                                # Deserializa y devuelve el diccionario


# ----------------------------- Encabezado -----------------------------
st.title("📈 Predicciones con Regresión Lineal Múltiple")
st.caption("Laboratorio de Minería de Datos · Metodología CRISP-DM")

# Selector del ejercicio (Dólar, Glucosa o Energía)
opcion = st.radio("Seleccione el ejercicio", list(EJERCICIOS), horizontal=True)
cfg = EJERCICIOS[opcion]                                    # Configuración del escenario elegido

# Intenta cargar el modelo; si no existe, muestra un error claro
try:
    paquete = cargar_modelo(cfg["archivo"])
except FileNotFoundError:
    st.error(f"No se encontró `modelos/{cfg['archivo']}`. Ejecute primero el notebook para exportar los modelos.")
    st.stop()

modelo = paquete["modelo"]                                  # Objeto LinearRegression entrenado
variables = paquete["variables"]                            # Orden de las variables de entrada
rangos = paquete["rangos"]                                  # Mín / media / máx observados en el dataset

st.markdown(cfg["descripcion"])

# ------------------------- Formulario de entrada -------------------------
with st.form("formulario"):                                 # El formulario evita recalcular en cada tecla
    columnas = st.columns(len(variables))                   # Una columna por variable
    valores = {}                                            # Diccionario con los valores digitados
    for col, var in zip(columnas, variables):
        c = cfg["campos"][var]                              # Configuración del campo
        r = rangos[var]                                     # Rango observado de la variable
        ayuda = f"Rango en los datos: {r['min']:.4g} – {r['max']:.4g}"
        if c["tipo"] == "int":                              # Campos enteros (día, edad, hora…)
            valores[var] = col.number_input(
                c["label"], value=int(round(r["mean"])), step=int(c["step"]),
                min_value=c.get("min"), max_value=c.get("max"), help=ayuda)
        else:                                               # Campos decimales
            valores[var] = col.number_input(
                c["label"], value=float(r["mean"]), step=float(c["step"]), format=c["fmt"],
                min_value=c.get("min"), max_value=c.get("max"), help=ayuda)
    enviar = st.form_submit_button("Predecir", type="primary", use_container_width=True)

# ------------------------------ Predicción ------------------------------
if enviar:
    entrada = pd.DataFrame([valores])[variables]            # Fila con el mismo orden de columnas del entrenamiento
    prediccion = float(modelo.predict(entrada)[0])          # Aplica el modelo y obtiene el valor estimado
    st.success(f"**{paquete['objetivo']}** estimado: **{prediccion:,.2f} {cfg['unidad']}**")

    # Advierte si algún valor está fuera del rango con que se entrenó (extrapolación)
    fuera = [v for v in variables if not (rangos[v]["min"] <= valores[v] <= rangos[v]["max"])]
    if fuera:
        st.warning("Valores fuera del rango de entrenamiento: " + ", ".join(fuera)
                   + ". La predicción es una extrapolación y puede ser menos confiable.")

# ------------------------- Información del modelo -------------------------
with st.expander("Detalles del modelo"):
    terminos = " ".join(f"{'+' if b >= 0 else '−'} {abs(b):.4f}·{v}" for v, b in zip(variables, modelo.coef_))
    st.markdown(f"**Ecuación:** `{paquete['objetivo']} = {modelo.intercept_:.4f} {terminos}`")
    m = paquete["metricas"]                                 # Métricas calculadas en el conjunto de prueba
    c1, c2, c3 = st.columns(3)
    c1.metric("R² (prueba)", f"{m['R2']:.4f}")
    c2.metric("MSE", f"{m['MSE']:,.2f}")
    c3.metric("RMSE", f"{m['RMSE']:,.2f}")
