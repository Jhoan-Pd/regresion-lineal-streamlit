# Predicciones con Regresión Lineal Múltiple

Aplicación web (Streamlit) del Laboratorio de Minería de Datos — metodología CRISP-DM.

Permite seleccionar un ejercicio y obtener la predicción ingresando los datos por teclado:

| Ejercicio | Variable predicha | Entradas |
|---|---|---|
| Dólar | Precio_Dolar (COP) | Dia, Inflacion, Tasa_interes |
| Glucosa | Nivel_Glucosa (mg/dL) | Edad, IMC, Actividad_Fisica |
| Energía | Consumo_Energia (kWh) | Temperatura, Hora, Dia_Semana |

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
