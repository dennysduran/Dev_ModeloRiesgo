"""
app.py
Aplicación streamlit para visualizar el monitoreo de data drift del modelo de riesgo crediticio.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ft_engineering import construir_dataset
from model_monitoring import(
    generar_reporte_drift,
    simular_datos_actuales,
    calcular_psi,
)

st.set_page_config(page_title="Monitoreo de Data Drift", layout="wide")
st.title("📊 Monitoreo de Data Drift — Modelo de Riesgo Crediticio")
st.markdown("""
Este panel compara la distribución de los datos usados para entrenar el modelo
(**histórico**) contra los datos de clientes recientes (**actual**), para detectar
si el modelo podría estar perdiendo precisión por cambios en la población.
""")

#-----------------Cargar Datos------------------
@st.cache_data
def cargar_datos():
    X_train, X_eval, y_train, y_eval= construir_dataset()
    return X_train, X_eval

df_historico, df_eval= cargar_datos()

num_cols= ['salario_cliente','capital_prestado','puntaje_datacredito','saldo_total',
           'cuota_pactada','edad_cliente']
cat_cols= ['tendencia_ingresos']

#------------------Sidebar: nivel de drift simulado----------------------------
st.sidebar.header("Configuración")
nivel_drift =  st.sidebar.slider(
    "Nivel de desplazamiento simulado (%)", 0, 50, 15,
    help="Simula cuánto han cambiando salario y capital prestado respecto al historico"
)

df_actual = simular_datos_actuales(df_eval)
factor= 1 + (nivel_drift/100)
df_actual['salario_cliente']  = df_eval['salario_cliente'] * factor
df_actual['capital_prestado']= df_eval['capital_prestado'] * factor

#--------------- Reporte de Drift -----------------------
reporte= generar_reporte_drift(df_historico, df_actual, num_cols, cat_cols)

#---------------Semáforo resumen-------------------------
st.header("🚦 Resumen de alertas")
col1, col2, col3 = st.columns(3)

n_estables= (reporte['alerta'] == 'Estable').sum()
n_moderadas= reporte['alerta'].str.contains('moderada|Revisar', na=False).sum()
n_criticas= (reporte['alerta'] == 'Drift significativo').sum()

col1.metric("🟢 Estables", n_estables)
col2.metric("🟡 Alerta moderada / revisar", n_moderadas)
col3.metric("🔴 Drift significativo", n_criticas)

if n_criticas >0:
    st.error (f"⚠️ Se detectó drift significativo en {n_criticas} variable(s). Se recomienda revisar el modelo antes de continuar usándolo en producción.")
    
#------------------- Tabla de métricas -----------------------
st.header("📋 Métricas de drift por variable")

def color_alerta(val):
    if val == 'Estable':
        return 'background-color: #d4edda'
    elif 'moderada' in str(val) or 'Revisar' in str(val):
        return 'background-color: #fff3cd'
    elif 'significativo' in str(val):
        return 'background-color: #f8d7da'
    return ''

st.dataframe(reporte.style.map(color_alerta,subset=['alerta']), use_container_width=True)

#---------------- Gráfico de distribución --------------------
st.header("📈 Distribución histórica vs actual")

variable_seleccionada = st.selectbox("Seleccioná una variable numérica", num_cols)

datos_hist = df_historico[variable_seleccionada].dropna()
datos_act = df_actual[variable_seleccionada].dropna()

# Recorte visual a percentiles 1-99 para que los outliers no aplasten el gráfico
limite_inf = min(datos_hist.quantile(0.01), datos_act.quantile(0.01))
limite_sup = max(datos_hist.quantile(0.99), datos_act.quantile(0.99))

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(datos_hist, bins=30, range=(limite_inf, limite_sup), alpha=0.6, label='Histórico',
        weights=np.ones(len(datos_hist)) / len(datos_hist) * 100)
ax.hist(datos_act, bins=30, range=(limite_inf, limite_sup), alpha=0.6, label='Actual',
        weights=np.ones(len(datos_act)) / len(datos_act) * 100)
ax.set_title(f'Distribución de {variable_seleccionada} (percentiles 1-99, outliers excluidos de la vista)')
ax.set_ylabel('% de clientes')
ax.legend()
st.pyplot(fig)

#----------------- Evolución temporal simulada -----------------------
st.header("⏱️ Evolución del drift en el tiempo (simulado)")

periodos= ['Mes 1', 'Mes 2', 'Mes 3', 'Mes 4 (actual)']
desplazamientos = [0.02, 0.05, 0.10, nivel_drift / 100]

evolucion = []
for periodo, despl in zip(periodos, desplazamientos):
    df_temp= df_eval.copy()
    df_temp['salario_cliente'] = df_eval['salario_cliente'] * (1 + despl)
    psi_periodo = calcular_psi(df_historico['salario_cliente'], df_temp['salario_cliente'])
    evolucion.append({'periodo':periodo, 'psi_salario_cliente': psi_periodo})
    
df_evolucion = pd.DataFrame(evolucion)

fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(df_evolucion['periodo'], df_evolucion['psi_salario_cliente'], marker='o')
ax2.axhline(0.1, color='orange', linestyle='--', label='Umbral alerta moderada')
ax2.axhline(0.25, color='red', linestyle='--', label='Umbral drift significativo')
ax2.set_ylabel('PSI (salario_cliente)')
ax2.legend()
st.pyplot(fig2)

st.caption("Nota: los datos 'actuales' y la evolucion temporal son simulados con fines demostrativos, dado que este es un dataset estático de ejmeplo (no productivo).")