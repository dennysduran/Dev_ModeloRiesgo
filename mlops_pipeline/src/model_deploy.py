"""
model_deploy.py
Expone el modelo de riesgo crediticio como una API REST usando FastAPI.
Recibe datos "crudos" de un cliente, aplica las mismas transformaciones
usadas en el entrenamiento, y retorna la predicción de Pago_atiempo.
Soporta predicción individual y por lotes (batch).
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

from ft_engineering import limpiar_datos, crear_features, imputar_nulos, encodear_variables

#----------------------- Cargar el modelo entrenado------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "modelo_riesgo_crediticio.joblib")
modelo = joblib.load(RUTA_MODELO)

#----------------------- Definir la app FastAPI ------------------------
app = FastAPI(
    title= "API de Riesgo  Crediticio",
    description= "Predice si un cliente pagará a tiempo (Pago_atiempo) según su información crediticia.",
    version= "1.0.0",
)

#----------------------- Esquema de datos de entrada (un cliente) ------------------------
class Cliente(BaseModel):
    tipo_credito:                int
    capital_prestado:               float
    plazo_meses:                    int
    edad_cliente:                   int
    tipo_laboral:                   str
    salario_cliente:                float
    total_otros_prestamos:          int
    cuota_pactada:                  float
    puntaje_datacredito:            float
    cant_creditosvigentes:          int
    huella_consulta:                int
    saldo_mora:                     float
    saldo_total:                    float
    saldo_principal:                float
    saldo_mora_codeudor:            float
    creditos_sectorFinanciero:      int
    creditos_sectorCooperativo:     int
    creditos_sectorReal:            int
    promedio_ingresos_datacredito:  float
    tendencia_ingresos:             str


class LoteClientes(BaseModel):
    clientes: List[Cliente]
    
# --------- Lógica de predicción reutilizando ft_engineering ---------
def preparar_datos(df_crudo: pd.DataFrame, columnas_modelo) -> pd.DataFrame:
    """
    Aplica el mismo pipeline de limpieza/encoding usado en el entrenamiento,
    y alinea las columnas resultantes con las que el modelo espera
    (por si el One-Hot Encoding genera categorías distintas en un lote chico).
    """
    df= limpiar_datos(df_crudo)
    df= crear_features(df)
    df= imputar_nulos(df)
    df= encodear_variables(df)
    
    # Alinear columnas: agregar las que falten (con 0) y ordenar igual que en el entrenamiento
    for col in columnas_modelo:
        if col not in df.columns:
            df[col] = 0
    df = df[columnas_modelo]

    return df

def predecir(df_crudo: pd.DataFrame) -> list:
    columnas_modelo = modelo.feature_names_in_
    df_listo= preparar_datos(df_crudo, columnas_modelo)
    
    predicciones= modelo.predict(df_listo)
    probabilidades= modelo.predict_proba(df_listo)[:, 1]
    
    resultados = []
    for pred, prob in zip(predicciones, probabilidades):
        resultados.append({
            "pago_a_tiempo_predicho": int(pred),
            "probabilidad_pago_a_tiempo": round(float(prob), 4),
        })
    return resultados

# --------- Endpoints ---------
@app.get("/")
def home():
    return {"mensaje": "API de Riesgo Crediticio activa. Usá /predict para obtener predicciones."}

@app.post("/predict")
def predict(lote: LoteClientes):
    """
    Recibe uno o varios clientes (formato batch) y devuelve la predicción de cada uno: si pagará a tiempo o no,
    y la probabilidad asociada.
    """
    df_crudo= pd.DataFrame([c.dict() for c in lote.clientes])
    resultados= predecir(df_crudo)
    return {"predicciones": resultados}
    