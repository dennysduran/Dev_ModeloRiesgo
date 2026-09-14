"""
model_training_evaluation.py
Entrena y evalúa varios modelos de clasificación para predecir Pago_atiempo.
Selecciona el mejor según performance, consistency y scalability.
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from ft_engineering import construir_dataset

def summarize_clasification(y_true, y_pred, y_prob):
    """Resume las métricas clave de un modelo de clasificación binaria"""
    return{
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
    }
    
def build_model(nombre, modelo, X_train, y_train, X_eval, y_eval):
    """Entrena un modelo, mide tiempo de entrenamiento  y devuelve sus métricas."""
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio
    
    y_pred = modelo.predict(X_eval)
    y_prob = modelo.predict_proba(X_eval)[:,1]
    
    metricas = summarize_clasification(y_eval, y_pred, y_prob)
    metricas['modelo'] = nombre
    metricas['tiempo_seg'] = round(tiempo_entrenamiento, 3)
    return modelo, metricas

def entrenar_modelos(X_train, X_eval, y_train, y_eval):
    """Entrena varios modelos y devuelve resultados comparables."""
    modelos = {
        'Regresión Logística': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Árbol de Decisión': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=200),
        'XGBoost': XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                                   random_state=42, eval_metric='logloss'),
    }

    resultados = []
    modelos_entrenados = {}

    for nombre, modelo in modelos.items():
        modelo_ajustado, metricas = build_model(nombre, modelo, X_train, y_train, X_eval, y_eval)
        resultados.append(metricas)
        modelos_entrenados[nombre] = modelo_ajustado
        print(f"{nombre} entrenado. PR-AUC: {metricas['pr_auc']:.3f} | Recall: {metricas['recall']:.3f}")

    return pd.DataFrame(resultados), modelos_entrenados

def graficar_comparacion(df_resultados):
    """Gráfico de barras comparando PR-AUC y Recall entre modelos."""
    fig, ax = plt.subplots(figsize=(8, 4))
    df_resultados.set_index('modelo')[['pr_auc', 'recall', 'roc_auc']].plot(kind='bar', ax=ax)
    plt.title('Comparación de modelos')
    plt.ylabel('Score')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


def seleccionar_mejor_modelo(df_resultados, modelos_entrenados, criterio='pr_auc'):
    """Selecciona el mejor modelo según el criterio principal (default: PR-AUC, por el desbalance)."""
    mejor_fila = df_resultados.loc[df_resultados[criterio].idxmax()]
    nombre_mejor = mejor_fila['modelo']
    return modelos_entrenados[nombre_mejor], mejor_fila

if __name__ == "__main__":
    X_train, X_eval, y_train, y_eval = construir_dataset()

    df_resultados, modelos_entrenados = entrenar_modelos(X_train, X_eval, y_train, y_eval)

    print("\n--- Tabla resumen ---")
    print(df_resultados.set_index('modelo').round(3))

    graficar_comparacion(df_resultados)

    mejor_modelo, mejor_fila = seleccionar_mejor_modelo(df_resultados, modelos_entrenados)
    print(f"\nMejor modelo: {mejor_fila['modelo']} (PR-AUC: {mejor_fila['pr_auc']:.3f})")

    # Diagnóstico: revisar qué variable está "filtrando" la respuesta
    importancias = pd.Series(
        modelos_entrenados['Árbol de Decisión'].feature_importances_,
        index=X_train.columns
    )
    print("\n--- Top 10 variables más importantes (Árbol de Decisión) ---")
    print(importancias.sort_values(ascending=False).head(10))
    
  