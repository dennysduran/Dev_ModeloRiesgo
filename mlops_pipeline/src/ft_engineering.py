"""
ft_engineering.py
Pipeline de ingeniería de características para el modelo de riesgo crediticio.
Genera los conjuntos de entrenamiento y evaluación a partir del dataset crudo.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.api.types import CategoricalDtype

RUTA_DATOS = "../../Base_de_datos.csv"
TARGET = "Pago_atiempo"


def cargar_datos(ruta=RUTA_DATOS):
    return pd.read_csv(ruta)


def limpiar_datos(df):
    df = df.copy()

    # Unificar valores fuera de dominio -> NaN
    categorias_validas = ['Creciente', 'Estable', 'Decreciente']
    df.loc[~df['tendencia_ingresos'].isin(categorias_validas), 'tendencia_ingresos'] = np.nan
    df.loc[df['tipo_credito'] == 68, 'tipo_credito'] = np.nan

    return df


def crear_features(df):
    df = df.copy()
    # Variable derivada: presencia de mora (hallazgo clave del EDA)
    df['tiene_mora'] = (df['saldo_mora'] > 0).astype(int)
    return df


def imputar_nulos(df):
    df = df.copy()

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(TARGET, errors='ignore')
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encodear_variables(df):
    df = df.copy()

    # Ordinal: tendencia_ingresos
    orden = CategoricalDtype(categories=['Decreciente', 'Estable', 'Creciente'], ordered=True)
    df['tendencia_ingresos'] = df['tendencia_ingresos'].astype(orden).cat.codes

    # Nominales: One-Hot
    nominales = ['tipo_credito', 'tipo_laboral']
    df = pd.get_dummies(df, columns=nominales, drop_first=True)

    return df


def construir_dataset(ruta=RUTA_DATOS, test_size=0.2, random_state=42):
    df = cargar_datos(ruta)
    df = limpiar_datos(df)
    df = crear_features(df)
    df = imputar_nulos(df)
    df = encodear_variables(df)

    # Columnas irrelevantes para el modelo (fechas, ids si existieran)
    columnas_a_eliminar = ['fecha_prestamo', 'puntaje']
    df = df.drop(columns=[c for c in columnas_a_eliminar if c in df.columns])
    
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    X_train, X_eval, y_train, y_eval = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_eval, y_train, y_eval


if __name__ == "__main__":
    X_train, X_eval, y_train, y_eval = construir_dataset()
    print(f"Train: {X_train.shape} | Eval: {X_eval.shape}")
    print(f"Balance train: {y_train.value_counts(normalize=True).round(3).to_dict()}")
