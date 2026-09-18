"""
model_monitoring.py
Monitoreo de data drift: compara la distribución de datos históricos
(usados para entrenar) contra datos actuales (nuevos clientes),
calcula métricas de drift y genera alertas.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
from ft_engineering import construir_dataset

#---------------------- Métricas de Drift-------------------------

def calcular_psi(hist, actual, buckets=10):
    """Population Stability Index para una variable numérica."""
    hist= hist.dropna()
    actual= actual.dropna()
    
    puntos_corte     = np.percentile(hist, np.linspace(0, 100, buckets + 1))
    puntos_corte[0]  -= 1e-9 
    puntos_corte[-1] += 1e-9
    
    hist_bins   = pd.cut(hist,   bins=puntos_corte)
    actual_bins = pd.cut(actual, bins=puntos_corte)
    
    hist_pct    = hist_bins.value_counts(normalize=True, sort=False)   + 1e-6
    actual_pct  = actual_bins.value_counts(normalize=True, sort=False) + 1e-6
    
    psi= np.sum((actual_pct - hist_pct) * np.log(actual_pct / hist_pct))
    return psi

def calcular_ks(hist, actual):
    """KS test: devuelve el estadístico y el p-valor."""
    stat, p_value = ks_2samp(hist.dropna(), actual.dropna())
    return stat , p_value

def calcular_js(hist, actual, buckets=10):
    """Jensen-Shannon divergence para una variable númerica"""
    hist  = hist.dropna()
    actual= actual.dropna()
    
    minimo= min(hist.min(), actual.min())
    maximo= max(hist.max(), actual.max())
    bins  = np.linspace(minimo, maximo, buckets + 1)
    
    hist_hist, _   = np.histogram(hist, bins=bins, density=True)
    actual_hist, _ = np.histogram(actual, bins=bins, density=True)
    return jensenshannon(hist_hist + 1e-9, actual_hist + 1e-9)

def calcular_chi2(hist, actual):
    """Chi-cuadrado para una variable categórica: devuelve el p-valor."""
    categorias = sorted(set(hist.dropna().unique()) | set(actual.dropna().unique()))
    tabla = pd.DataFrame({
        'historico': hist.value_counts(). reindex(categorias, fill_value=0),
        'actual': actual.value_counts(). reindex(categorias, fill_value=0),
    })
    _, p_value, _, _=chi2_contingency(tabla)
    return p_value

#-------------- Nivel de alerta segun PSI (estándar de industria)---------------------

def nivel_alerta_combinada(psi, ks_p_value, umbral_significancia=0.05):
    """
    Combina PSI (magnitud del cambio, estándar de industria crediticia) con el
    p-valor de KS (significancia estadística) para confirmar el drift y reducir
    falsos positivos. Se prioriza PSI como criterio principal: cambios con PSI < 0.1
    se consideran estables aunque sean estadísticamente significativos, siguiendo
    el estándar de industria de que el umbral de negocio pesa más que la
    significancia estadística pura para decidir una alerta operativa.
    """
    significativo = ks_p_value < umbral_significancia

    if psi < 0.1:
        return "Estable"
    elif psi < 0.25:
        return "Alerta moderada" if significativo else "Revisar (no significativo)"
    else:
        return "Drift significativo" if significativo else "Revisar (PSI alto, no confirmado)"
    
#-------------------------------- Reporte Completo ------------------------------------

def generar_reporte_drift(df_historico, df_actual, num_cols, cat_cols):
    resultados = []
    
    for col in num_cols:
        psi= calcular_psi(df_historico[col], df_actual[col])
        ks_stat, ks_p= calcular_ks(df_historico[col], df_actual[col])
        js= calcular_js(df_historico[col], df_actual[col]) 
        
        resultados.append({
            'variable': col,
            'tipo': 'numérica',
            'psi': round(psi, 4),
            'ks_stat': round(ks_stat, 4),
            'ks_p_value': round(ks_p, 4),
            'js_divergence': round(js, 4),
            'alerta': nivel_alerta_combinada(psi, ks_p),
        })
    
    for col in cat_cols:
        chi2_p= calcular_chi2(df_historico[col], df_actual[col])
        resultados.append({
            'variable':col,
            'tipo': 'categórica',
            'psi': np.nan,
            'ks_stat': np.nan,
            'ks_p_value': np.nan,
            'js_divergence': np.nan,
            'chi2_p_value': round(chi2_p, 4),
            'alerta': "Drift significativo" if chi2_p < 0.05 else "Estable"
        })
    return pd.DataFrame(resultados)

    
#--------------------- Simulación de datos "actuales" (para fines demostrativos)--------------------------
def simular_datos_actuales(df, random_state= 42):
    """
    Simula el paso del tiempo  introduciendo un desplazamiento artificial en variables claves,
    para poder demostrar la detección del drift.
    """
    df_actual= df.copy()
    rng= np.random.default_rng(random_state)
    
    # Simula inflación: salarios y capital prestado suben ~15%
    df_actual['salario_cliente']  = df_actual['salario_cliente']  * 1.15
    df_actual['capital_prestado'] = df_actual['capital_prestado'] * 1.15
    
    # Simula cambio en política de crédito: sube el puntaje_datacredito promedio
    df_actual['puntaje_datacredito'] = df_actual['puntaje_datacredito'] + rng.normal(10, 5, len(df_actual))
    return df_actual

if __name__ == "__main__":
    X_train, X_eval, y_train, y_eval = construir_dataset()
    
    # Histórico = datos de entrenamiento. Actual= eval simulado con drift.
    df_historico = X_train
    df_actual = simular_datos_actuales(X_eval)
    
    num_cols = ['salario_cliente', 'capital_prestado', 'puntaje_datacredito',
                'saldo_total', 'cuota_pactada', 'edad_cliente']
    cat_cols = ['tendencia_ingresos'] #Queda númerica ordinal codificada, pero sirve de ejemplo    

    reporte  = generar_reporte_drift(df_historico, df_actual, num_cols, cat_cols)
    
    print("--- Reporte de Data Drift ---")
    print(reporte.to_string(index=False))