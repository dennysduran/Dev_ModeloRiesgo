# Modelo de Riesgo Crediticio — Dev_ModeloRiesgo

## Caso de negocio

Una entidad financiera otorga créditos de consumo y necesita anticipar, al momento de evaluar una solicitud, la probabilidad de que un cliente **pague a tiempo** o **no pague a tiempo** (`Pago_atiempo`). Predecir este comportamiento permite:

- Priorizar la aprobación de créditos con menor riesgo.
- Ajustar condiciones (tasa, monto, plazo) según el perfil de riesgo del solicitante.
- Reducir pérdidas por cartera vencida sin frenar el crecimiento del negocio.

El proyecto simula un **pipeline de MLOps de punta a punta**: desde la exploración de datos hasta un modelo entrenado, versionado con Git/GitHub, y con un sistema de monitoreo que detecta cuándo el modelo podría estar perdiendo vigencia por cambios en la población de clientes.

## Estructura del repositorio

mlops_pipeline/
└── src/
├── Cargar_datos.ipynb # Carga inicial del dataset (uso exploratorio)
├── comprension_eda.ipynb # Análisis exploratorio de datos (EDA)
├── ft_engineering.py # Pipeline de limpieza y features
├── model_training_evaluation.py # Entrenamiento y evaluación de modelos
├── model_monitoring.py # Cálculo de métricas de data drift
└── app.py # Dashboard de monitoreo (Streamlit)
Base_de_datos.csv / .xlsx
requirements.txt
.gitignore


## Dataset

10,763 registros de clientes con 23 variables originales: información sociodemográfica (edad, tipo laboral, salario), del crédito (capital, plazo, cuota), y de historial crediticio (score externo, saldos, créditos vigentes). Variable objetivo: `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo).

## Proceso y hallazgos principales

### 1. Exploración de datos (EDA)

- **Desbalance fuerte de clases**: ~95% paga a tiempo vs. ~5% no paga — condiciona toda la estrategia de modelado y evaluación posterior (no se puede usar accuracy como métrica principal).
- **Valores fuera de dominio**: la columna `tendencia_ingresos` contenía valores numéricos residuales mezclados con sus 3 categorías válidas (Creciente/Estable/Decreciente); `tipo_credito` tenía un valor inválido (`68`) fuera de las categorías reales. Ambos se unificaron a `NaN`.
- **Hallazgo bivariable clave**: la variable derivada `tiene_mora` (saldo_mora > 0) es ~13 veces más frecuente en clientes que no pagan a tiempo — la señal más fuerte encontrada en el EDA.
- Multicolinealidad baja entre variables numéricas — sin riesgo relevante de redundancia para el modelado.

### 2. Ingeniería de características (`ft_engineering.py`)

- Limpieza de valores fuera de dominio.
- Creación de la variable derivada `tiene_mora`.
- Imputación de nulos (mediana para numéricas, moda para categóricas).
- Encoding: ordinal para `tendencia_ingresos`, One-Hot para variables nominales.
- Split estratificado 80/20 (por el desbalance de clases).

### 3. Entrenamiento y evaluación de modelos (`model_training_evaluation.py`)

Se compararon 4 modelos: Regresión Logística, Árbol de Decisión, Random Forest y XGBoost, usando funciones reutilizables (`build_model`, `summarize_classification`) y PR-AUC como criterio principal de selección (más robusto que accuracy o ROC-AUC dado el desbalance).

**Hallazgo crítico — Data Leakage:** la primera corrida arrojó métricas perfectas (PR-AUC = 1.000, Recall = 1.000) en 3 de los 4 modelos, una señal de alarma más que un buen resultado. Al revisar la importancia de variables, se detectó que la columna `puntaje` concentraba el 100% de la importancia del modelo. Un análisis estadístico confirmó que sus rangos por clase no se superponían en absoluto (máximo de la clase "no paga" < mínimo de la clase "sí paga"), lo cual indica que `puntaje` es un score recalculado *después* del resultado de pago — no disponible al momento real de otorgar el crédito.

Tras eliminar `puntaje` del conjunto de features, los resultados bajaron a valores realistas y defendibles:

| Modelo | Accuracy | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| Regresión Logística | 0.719 | 0.961 | 0.735 | 0.963 |
| Árbol de Decisión | 0.909 | 0.954 | 0.950 | 0.954 |
| **Random Forest** | **0.954** | **0.954** | **0.999** | **0.972** |
| XGBoost | 0.905 | 0.961 | 0.939 | 0.971 |

**Modelo seleccionado: Random Forest** (mejor PR-AUC, 0.972), con `puntaje_datacredito` (score externo de central de riesgo) como variable más importante tras la corrección del leakage.

### 4. Monitoreo de Data Drift (`model_monitoring.py` + `app.py`)

Se implementó un sistema de monitoreo que compara la distribución de datos históricos (usados para entrenar) contra datos actuales, usando 4 métricas complementarias:

- **PSI** (Population Stability Index) — estándar de industria crediticia, criterio principal de alerta (umbrales: < 0.1 estable, 0.1–0.25 alerta moderada, > 0.25 drift significativo).
- **KS test** — confirma significancia estadística del cambio.
- **Jensen-Shannon divergence** — magnitud de separación entre distribuciones.
- **Chi-cuadrado** — equivalente para variables categóricas.

**Nota metodológica:** dado que el dataset es estático (no productivo), los "datos actuales" y la evolución temporal del drift se generan mediante una simulación (desplazamiento artificial en variables clave como `salario_cliente` y `puntaje_datacredito`), con fines demostrativos del funcionamiento del sistema.

El dashboard en Streamlit (`app.py`) muestra:
- Semáforo resumen de alertas por variable.
- Tabla de métricas de drift con codificación por color.
- Comparación visual de distribuciones (histórico vs. actual).
- Evolución simulada del drift a lo largo de 4 períodos.

**Cómo ejecutar el dashboard:**
```bash
cd mlops_pipeline/src
streamlit run app.py
```

## Conclusiones y recomendaciones

- El pipeline es reproducible de punta a punta: cualquier persona que clone el repo, instale `requirements.txt` y ejecute los scripts en orden (`ft_engineering.py` → `model_training_evaluation.py` → `model_monitoring.py`) puede reconstruir el modelo y el monitoreo desde cero.
- El hallazgo de data leakage refuerza la importancia de validar la disponibilidad temporal real de cada variable antes de incluirla en el modelo, más allá de su poder predictivo aparente.
- Se recomienda, en un escenario productivo real, reemplazar la simulación de drift por datos reales de nuevos clientes muestreados con una periodicidad definida (ej. mensual), y activar reentrenamiento del modelo cuando el PSI de variables clave supere el umbral de 0.25 de forma sostenida.

