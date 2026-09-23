# Dockerfile
# Imagen para desplegar la API de riesgo crediticio (FastAPI + modelo entrenado)

# 1. Imagen base: Python liviano (slim = sin extras innecesarios, imagen más chica)
FROM python:3.11-slim

# 2. Carpeta de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar primero el requirements.txt e instalar dependencias
#    (se hace antes de copiar el resto del código para aprovechar el cache de Docker:
#    si el código cambia pero las librerías no, Docker no reinstala todo de nuevo)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el código fuente necesario para la API
COPY mlops_pipeline/src/model_deploy.py .
COPY mlops_pipeline/src/ft_engineering.py .
COPY mlops_pipeline/src/modelo_riesgo_crediticio.joblib .

# 5. Exponer el puerto donde va a correr la API
EXPOSE 8000

# 6. Comando que se ejecuta al iniciar el contenedor: levantar la API con Uvicorn
CMD ["Uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]