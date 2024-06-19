# Predicción de Demanda de Taxis para Aeropuertos

Este proyecto de Machine Learning tiene como objetivo predecir la cantidad de taxis que se necesitarán en el próximo intervalo de tiempo para pedidos a un aeropuerto, utilizando datos históricos de pedidos de taxis.

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características](#características)
- [Instalación](#instalación)
- [Modelos](#modelos)
- [Datos](#datos)

## Descripción General

El objetivo de este proyecto es desarrollar modelos de Machine Learning para predecir la demanda futura de taxis en intervalos de una hora en pedidos hacia un aeropuerto. Se analizan varios modelos para determinar cuál proporciona las mejores predicciones.

## Características

- Preprocesamiento de datos de series temporales
- Entrenamiento y evaluación de múltiples modelos
- Optimización de hiperparámetros
- Comparación de rendimiento entre modelos

## Instalación

Para utilizar este proyecto, sigue estos pasos:

1. Clona el repositorio:

```bash
git clone git@github.com:joaquincardosorios/tripleten_13_ML_temporal_series.git
cd tripleten_13_ML_temporal_series
python3 -m venv .venv
source .venv/bin/activate # En Windows usa `.venv\Scripts\activate`
pip install -r requirements.txt
```

## Modelos

### Regresión Lineal

La Regresión Lineal es un modelo básico que establece una relación lineal entre las variables predictoras y la variable objetivo. En este proyecto, se utiliza para predecir la cantidad de taxis necesarios en función de características históricas de los pedidos.

### Árbol de Decisiones

El Árbol de Decisiones es un modelo no lineal que divide el conjunto de datos en subconjuntos más pequeños, basándose en las características más importantes para mejorar la precisión de las predicciones de demanda de taxis.

### Bosque Aleatorio

El Bosque Aleatorio es un ensamble de múltiples árboles de decisión que promedia las predicciones individuales de cada árbol. Se utiliza para reducir el sobreajuste y mejorar la robustez del modelo en la predicción de series temporales.

### LightGBM, Catboost, XGBoost

Estos modelos de gradient boosting (LightGBM, Catboost y XGBoost) son conocidos por su capacidad para manejar grandes volúmenes de datos y mejorar la precisión en problemas complejos de series temporales como la predicción de la demanda de taxis hacia aeropuertos.

- **LightGBM:** Optimiza el entrenamiento utilizando árboles de decisión en hojas.
  
- **Catboost:** Optimizado para manejar automáticamente variables categóricas y proporcionar una alta precisión sin ajuste de hiperparámetros.
  
- **XGBoost:** Implementa regularización, manejo de missing values y optimización de funciones de coste.

## Datos

Los datos utilizados para este proyecto incluyen registros históricos de pedidos de taxis hacia el aeropuerto, capturando variables temporales y características relevantes que afectan la demanda.

