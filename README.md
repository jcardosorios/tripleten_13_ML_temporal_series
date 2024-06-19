# Predicción de Demanda de Taxis para Aeropuertos

Este proyecto de Machine Learning tiene como objetivo predecir la cantidad de taxis que se necesitarán en el próximo intervalo de tiempo para pedidos a un aeropuerto, utilizando datos históricos de pedidos de taxis.

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características](#características)
- [Instalación](#instalación)
- [Uso](#uso)
- [Modelos](#modelos)
- [Datos](#datos)
- [Resultados](#resultados)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

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
git clone git@github.com:joaquincardosorios/tripleten_14_ML_text.git
cd tripleten_14_ML_text
python3 -m venv .venv
source .venv/bin/activate # En Windows usa `.venv\Scripts\activate`
pip install -r requirements.txt
```