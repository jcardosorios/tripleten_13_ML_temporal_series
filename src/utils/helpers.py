import time
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

def print_rmse(target_values: pd.Series, predicted_values: pd.Series) -> float:
    """
    Calcula y devuelve el valor RMSE entre los datos reales y una preidcción realizada por un modelo de ML.
    Parámetros:
    target_values (pd.Series): Conjunto de valores reales del objetivo.
    predicted_values (pd.Series): Predicciones realizadas.
    """
    rmse = mean_squared_error(target_values, predicted_values)**0.5
    print(f'El RMSE calculado es: {round(rmse, 2)}')
    return rmse

def train_pred_time(model: BaseEstimator, features_train: pd.DataFrame, target_train: pd.Series, features_test: pd.DataFrame):
    """
    Genera el entrenamiento de un modelo de ML, ingresando las caracteristicas y objetivos de entrenamiento,
    para luego generar una prediccion con el modelo utilizando las caracteristicas de prueba.
    A su vez calcula el tiempo entre cada evento para devolver los tiempos de entrenamiento,
    de prediccion ademas de la predicción en si.
    Parámetros:
    model: modelo de ML a utlizar.
    features_train: Conjunto de entrenamiento.
    target_train: Objetivo de entrenamiento.
    features_test: Conjunto de pruebas.
    """
    times_dict = {}
    prediction_dict = {}

    train_start_time = time.time()
    model.fit(features_train, target_train)
    train_end_time = time.time()

    prediction_dict['test'] = model.predict(features_test)
    prediction_end_time = time.time()

    prediction_dict['train'] = model.predict(features_train)

    times_dict['train'] = train_end_time - train_start_time
    times_dict['predict'] = prediction_end_time - train_end_time

    print(f"Tiempo de entrenamiento: {round(times_dict['train'], 4)} s")
    print(f"Tiempo de predicción: {round(times_dict['predict'], 4)} s")
    return times_dict, prediction_dict

def test_parameters(model: BaseEstimator, param_grid: dict, features_train: pd.DataFrame, target_train: pd.Series) -> dict:
    """
    Realiza una búsqueda de cuadrícula para encontrar los mejores hiperparámetros para un modelo dado.
    Parámetros:
    model (BaseEstimator): El modelo de ML a optimizar.
    param_grid (dict): Diccionario con los hiperparámetros a probar.
    features_train (pd.DataFrame): Características de entrenamiento.
    target_train (pd.Series): Objetivo de entrenamiento.
    Retorna:
    dict: Los mejores hiperparámetros encontrados.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3)
    grid_search.fit(features_train, target_train)
    best_params = grid_search.best_params_

    print("Mejores hiperparámetros encontrados:")
    print(best_params)
    return best_params