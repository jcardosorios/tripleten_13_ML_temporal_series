import time
import os
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

def predictions_to_parquet(result: pd.DataFrame, path: str):
    """
    Abre archivo parquet para actualizar su información y 
    luego volver a guardarlo.
    Párametros:
    result (pd.DataFrame): Columna de resultados.
    path (str): Ruta del archivo parquet.
    """
    if not os.path.exists(path):
        result.to_parquet(path)
        print(f"Archivo creado y guardado en: {path}")
    else:
        df = pd.read_parquet(path)

        for col in result.columns:
            # Actualiza o crea la columna con alineación por índice
            df[col] = result[col]

        df.to_parquet(path)
        print(f"Archivo actualizado en: {path}")


def results_to_parquet(results: list , model_name: str, path: str):
    """
    Guarda los resultados de las predicciones en un archivo parquet.
    Parámetros:
    results (dict): Diccionario con los resultados de las predicciones.
    model_name (str): Nombre del modelo.
    path (str): Ruta del archivo parquet.
    """
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['model', 'training_time', 'prediction_time', 'rmse_test'])
        df.loc[len(df)] = [model_name, results[0], results[1], results[2]]
        df.to_parquet(path, index=False)
        print(f"Archivo creado y guardado en: {path}")
    else:
        df = pd.read_parquet(path)
        if model_name in df['model'].values:
            df.loc[df['model'] == model_name, ['training_time', 'prediction_time', 'rmse_test']] = [results[0], results[1], results[2]]
            print(f"Fila para '{model_name}' actualizada en: {path}")
        else:
            df.loc[len(df)] = [model_name, results[0], results[1], results[2]]
            print(f"Nueva fila para '{model_name}' agregada en: {path}")
        df.to_parquet(path, index=False)