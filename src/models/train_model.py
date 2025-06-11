import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from src.utils.helpers import train_pred_time, test_parameters
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb 

def split_data(df_features: pd.DataFrame, target_column: str, test_size: float = 0.1, state: int = 31415):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    :param df: DataFrame con características y la columna
    objetivo.
    :param target_column: Nombre de la columna objetivo
    :test_size: Proporción del conjunto de prueba.
    :param state: Semilla para la reproducibilidad.
    :return: Conjuntos de entrenamiento y prueba.
    """

    # Eliminar NaNs generados por lags y rolling_mean
    df_features.dropna(inplace=True)
    
    # Separar características y objetivo
    features = df_features.drop([target_column], axis=1)
    target = df_features[target_column]

    # Dividir en conjuntos de entrenamiento y prueba
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, shuffle=False, test_size=test_size, random_state=state
    )
    
    print(f'Tamaños conjunto de entrenamiento: {features_train.shape}')
    print(f'Tamaños conjunto de prueba: {features_test.shape}')

    return features_train, features_test, target_train, target_test

def train_lr_model(df_features: pd.DataFrame, target_column: str, test_size: float = 0.1, state: int = 31415, model_save_path: str = None):
    """
    Entrena el modelo de Regresión Lineal con los mejores parámetros encontrados
    y guarda el modelo entrenado.

    :param df: DataFrame con características y la columna
    objetivo.
    :param target_column: Nombre de la columna objetivo.
    :param test_size: Proporción del conjunto de prueba.
    :param state: Semilla para la reproducibilidad.
    :param model_save_path: Ruta para guardar el modelo entrenado.
    :return: El modelo entrenado y los DataFrames de características y objetivo de prueba.
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    features_train, features_test, target_train, target_test = split_data(df_features, target_column, test_size, state)

    # Declaración del modelo de Regresión Lineal
    model = LinearRegression()

    # Entrenar el modelo 
    print("\nEntrenando el modelo...")    
    times, predictions = train_pred_time(model, features_train, target_train, features_test)

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
        print(f"Modelo entrenado guardado en: {model_save_path}")

    return model, features_test, target_test, predictions, times

def train_dt_model(df_features: pd.DataFrame, target_column: str, test_size: float = 0.1, state: int = 31415, model_save_path: str = None):
    """
    Entrena el modelo de Arbol de desiones con los mejores parámetros encontrados
    y guarda el modelo entrenado.

    :param df: DataFrame con características y la columna
    objetivo.
    :param target_column: Nombre de la columna objetivo.
    :param test_size: Proporción del conjunto de prueba.
    :param state: Semilla para la reproducibilidad.
    :param model_save_path: Ruta para guardar el modelo entrenado.
    :return: El modelo entrenado y los DataFrames de características y objetivo de prueba.
    """

    # Dividir los datos en conjuntos de entrenamiento y prueba
    features_train, features_test, target_train, target_test = split_data(df_features, target_column, test_size, state)

    # Declaración del modelo LightGBM
    model = DecisionTreeRegressor(random_state=state)

    # Parámetros para la búsqueda de cuadrícula
    param_grid = {
        'max_depth': list(range(2, 51, 2)),
    }

    print("\nBuscando los mejores hiperparámetros para Arbol de decisiones...")
    best_params = test_parameters(model, param_grid, features_train, target_train)

    # Entrenar el modelo con los mejores hiperparámetros
    print("\nEntrenando el modelo final con los mejores hiperparámetros...")
    final_model = DecisionTreeRegressor(**best_params, random_state=state)
    
    # Predecur utilizando modelo entrenado
    print("\nRealizando predicciones...")    
    times, predictions = train_pred_time(final_model, features_train, target_train, features_test)

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(final_model, model_save_path)
        print(f"Modelo entrenado guardado en: {model_save_path}")

    return final_model, features_test, target_test, predictions, times

def train_rf_model(df_features: pd.DataFrame, target_column: str, test_size: float = 0.1, state: int = 31415, model_save_path: str = None):
    """
    Entrena el modelo de Bosque aleatorio con los mejores parámetros encontrados
    y guarda el modelo entrenado.

    :param df: DataFrame con características y la columna
    objetivo.
    :param target_column: Nombre de la columna objetivo.
    :param test_size: Proporción del conjunto de prueba.
    :param state: Semilla para la reproducibilidad.
    :param model_save_path: Ruta para guardar el modelo entrenado.
    :return: El modelo entrenado y los DataFrames de características y objetivo de prueba.
    """

    # Dividir los datos en conjuntos de entrenamiento y prueba
    features_train, features_test, target_train, target_test = split_data(df_features, target_column, test_size, state)

    # Declaración del modelo LightGBM
    model = RandomForestRegressor(random_state=state)

    # Parámetros para la búsqueda de cuadrícula
    param_grid = {
        'n_estimators': list(range(30, 41, 2)),
        'max_depth': list(range(10, 21, 2)),
    }

    print("\nBuscando los mejores hiperparámetros para Bosque aleatorio...")
    best_params = test_parameters(model, param_grid, features_train, target_train)

    # Entrenar el modelo con los mejores hiperparámetros
    print("\nEntrenando el modelo final con los mejores hiperparámetros...")
    final_model = RandomForestRegressor(**best_params, random_state=state)
    
    # Realizar predicciones
    print("\nRealizando predicciones...")    
    times, predictions = train_pred_time(final_model, features_train, target_train, features_test)

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(final_model, model_save_path)
        print(f"Modelo entrenado guardado en: {model_save_path}")

    return final_model, features_test, target_test, predictions, times

def train_lgbt_model(df_features: pd.DataFrame, target_column: str, test_size: float = 0.1, state: int = 31415, model_save_path: str = None):
    """
    Entrena el modelo de Regresión Lineal con los mejores parámetros encontrados
    y guarda el modelo entrenado.

    :param df: DataFrame con características y la columna
    objetivo.
    :param target_column: Nombre de la columna objetivo.
    :param test_size: Proporción del conjunto de prueba.
    :param state: Semilla para la reproducibilidad.
    :param model_save_path: Ruta para guardar el modelo entrenado.
    :return: El modelo entrenado y los DataFrames de características y objetivo de prueba.
    """
    # Dividir los datos en conjuntos de entrenamiento y prueba
    features_train, features_test, target_train, target_test = split_data(df_features, target_column, test_size, state)

    # Declaración del modelo LightGBM
    model = lgb.LGBMRegressor(boosting_type='gbdt', random_state=state, force_row_wise=True, verbose=-1)

    # Parámetros para la búsqueda de cuadrícula
    param_grid = {
        'max_depth': [10, 20, 30],
        'num_leaves': [10, 20, 30],
        'n_estimators': [10, 100, 1000],
        'learning_rate': [0.1, 0.2]
    }

    print("\nBuscando los mejores hiperparámetros para LightGBM...")
    best_params = test_parameters(model, param_grid, features_train, target_train)

    # Entrenar el modelo con los mejores hiperparámetros
    print("\nEntrenando el modelo final con los mejores hiperparámetros...")
    final_model = lgb.LGBMRegressor(**best_params, boosting_type='gbdt', random_state=state, force_row_wise=True, verbose=-1)
    
    print('\nRealizando predicciones...')
    times, predictions = train_pred_time(final_model, features_train, target_train, features_test)

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(final_model, model_save_path)
        print(f"Modelo entrenado guardado en: {model_save_path}")

    return final_model, features_test, target_test, predictions, times

def train_cb_model(df_features: pd.DataFrame, target_column: str, test_size: float = 0.1, state: int = 31415, model_save_path: str = None):
    """
    Entrena el modelo de Catboost con los mejores parámetros encontrados
    y guarda el modelo entrenado.

    :param df: DataFrame con características y la columna
    objetivo.
    :param target_column: Nombre de la columna objetivo.
    :param test_size: Proporción del conjunto de prueba.
    :param state: Semilla para la reproducibilidad.
    :param model_save_path: Ruta para guardar el modelo entrenado.
    :return: El modelo entrenado y los DataFrames de características y objetivo de prueba.
    """

    # Dividir los datos en conjuntos de entrenamiento y prueba
    features_train, features_test, target_train, target_test = split_data(df_features, target_column, test_size, state)

    # Declaración del modelo CatBoost
    model = CatBoostRegressor(random_state=state, verbose=False,loss_function='RMSE')

    # Parámetros para la búsqueda de cuadrícula
    param_grid = {
    'iterations': [10, 100, 500],
    'learning_rate': [0.01, 0.1, 0.5],
    'depth': [1,5,10],
}

    print("\nBuscando los mejores hiperparámetros para CatBoost...")
    best_params = test_parameters(model, param_grid, features_train, target_train)

    # Entrenar el modelo con los mejores hiperparámetros
    print("\nEntrenando el modelo final con los mejores hiperparámetros...")
    final_model = CatBoostRegressor(**best_params, random_state=state, verbose=False,loss_function='RMSE')
    
    # Realizar predicciones 
    print("\nRealizando predicciones...")    
    times, predictions = train_pred_time(final_model, features_train, target_train, features_test)

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(final_model, model_save_path)
        print(f"Modelo entrenado guardado en: {model_save_path}")

    return final_model, features_test, target_test, predictions, times

def train_xgb_model(df_features: pd.DataFrame, target_column: str, test_size: float = 0.1, state: int = 31415, model_save_path: str = None):
    """
    Entrena el modelo de XGBoost con los mejores parámetros encontrados
    y guarda el modelo entrenado.

    :param df: DataFrame con características y la columna
    objetivo.
    :param target_column: Nombre de la columna objetivo.
    :param test_size: Proporción del conjunto de prueba.
    :param state: Semilla para la reproducibilidad.
    :param model_save_path: Ruta para guardar el modelo entrenado.
    :return: El modelo entrenado y los DataFrames de características y objetivo de prueba.
    """

    # Dividir los datos en conjuntos de entrenamiento y prueba
    features_train, features_test, target_train, target_test = split_data(df_features, target_column, test_size, state)

    # Declaración del modelo CatBoost
    model = xgb.XGBRegressor( random_state=state, eval_metric='rmse')

    # Parámetros para la búsqueda de cuadrícula
    param_grid = {
        'n_estimators': [10, 100, 500],
        'max_depth': [1,5,10],
        'learning_rate': [0.01, 0.1, 0.5],
    }

    print("\nBuscando los mejores hiperparámetros para XGBoost...")
    best_params = test_parameters(model, param_grid, features_train, target_train)

    # Entrenar el modelo con los mejores hiperparámetros
    print("\nEntrenando el modelo final con los mejores hiperparámetros...")
    final_model = xgb.XGBRegressor(**best_params, random_state=state, eval_metric='rmse')
    
    # Realizar predicciones... 
    print("\nRealizando predicciones...")    
    times, predictions = train_pred_time(final_model, features_train, target_train, features_test)

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(final_model, model_save_path)
        print(f"Modelo entrenado guardado en: {model_save_path}")

    return final_model, features_test, target_test, predictions, times
