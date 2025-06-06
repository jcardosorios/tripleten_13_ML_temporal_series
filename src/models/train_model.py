import pandas as pd
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from src.utils.helpers import train_pred_time, test_parameters

def train_best_model(df_features: pd.DataFrame, target_column: str, test_size: float = 0.1, state: int = 31415, model_save_path: str = None):
    """
    Divide los datos, entrena el modelo LightGBM con los mejores parámetros encontrados
    y guarda el modelo entrenado.

    :param df_features: DataFrame con características y la columna objetivo.
    :param target_column: Nombre de la columna objetivo.
    :param test_size: Proporción del conjunto de prueba.
    :param state: Semilla para la reproducibilidad.
    :param model_save_path: Ruta para guardar el modelo entrenado.
    :return: El modelo entrenado y los DataFrames de características y objetivo de prueba.
    """
    # Eliminar NaNs generados por lags y rolling_mean
    df_features.dropna(inplace=True)

    # Separar características y objetivo
    features = df_features.drop([target_column], axis=1)
    target = df_features[target_column]

    # Dividir en conjuntos de entrenamiento y prueba
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, shuffle=False, test_size=test_size
    )
    
    print(f'Tamaños conjunto de entrenamiento: {features_train.shape}')
    print(f'Tamaños conjunto de prueba: {features_test.shape}')

    # Declaración del modelo LightGBM
    model = lgb.LGBMRegressor(boosting_type='gbdt', random_state=state, force_row_wise=True, verbose=-1)

    # Parámetros para la búsqueda de cuadrícula (puedes ajustar estos rangos)
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
    
    times, predictions = train_pred_time(final_model, features_train, target_train, features_test)

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(final_model, model_save_path)
        print(f"Modelo entrenado guardado en: {model_save_path}")

    return final_model, features_test, target_test, predictions