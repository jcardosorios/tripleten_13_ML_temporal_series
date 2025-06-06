import joblib
import pandas as pd
import os

def load_model(filepath: str):
    """
    Carga un modelo de Machine Learning desde un archivo.
    :param filepath: Ruta al archivo del modelo.
    :return: El modelo cargado.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo del modelo no se encontró en: {filepath}")
    
    print(f"Cargando modelo desde: {filepath}")
    model = joblib.load(filepath)
    print("Modelo cargado exitosamente.")
    return model

def make_predictions(model, features: pd.DataFrame) -> pd.Series:
    """
    Realiza predicciones utilizando un modelo cargado.
    :param model: El modelo de Machine Learning.
    :param features: DataFrame con las características para hacer predicciones.
    :return: Serie de Pandas con las predicciones.
    """
    print("Realizando predicciones...")
    predictions = model.predict(features)
    print("Predicciones completadas.")
    return pd.Series(predictions, index=features.index, name='predicted_orders')