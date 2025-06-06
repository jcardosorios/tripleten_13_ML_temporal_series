import pandas as pd

def make_features(df: pd.DataFrame, max_lag: int, rolling_mean_size: int) -> pd.DataFrame:
    """
    Preparación de df para entrenamiento, separando la información de las fechas,
    ingresando lags de los datos a cada característica y obteniendo la media
    móvil.
    
    :param df: DataFrame de entrada con la columna 'num_orders'.
    :param max_lag: Número máximo de lags a crear.
    :param rolling_mean_size: Tamaño de la ventana para la media móvil.
    :return: DataFrame con las características añadidas.
    """
    df_copy = df.copy() 
    
    # Información de la fecha
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['day'] = df_copy.index.day
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['hour'] = df_copy.index.hour
    
    # Lags de los datos
    for lag in range(1, max_lag + 1):
        df_copy[f'lag_{lag}'] = df_copy['num_orders'].shift(lag)
		
    # La media móvil del rango anterior a la fecha
    df_copy['rolling_mean'] = df_copy['num_orders'].shift().rolling(rolling_mean_size).mean()

    return df_copy