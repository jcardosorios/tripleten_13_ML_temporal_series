# src/data/make_dataset.py

import os
import requests
import pandas as pd
from pathlib import Path

def download_and_save(url: str, filename: str = None) -> None:
    '''
    Función que descarga los dataset desde los links entregados, y los guardará en la carpeta data/raw,
    si esta no existe se creará en la raiz.
    :param url: string
    :param filename (optional): string
    :return: None
    '''
    if not filename:
        start_index = url.rfind('/') + 1
        filename = url[start_index:].lower()

    project_root = Path(__file__).resolve().parent.parent.parent
    directory = project_root / 'data' / 'raw'


    if not os.path.exists(directory):
        os.makedirs(directory)
    
    full_path = os.path.join(directory, filename)

    if os.path.exists(full_path):
        print(f"El archivo {filename} ya existe en {directory}. Descarga cancelada.")
        return
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(full_path, 'wb') as file:
            file.write(response.content)
        print(f'Archivos guardado en: {full_path}')
    else:
        print('Error al descargar el archivo')

def load_and_resample_taxi_data(filepath: str, output_filepath: str = None) -> pd.DataFrame:
    """
    Carga el dataset de taxis, lo ordena por índice de tiempo, verifica si es monotónico creciente
    y lo remuestrea a una frecuencia horaria, sumando los pedidos.
    
    :param filepath: Ruta al archivo CSV original.
    :param output_filepath: Ruta opcional para guardar el DataFrame remuestreado.
    :return: DataFrame de Pandas con los datos remuestreados.
    """
    print(f"Cargando datos desde: {filepath}")
    df = pd.read_csv(filepath, index_col=[0], parse_dates=[0])
    df = df.sort_index()

    if not df.index.is_monotonic_increasing:
        raise ValueError("El índice del DataFrame no está ordenado de manera incremental.")
    
    print("Realizando remuestreo a frecuencia horaria...")
    df_resampled = df.resample('1h').sum()
    print("Remuestreo completado.")
    
    if output_filepath:
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_resampled.to_parquet(output_filepath, index=True)
        print(f"Datos remuestreados guardados en: {output_filepath}")

    return df_resampled