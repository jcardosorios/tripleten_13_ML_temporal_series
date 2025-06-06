# src/visualization/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_time_series(df: pd.DataFrame, title: str, ylabel: str, save_path: str = None):
    """
    Genera un gráfico de serie de tiempo.
    :param df: DataFrame con la serie de tiempo.
    :param title: Título del gráfico.
    :param ylabel: Etiqueta del eje Y.
    :param save_path: Ruta para guardar el gráfico (opcional).
    """
    plt.figure(figsize=(14, 7))
    df.plot(ax=plt.gca())
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Fecha')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Gráfico guardado en: {save_path}")
    plt.show()

def plot_seasonal_decompose(decomposed, save_path: str = None):
    """
    Plotea los componentes de la descomposición estacional (tendencia, estacionalidad, residuos).
    :param decomposed: Objeto de descomposición estacional (seasonal_decompose).
    :param save_path: Ruta para guardar el gráfico (opcional).
    """
    plt.figure(figsize=(10, 10)) # Aumenta el tamaño para una mejor visualización

    plt.subplot(311)
    decomposed.trend.plot(ax=plt.gca())
    plt.title('Tendencia')

    plt.subplot(312)
    decomposed.seasonal.plot(ax=plt.gca())
    plt.title('Estacionalidad')

    plt.subplot(313)
    decomposed.resid.plot(ax=plt.gca())
    plt.title('Residuos')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Gráfico de descomposición guardado en: {save_path}")
    plt.show()

def plot_model_comparison(models_df: pd.DataFrame, save_path: str = None):
    """
    Genera un gráfico comparativo de tiempos y RMSE para diferentes modelos.
    :param models_df: DataFrame con los resultados de los modelos.
    :param save_path: Ruta para guardar el gráfico (opcional).
    """
    models_df_sorted = models_df.sort_values(by='rmse_test')

    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.bar(models_df_sorted['model'], models_df_sorted['training_time'], color='skyblue', alpha=0.7, label='Tiempo de entrenamiento')
    ax1.bar(models_df_sorted['model'], models_df_sorted['prediction_time'], color='lightcoral', alpha=0.7, label='Tiempo de predicción', bottom=models_df_sorted['training_time'])
    ax1.set_xlabel('Modelo')
    ax1.set_ylabel('Tiempo (s)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Comparación de Métricas y Tiempos de Modelos de ML')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(models_df_sorted['model'], models_df_sorted['rmse_train'], marker='o', color='green', label='RMSE de entrenamiento')
    ax2.plot(models_df_sorted['model'], models_df_sorted['rmse_test'], marker='o', color='red', label='RMSE de prueba', linewidth=2)
    ax2.set_ylabel('RMSE', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Gráfico de comparación de modelos guardado en: {save_path}")
    plt.show()

def plot_test_predictions(original_df: pd.DataFrame, prediction_dfs: dict, start_date: str = '2018-08-01', window: int = 50, save_path: str = None):
    """
    Compara las predicciones de los modelos con los valores reales en el conjunto de prueba.
    
    :param original_df: DataFrame original con 'num_orders'.
    :param prediction_dfs: Diccionario de DataFrames de predicciones (ej. {'LR': df_test_lr}).
    :param start_date: Fecha de inicio para la visualización del conjunto de prueba.
    :param window: Tamaño de la ventana para la media móvil de suavizado.
    :param save_path: Ruta para guardar el gráfico (opcional).
    """
    plt.figure(figsize=(14, 7))

    # Suavizar el dataframe original
    smoothed_original = original_df['num_orders'].rolling(window).mean().loc[start_date:]
    plt.plot(smoothed_original.index, smoothed_original, label='Demanda Real (media móvil)', color='red', linewidth=2)

    colors = ['green', 'yellow', 'purple', 'blue', 'cyan', 'brown']
    for i, (model_name, pred_df) in enumerate(prediction_dfs.items()):
        smoothed_prediction = pred_df['num_orders'].rolling(window).mean()
        plt.plot(smoothed_prediction.index, smoothed_prediction, label=f'Predicción {model_name} (media móvil)', color=colors[i % len(colors)], linestyle='--')

    plt.title('Comparativa de Predicciones del Conjunto de Prueba vs. Demanda Real')
    plt.xlabel('Fecha')
    plt.ylabel('Número de pedidos (media móvil)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Gráfico de predicciones guardado en: {save_path}")
    plt.show()