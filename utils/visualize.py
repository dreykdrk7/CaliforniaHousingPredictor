import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def calculate_metrics(predictions, true_values):
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)
    return rmse, r2

def plot_training_metrics(history, output_dir=None):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.title("Evolución de la pérdida")
    if output_dir:
        plt.savefig(f"{output_dir}/loss_plot.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['mae'], label='MAE de entrenamiento')
    plt.plot(history.history['val_mae'], label='MAE de validación')
    plt.xlabel('Épocas')
    plt.ylabel('MAE')
    plt.legend()
    plt.title("Evolución del MAE")
    if output_dir:
        plt.savefig(f"{output_dir}/mae_plot.png")
    plt.show()

def plot_residuals(predictions, true_values, output_dir=None):
    residuals = true_values - predictions
    plt.figure(figsize=(10, 5))
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicciones')
    plt.ylabel('Residuales')
    plt.title('Gráfico de Residuales')
    if output_dir:
        plt.savefig(f"{output_dir}/residuals_plot.png")
    plt.show()
