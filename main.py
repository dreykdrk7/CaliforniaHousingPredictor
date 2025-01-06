import yaml
import json
import numpy as np
import os
from data.preprocess import load_and_preprocess_data
from model.build_model import build_model
from utils.logger import get_logger
from utils.visualize import calculate_metrics, plot_training_metrics, plot_residuals
from utils.report import generate_report
from tensorflow.keras.models import load_model
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def main():
    logger = get_logger("train")

    # Cargar configuración
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Crear directorios si no existen
    os.makedirs(config["output"]["logs_dir"], exist_ok=True)
    os.makedirs(config["output"]["predictions_dir"], exist_ok=True)
    os.makedirs(config["output"]["model_dir"], exist_ok=True)

    # Ruta del modelo guardado
    model_path = f"{config['output']['model_dir']}/fine_tuned_model.h5"

    # Cargar datos
    logger.info("Cargando y preprocesando los datos...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        config["data"]["test_size"], config["data"]["random_state"]
    )

    # Intentar cargar un modelo existente
    if os.path.exists(model_path):
        logger.info(f"Cargando modelo existente desde {model_path}...")
        model = load_model(model_path)
    else:
        logger.info("No se encontró un modelo previo. Construyendo un modelo nuevo...")
        model = build_model(
            input_shape=X_train.shape[1],
            layers_config=config["model"]["layers"],
            activation=config["model"]["activation"],
            optimizer=config["model"]["optimizer"],
            loss=config["model"]["loss"],
            metrics=config["model"]["metrics"],
        )

    # Si es necesario, entrenar el modelo desde cero o continuar con el fine-tuning
    if not os.path.exists(model_path):
        logger.info("Entrenando el modelo inicial...")
        history = model.fit(
            X_train, y_train,
            validation_split=config["training"]["validation_split"],
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            verbose=1
        )
        # Guardar modelo inicial
        model.save(model_path)
        logger.info(f"Modelo inicial guardado en {model_path}")

    # Fine-tuning del modelo
    logger.info("Realizando fine-tuning del modelo...")
    history_fine_tuning = model.fit(
        X_train, y_train,
        validation_split=config["training"]["validation_split"],
        epochs=config["training"]["fine_tuning_epochs"],
        batch_size=config["training"]["batch_size"],
        verbose=1
    )
    # Guardar modelo ajustado
    model.save(model_path)
    logger.info(f"Modelo ajustado guardado en {model_path}")

    # Evaluar modelo
    logger.info("Evaluando el modelo ajustado...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    logger.info(f"MAE en datos de prueba: {test_mae}")

    # Graficar métricas
    plot_training_metrics(history_fine_tuning, output_dir=config["output"]["logs_dir"])

    # Generar predicciones
    predictions = model.predict(X_test).flatten()

    # Calcular métricas adicionales
    rmse, r2 = calculate_metrics(predictions, y_test.values)
    logger.info(f"RMSE: {rmse}, R²: {r2}")

    # Graficar residuales
    plot_residuals(predictions, y_test.values, output_dir=config["output"]["logs_dir"])

    # Guardar historial
    history_file = f"{config['output']['logs_dir']}/training_history.json"
    with open(history_file, "w") as f:
        json.dump(history_fine_tuning.history, f)
    logger.info(f"Historial guardado en {history_file}")

    # Validación cruzada
    logger.info("Realizando validación cruzada...")
    def create_model():
        return build_model(
            input_shape=X_train.shape[1],
            layers_config=config["model"]["layers"],
            activation=config["model"]["activation"],
            optimizer=config["model"]["optimizer"],
            loss=config["model"]["loss"],
            metrics=config["model"]["metrics"],
        )
    model_sklearn = KerasRegressor(build_fn=create_model, epochs=50, batch_size=32, verbose=0)
    scores = cross_val_score(model_sklearn, X_train, y_train, cv=5)
    logger.info(f"Scores de validación cruzada: {scores}")

    # Generar reporte
    report_file = config["output"]["report_file"]
    generate_report(test_mae, rmse, r2, predictions, y_test.values, report_file)
    logger.info(f"Reporte generado en {report_file}")

if __name__ == "__main__":
    main()
