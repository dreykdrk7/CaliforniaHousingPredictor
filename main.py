import yaml
import json
from data.preprocess import load_and_preprocess_data
from model.build_model import build_model
from utils.logger import get_logger
from utils.visualize import calculate_metrics, plot_training_metrics, plot_residuals
from utils.report import generate_report
import os

def main():
    logger = get_logger("train")

    # Cargar configuración
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["output"]["logs_dir"], exist_ok=True)
    os.makedirs(config["output"]["predictions_dir"], exist_ok=True)
    os.makedirs(config["output"]["model_dir"], exist_ok=True)

    # Cargar y preprocesar datos
    logger.info("Cargando y preprocesando los datos...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        config["data"]["test_size"], config["data"]["random_state"]
    )

    # Construir modelo
    logger.info("Construyendo el modelo...")
    model = build_model(
        input_shape=X_train.shape[1],
        layers_config=config["model"]["layers"],
        activation=config["model"]["activation"],
        optimizer=config["model"]["optimizer"],
        loss=config["model"]["loss"],
        metrics=config["model"]["metrics"],
    )

    # Entrenar modelo
    logger.info("Entrenando el modelo...")
    history = model.fit(
        X_train, y_train,
        validation_split=config["training"]["validation_split"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        verbose=1
    )

    # Evaluar modelo
    logger.info("Evaluando el modelo...")
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    logger.info(f"MAE en datos de prueba: {test_mae}")

    # Graficar métricas
    plot_training_metrics(history, output_dir=config["output"]["logs_dir"])

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
        json.dump(history.history, f)
    logger.info(f"Historial guardado en {history_file}")

    # Generar reporte
    report_file = config["output"]["report_file"]
    generate_report(test_mae, rmse, r2, predictions, y_test.values, report_file)
    logger.info(f"Reporte generado en {report_file}")

if __name__ == "__main__":
    main()
