def generate_report(test_mae, rmse, r2, predictions, true_values, output_file):
    with open(output_file, "w") as f:
        f.write("<html><body>")
        f.write("<h1>Reporte de Resultados</h1>")
        f.write(f"<p>MAE en datos de prueba: {test_mae:.4f}</p>")
        f.write(f"<p>RMSE: {rmse:.4f}</p>")
        f.write(f"<p>R²: {r2:.4f}</p>")
        f.write("<h2>Predicciones</h2>")
        for pred, true in zip(predictions[:10], true_values[:10]):
            f.write(f"<p>Predicción: {pred:.4f}, Valor Real: {true:.4f}</p>")
        f.write("</body></html>")
