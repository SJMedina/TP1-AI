import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)

def plot_training_curves(history):
    """
    Grafica las curvas de accuracy y loss del entrenamiento y validación.
    """
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title("Accuracy por época")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title("Pérdida (Loss) por época")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, x_test, y_test, history=None):
    """
    Evalúa el modelo con datos de test y muestra métricas y gráficos.
    """
    y_prob = model.predict(x_test).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print("=== Evaluación del Modelo ===")
    print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_prob))

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("Curva ROC - MLP")
    plt.show()

    if history:
        plot_training_curves(history)
    else:
        print("No se proporcionó historial de entrenamiento para graficar.")