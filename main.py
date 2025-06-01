from src.preprocessing import load_data
from src.model import build_mlp
from src.train import train_model
from src.metrics import evaluate_model

if __name__ == "__main__":

    # 1. Cargar datos
    x_train, x_test, y_train, y_test = load_data('data/dataset.csv', 'data/clean_dataset.csv')

    # 2. Construir modelo
    model = build_mlp(input_dim=x_train.shape[1])

    # 3. Compilar y entrenar
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = train_model(model, x_train, y_train, x_test, y_test)

    # 4. Evaluar
    if model is not None and x_test is not None and y_test is not None and history is not None:
        evaluate_model(model, x_test, y_test, history)
    else:
        print("Error: El modelo, los datos de prueba o el historial no están disponibles.")