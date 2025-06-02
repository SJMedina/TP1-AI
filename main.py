from src.preprocessing import load_data
from src.model import build_mlp
from src.train import train_model
from src.metrics import evaluate_model

if __name__ == "__main__":
    # 1. Cargar datos
    x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test = load_data(
        'data/dataset.csv', 'data/clean_dataset.csv'
    )

    # 2. Construir modelo con dimensión de variables numéricas
    model = build_mlp(input_dim_numeric=x_num_train.shape[1])

    # 3. Compilar y entrenar
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = train_model(
        model,
        x_cat_train=x_cat_train,
        x_num_train=x_num_train,
        y_train=y_train,
        x_cat_val=x_cat_test,
        x_num_val=x_num_test,
        y_val=y_test
    )

    # 4. Evaluar
    if model and history:
        x_test_all = x_cat_test + [x_num_test]
        evaluate_model(model, x_test_all, y_test, history)
    else:
        print("Error: El modelo o el historial no están disponibles.")