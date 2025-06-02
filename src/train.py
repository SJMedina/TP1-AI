def train_model(model, x_cat_train,x_num_train, y_train,
                x_cat_val, x_num_val, y_val,
                epochs=50, batch_size=32):
    """
    Entrena el modelo con entradas separadas para embeddings y numéricas.
    """
    # Unir inputs (orden: embeddings + numéricas)
    x_train = x_cat_train + [x_num_train]
    x_val = x_cat_val + [x_num_val]

    # Entrenar
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight={0: 1, 1: 2},
        verbose=1
    )
    return history