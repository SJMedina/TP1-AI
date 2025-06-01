def train_model(model, x_train, y_train, x_val, y_val, epochs=50, batch_size=32):
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight={0: 1, 1: 2},
        verbose=1
    )
    return history