from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Activation

def build_mlp(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    # Capa de entrada
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Capa oculta
    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))
    return model