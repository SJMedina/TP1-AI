from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Activation

def build_mlp(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    # Primera capa oculta
    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # Segunda capa oculta
    model.add(Dense(8))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))
    return model
