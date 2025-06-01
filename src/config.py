# === Hiperparámetros de la red ===
INPUT_DIM = 20  # Reemplazar por la cantidad real de columnas de X
HIDDEN_LAYERS = [16, 8]
ACTIVATION_HIDDEN = 'relu'
ACTIVATION_OUTPUT = 'sigmoid'

# === Entrenamiento ===
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.3