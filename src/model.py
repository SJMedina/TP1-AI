from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization, Flatten, Concatenate, Activation

def build_mlp(input_dim_numeric):
    # === Entradas categóricas ===
    input_browser = Input(shape=(1,), name='browser_type_input')
    embedding_browser = Embedding(input_dim=4, output_dim=2, name='browser_embedding')(input_browser)
    embedding_browser = Flatten()(embedding_browser)

    input_protocol = Input(shape=(1,), name='protocol_input')
    embedding_protocol = Embedding(input_dim=5, output_dim=2, name='protocol_embedding')(input_protocol)
    embedding_protocol = Flatten()(embedding_protocol)

    input_encryption = Input(shape=(1,), name='encryption_input')
    embedding_encryption = Embedding(input_dim=2, output_dim=2, name='encryption_embedding')(input_encryption)
    embedding_encryption = Flatten()(embedding_encryption)

    # === Entrada numérica ===
    input_numeric = Input(shape=(input_dim_numeric,), name='numeric_input')

    # === Concatenar todo ===
    merged = Concatenate(name='merged')([
        embedding_browser,
        embedding_protocol,
        embedding_encryption,
        input_numeric
    ])

    # === MLP ===
    x = Dense(16)(merged)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(8)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    output = Dense(1, activation='sigmoid', name='output')(x)

    # === Modelo final ===
    model = Model(
        inputs=[input_browser, input_protocol, input_encryption, input_numeric],
        outputs=output
    )

    return model