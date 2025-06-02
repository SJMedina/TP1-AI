import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(input_path: str, output_path: str, test_size=0.3, random_state=42):
    """
    Carga y preprocesa el dataset para entrenamiento con embeddings.
    Retorna: X_train_num, X_test_num, y_train, y_test, + listas categóricas codificadas
    """
    if not os.path.exists(output_path):
        df = pd.read_csv(input_path)

        # Eliminar columna ID
        df = df.drop(columns=['ID'])

        # Imputar valores nulos
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Codificar columna 'attack'
        df['attack'] = df['attack'].apply(lambda l: 1 if str(l).strip().upper() == 'Y' else 0)

        # Codificar categóricas como enteros para embeddings
        label_encoders = {}
        for col in ['protocol', 'browser_type', 'encryption_used']:
            le = LabelEncoder()
            df[col + '_int'] = le.fit_transform(df[col])
            label_encoders[col] = le

        # 🛠️ Eliminar columnas categóricas originales (string)
        df = df.drop(columns=['protocol', 'browser_type', 'encryption_used'])

        # Guardar dataset limpio
        df.to_csv(output_path, index=False)
        print(f"Dataset limpio guardado en {output_path}")

    # Cargar dataset limpio
    df = pd.read_csv(output_path)

    # Separar columnas
    categorical_cols = ['protocol_int', 'browser_type_int', 'encryption_used_int']
    numeric_cols = [col for col in df.columns if col not in categorical_cols + ['attack']]

    # Separar variables y target
    x_cat = df[categorical_cols]
    x_num = df[numeric_cols]
    y = df['attack']

    # Separar train/test
    x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test = train_test_split(
        x_num, x_cat, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Escalar solo las variables numéricas
    scaler = StandardScaler()
    x_num_train = scaler.fit_transform(x_num_train)
    x_num_test = scaler.transform(x_num_test)

    # Convertir categóricas a arrays para pasarlas como entrada a embeddings
    x_cat_train = [x_cat_train[col].values for col in categorical_cols]
    x_cat_test = [x_cat_test[col].values for col in categorical_cols]

    return x_num_train, x_num_test, x_cat_train, x_cat_test, y_train, y_test