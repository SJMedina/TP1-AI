import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(input_path: str, output_path: str, test_size=0.3, random_state=42):
    """
    Carga el dataset preprocesado y lo prepara para entrenar el modelo MLP.
    Si el archivo en outputPath no existe, lo crea a partir de inputPath.
    """
    # 1. Verificar si existe el archivo preprocesado
    if not os.path.exists(output_path):
        df = pd.read_csv(input_path)

        # Eliminar columna ID
        df = df.drop(columns=['ID'])

        # Imputar valores nulos. Numericos con mediana, categóricos con moda
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Codificar columna 'attack' (Y/N → 1/0)
        df['attack'] = df['attack'].apply(lambda l: 1 if str(l).strip().upper() == 'Y' else 0)

        # Codificar categóricas con One-Hot Encoding
        categorical_cols = ['protocol', 'browser_type', 'encryption_used']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Normalizar columnas numéricas (menos 'attack')
        scaler = StandardScaler()
        num_cols = df.drop(columns=['attack']).select_dtypes(include='number').columns
        df[num_cols] = scaler.fit_transform(df[num_cols])

        # Guardar dataset limpio
        df.to_csv(output_path, index=False)
        print(f"Dataset limpio guardado en {output_path}")

    # 2. Cargar el dataset preprocesado
    df = pd.read_csv(output_path)

    # 3. Separar variables predictoras y target
    x = df.drop(columns=['attack'])
    y = df['attack']
    # 4. Dividir en entrenamiento y test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # 5. Normalizar (StandardScaler)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test