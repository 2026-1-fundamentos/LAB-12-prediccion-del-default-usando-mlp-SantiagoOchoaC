# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import os
import gzip
import pickle
import json

def load_data_csv(path):
    """Carga los datos desde un archivo CSV y devuelve un DataFrame de pandas."""
    return pd.read_csv(path, index_col= False, compression='zip', encoding='utf-8')
# Paso 1
def clean_data(df):
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    # Eliminar registros con información no disponible
    df = df.loc[df["EDUCATION"] != 0]
    df = df.loc[df["MARRIAGE"] != 0]
    # Agrupar valores > 4 en la categoría "others" para EDUCATION
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 4 else 4)
    df = df.dropna()
    return df
# Paso 2
def split_data(df_train, df_test):
    x_train = df_train.drop(columns=["default"])
    y_train = df_train["default"]
    x_test = df_test.drop(columns=["default"])
    y_test = df_test["default"]
    return x_train, y_train, x_test, y_test
# Paso 3
def create_pipeline(categorical_features, numerical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "scaler",
                StandardScaler(),
                numerical_features,
            ),
        ],
        remainder="passthrough",
        sparse_threshold=0,
    )

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("sel", SelectKBest(score_func=f_classif, k=20)),
        ("pca", PCA(n_components=None)),
        ("clf", MLPClassifier(hidden_layer_sizes=(50, 30, 40, 60),alpha=0.26,learning_rate_init=0.001,max_iter=15000,random_state=21
        ))
    ])

    return pipeline
# Paso 4
def optimize_hyperparameters(pipeline, x_train, y_train):
    grid_search = GridSearchCV(pipeline, param_grid={}, cv=10, scoring="balanced_accuracy", n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search
# Paso 5
def save_model(model):
    os.makedirs("./files/models", exist_ok=True)
    with gzip.open("./files/models/model.pkl.gz", "wb") as file:
        pickle.dump(model, file)
# Paso 6
def calculate_metrics(dataset, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": float(precision_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
    }
# Paso 7
def calculate_confusion_matrix(dataset, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }

def save_metrics(metrics):
    os.makedirs("./files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as file:
        for metric in metrics:
            file.write(json.dumps(metric) + "\n")
# Función principal
def main():
    # Cargar los datos
    df_train = load_data_csv("./files/input/train_data.csv.zip")
    df_test = load_data_csv("./files/input/test_data.csv.zip")

    # Limpiar los datos
    df_train_cleaned = clean_data(df_train)
    df_test_cleaned = clean_data(df_test)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, y_train, x_test, y_test = split_data(df_train_cleaned, df_test_cleaned)

    # Crear el pipeline
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric = [column for column in x_train.columns if column not in categorical]
    pipeline = create_pipeline(categorical, numeric)
    # Optimizar los hiperparámetros
    model = optimize_hyperparameters(pipeline, x_train, y_train)
    # Guardar el modelo
    save_model(model)
    # Calcular métricas
    metrics_train = calculate_metrics("train", y_train, model.predict(x_train))
    metrics_test = calculate_metrics("test", y_test, model.predict(x_test))
    cm_train = calculate_confusion_matrix("train", y_train, model.predict(x_train))
    cm_test = calculate_confusion_matrix("test", y_test, model.predict(x_test))
    # Guardar métricas
    save_metrics([metrics_train, metrics_test, cm_train, cm_test])

if __name__ == "__main__":
    main()