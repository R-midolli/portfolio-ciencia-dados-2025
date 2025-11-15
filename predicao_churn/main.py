"""
Projeto 2 – Predição de churn de clientes.

Este script gera um conjunto de dados sintético usando `sklearn.datasets.make_classification`
para simular características de clientes e um rótulo de churn. Em seguida,
ele divide os dados em treino e teste, treina modelos de classificação e avalia
suas métricas. Para usar dados reais, substitua a função de geração de dados
pela leitura de um arquivo CSV e ajuste as transformações.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42):
    """Gera dados sintéticos para churn usando make_classification.

    Retorna um DataFrame com colunas de características e um alvo `churn`.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        flip_y=0.03,
        class_sep=1.0,
        random_state=seed,
    )
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["churn"] = y
    return df


def preprocess_data(df: pd.DataFrame):
    """Separa recursos e alvo, aplica padronização e cria conjuntos de treino/teste."""
    X = df.drop("churn", axis=1)
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Treina modelos de classificação e imprime métricas de avaliação."""
    models = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, digits=3))
        cm = confusion_matrix(y_test, y_pred)
        print("Matriz de confusão:\n", cm)


def main():
    df = generate_synthetic_data()
    print("Exemplo de dados:")
    print(df.head())

    # Análise simples: distribuição de classes
    print("\nDistribuição de churn:")
    print(df["churn"].value_counts(normalize=True))

    # Pré-processamento
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Treinar e avaliar
    train_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
