"""
Projeto 1 – Previsão de vendas para lojas de varejo.

Este script gera um conjunto de dados sintético que simula as vendas semanais de uma loja de varejo,
realiza análise exploratória simples e treina modelos de regressão para prever vendas futuras.

Para usar dados reais, substitua a função `generate_synthetic_data()` pela leitura de seu dataset
de vendas e ajuste as etapas de pré‑processamento conforme necessário.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime as dt


def generate_synthetic_data(n_days: int = 365 * 2, seed: int = 42) -> pd.DataFrame:
    """Gera um dataset sintético de datas e vendas.

    O volume de vendas é composto por uma tendência linear,
    um componente sazonal semanal e ruído aleatório.
    """
    rng = np.random.default_rng(seed)
    start_date = dt.date(2023, 1, 1)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    # tendência linear de vendas
    trend = np.linspace(100, 200, n_days)
    # sazonalidade semanal (mais vendas aos fins de semana)
    weekly = 20 * np.sin(2 * np.pi * dates.dayofweek / 7)
    # ruído aleatório
    noise = rng.normal(0, 10, n_days)
    sales = trend + weekly + noise
    data = pd.DataFrame({"date": dates, "sales": sales})
    return data


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas a partir da coluna de datas."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # convertendo a data para ordinal para capturar tendência
    df["date_ordinal"] = df["date"].map(dt.datetime.toordinal)
    return df


def train_models(df: pd.DataFrame):
    """Treina modelos de regressão e retorna métricas de performance."""
    features = ["day_of_week", "month", "day_of_year", "is_weekend", "date_ordinal"]
    X = df[features]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Modelo de regressão linear
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lr = lin_reg.predict(X_test)

    # Modelo de Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Avaliação
    def evaluate(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        print(f"Modelo {name} – MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    evaluate(y_test, y_pred_lr, "Regressão Linear")
    evaluate(y_test, y_pred_rf, "Random Forest")

    return lin_reg, rf, X_test, y_test, y_pred_lr, y_pred_rf


def plot_results(y_test: pd.Series, preds: dict):
    """Plota vendas reais vs previsões."""
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test.values, label="Vendas reais")
    for name, y_pred in preds.items():
        plt.plot(y_test.index, y_pred, label=f"Previsão {name}")
    plt.xlabel("Índice de tempo")
    plt.ylabel("Vendas")
    plt.title("Previsão de vendas – comparação de modelos")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Gerar dados sintéticos
    df = generate_synthetic_data(n_days=365 * 2)
    df = create_features(df)

    # Visualização simples
    print(df.head())
    df.plot(x="date", y="sales", figsize=(10, 4), title="Vendas ao longo do tempo")
    plt.xlabel("Data")
    plt.ylabel("Vendas")
    plt.tight_layout()
    plt.show()

    # Treinar modelos
    lin_reg, rf, X_test, y_test, y_pred_lr, y_pred_rf = train_models(df)

    # Plotar resultados
    plot_results(
        y_test.reset_index(drop=True),
        {"Regressão Linear": y_pred_lr, "Random Forest": y_pred_rf},
    )


if __name__ == "__main__":
    main()
