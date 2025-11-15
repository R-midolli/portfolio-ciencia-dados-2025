"""
Projeto 3 – Estratégia de precificação dinâmica para assentos de aviação.

Este script gera um conjunto de dados sintético que simula reservas de passagens aéreas,
treina um modelo de regressão usando XGBoost para prever a taxa de ocupação (venda de assentos)
e aplica uma estratégia de precificação dinâmica baseada em dias restantes para a viagem e
na demanda prevista. A receita gerada pela estratégia dinâmica é comparada com um
preço fixo para demonstrar potenciais ganhos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def generate_synthetic_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Gera dados sintéticos de reservas de voos.

    Cada amostra representa a situação de um voo em uma data específica.
    Variáveis:
      - days_to_travel: dias restantes até a viagem (0 a 30)
      - base_price: preço base da passagem
      - weekday: dia da semana (0=segunda, 6=domingo)
      - demand_factor: indicador de demanda sazonal (0 a 1)
      - occupancy_rate: taxa de ocupação (0 a 1), alvo para o modelo
    """
    rng = np.random.default_rng(seed)
    days_to_travel = rng.integers(0, 31, n_samples)
    weekday = rng.integers(0, 7, n_samples)
    # fator de demanda sazonal (ex.: férias têm maior demanda)
    demand_factor = rng.uniform(0.5, 1.2, n_samples)
    # preço base ajustado (voos mais próximos tendem a ser mais caros)
    base_price = 100 + 3 * (30 - days_to_travel) + 10 * demand_factor
    # taxa de ocupação simulada com influência inversa dos dias restantes
    occupancy_rate = 0.3 + 0.5 * (1 - days_to_travel / 30) + 0.2 * demand_factor
    # adicionar ruído e limitar entre 0 e 1
    occupancy_rate += rng.normal(0, 0.05, n_samples)
    occupancy_rate = np.clip(occupancy_rate, 0, 1)
    df = pd.DataFrame({
        "days_to_travel": days_to_travel,
        "weekday": weekday,
        "demand_factor": demand_factor,
        "base_price": base_price,
        "occupancy_rate": occupancy_rate,
    })
    return df


def train_model(df: pd.DataFrame):
    """Treina um modelo XGBoost para prever a taxa de ocupação."""
    features = ["days_to_travel", "weekday", "demand_factor", "base_price"]
    X = df[features]
    y = df["occupancy_rate"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE do modelo de previsão de ocupação: {rmse:.4f}")
    return model


def dynamic_pricing(df: pd.DataFrame, model: XGBRegressor) -> pd.DataFrame:
    """Aplica uma estratégia de precificação dinâmica.

    A estratégia ajusta o preço base de acordo com o número de dias restantes
    (diminui quando a viagem está distante para incentivar compras e aumenta
    quando se aproxima da data) e considera a demanda prevista.

    Retorna DataFrame com colunas adicionais: predicted_occupancy, dynamic_price e revenue.
    """
    df = df.copy()
    features = ["days_to_travel", "weekday", "demand_factor", "base_price"]
    df["predicted_occupancy"] = model.predict(df[features])
    # Estratégia simples de preço: começa no preço base e aumenta à medida que a ocupação esperada se aproxima de 100%
    # e os dias restantes diminuem. A constante alpha controla a magnitude do ajuste.
    alpha = 0.3
    df["dynamic_price"] = df["base_price"] * (
        1 + alpha * (df["predicted_occupancy"] - 0.5) * (1 / (df["days_to_travel"] + 1))
    )
    # calcular receita estimada com a estratégia dinâmica
    df["revenue_dynamic"] = df["dynamic_price"] * df["predicted_occupancy"]
    # receita base (preço fixo * taxa de ocupação prevista)
    df["revenue_base"] = df["base_price"] * df["predicted_occupancy"]
    return df


def evaluate_revenue(df: pd.DataFrame):
    """Compara receita média entre estratégia dinâmica e preço base."""
    mean_dynamic = df["revenue_dynamic"].mean()
    mean_base = df["revenue_base"].mean()
    improvement = (mean_dynamic - mean_base) / mean_base * 100
    print(f"Receita média (base): R${{mean_base:.2f}}")
    print(f"Receita média (dinâmica): R${{mean_dynamic:.2f}}")
    print(f"Melhoria percentual: {improvement:.2f}%")

    # Plotar comparação de preços e receitas
    plt.figure(figsize=(10, 5))
    plt.scatter(df["days_to_travel"], df["base_price"], label="Preço base", alpha=0.5)
    plt.scatter(df["days_to_travel"], df["dynamic_price"], label="Preço dinâmico", alpha=0.5)
    plt.xlabel("Dias até a viagem")
    plt.ylabel("Preço")
    plt.title("Preço base vs. preço dinâmico em função dos dias restantes")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    df = generate_synthetic_data()
    print("Amostra dos dados gerados:")
    print(df.head())

    model = train_model(df)
    df_results = dynamic_pricing(df, model)
    evaluate_revenue(df_results)


if __name__ == "__main__":
    main()
