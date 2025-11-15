# Projeto 1 – Previsão de vendas para lojas de varejo

Este projeto demonstra como utilizar Python e bibliotecas de aprendizado de máquina para prever o volume de vendas de uma loja de varejo. O objetivo é gerar previsões semanais/mensais que ajudem a otimizar estoques, escalas de funcionários e campanhas de marketing. A abordagem inclui geração ou importação de dados, análise exploratória, modelagem preditiva e avaliação.

## Como executar

1. Certifique‑se de ter instaladas as bibliotecas `pandas`, `numpy`, `scikit‑learn` e `matplotlib`.
2. Execute o script `main.py` para gerar um conjunto de dados sintético, treinar modelos de previsão e visualizar os resultados. O script demonstra a criação de variáveis derivadas, treinamento de modelos de regressão e cálculo de métricas de erro.
3. Para utilizar dados reais, substitua a função `generate_synthetic_data()` pelos seus dados de vendas, mantendo a estrutura de datas e valores.

## Estrutura

- **main.py** – script Python que gera dados sintéticos, realiza análise exploratória, treina modelos de regressão e avalia a performance.
- **README.md** – este arquivo, com explicações sobre o propósito do projeto e como executá‑lo.

## Observações

O modelo implementado é uma base que pode ser estendida. Para previsões mais precisas, considere:
- Incluir variáveis externas como clima, promoções e datas comemorativas.
- Testar modelos de séries temporais como Prophet ou ARIMA.
- Construir um painel interativo (Streamlit, Dash) para visualização das previsões.
