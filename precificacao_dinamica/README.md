# Projeto 3 – Estratégia de precificação dinâmica para assentos de aviação

Este projeto demonstra como construir um modelo preditivo e uma função de otimização para ajustar dinamicamente o preço de passagens aéreas. O objetivo é maximizar a receita, reduzir assentos vazios e oferecer preços competitivos conforme a proximidade da data de viagem. A implementação é inspirada em um estudo de caso apresentado pelo ProjectPro, que descreve um modelo de previsão de ocupação com **XGBRegressor** e uma camada de otimização para determinar a curva de preços ideal【424104200682404†L244-L266】.

## Como executar

1. Certifique -se de ter as bibliotecas `pandas`, `numpy`, `scikit‑learn`, `xgboost` e `matplotlib` instaladas.
2. Execute o script `main.py`. O script gera dados sintéticos de reservas de voos, treina um modelo de regressão (XGBoost) para prever a taxa de ocupação e aplica uma regra simples de precificação dinâmica com base no número de dias restantes e na demanda prevista.
3. Para utilizar dados reais, substitua a função `generate_synthetic_data()` pela leitura de dados de reservas e preços de uma companhia aérea e ajuste a função de estratégia de preços conforme as políticas internas.

## Estrutura

- **main.py** – script Python que gera dados sintéticos, treina o modelo de regressão, implementa uma estratégia de precificação dinâmica e avalia o impacto na receita.
- **README.md** – este documento.

## Observações

Para um cenário realista, considere:
- Incluir variáveis adicionais: classe da passagem, concorrência, eventos sazonais, etc.
- Empregar algoritmos de otimização mais sofisticados (programação linear/quadrática) para definir a curva de preço ótima.
- Atualizar o modelo de previsão em tempo real conforme novas reservas são feitas.
