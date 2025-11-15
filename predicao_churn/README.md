# Projeto 2 – Predição de evasão (churn) de clientes

Este projeto ilustra como construir um modelo preditivo para identificar clientes com alta probabilidade de cancelar um serviço. A detecção precoce de churn permite que as empresas adotem ações de retenção direcionadas e reduza o impacto financeiro das perdas.

## Como executar

1. Certifique‑se de ter instaladas as bibliotecas `pandas`, `numpy`, `scikit‑learn` e `matplotlib`.
2. Execute o script `main.py`. O script gera um conjunto de dados sintético (ou usa dados reais se você substituir a função `generate_synthetic_data()`), explora a distribuição das variáveis, treina modelos de classificação e avalia a performance usando métricas como precisão, recall e F1‑score.
3. Para utilizar dados reais de churn (telecom, bancos, SaaS etc.), substitua a geração de dados pela leitura de um arquivo CSV e adapte as etapas de pré‑processamento conforme necessário.

## Estrutura

- **main.py** – script Python que gera dados sintéticos de churn, realiza análise exploratória, treina modelos de classificação e avalia resultados.
- **README.md** – este arquivo explicativo.

## Observações

Algumas melhorias para projetos reais incluem:
- Aplicar técnicas de balanceamento de classes (SMOTE ou ajuste de pesos) quando houver desbalanceamento entre clientes que saem e permanecem.
- Usar explicabilidade (como SHAP) para interpretar a importância das variáveis.
- Criar um pipeline de ML Ops para monitorar a performance do modelo em produção.
