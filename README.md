# Dashboard de predição da resistência à compressão do concreto (fck)

Implementação do dashboard interativo prometido na Metodologia e na Conclusão do TCC
**"Modelos de Machine Learning para Predição e Avaliação da Resistência à Compressão do Concreto"**
— Luan Carrera Santos, MBA em Data Science and Analytics, USP/ESALQ, 2026.

---

## Como rodar

**Na nuvem (sem instalar nada):** siga o [DEPLOY.md](DEPLOY.md) para publicar no
Streamlit Community Cloud e obter um link público.

**Local:**

```bash
pip install -r requirements.txt
python train_model.py        # baixa o dataset UCI, limpa, treina e valida (~2-4 min)
streamlit run app.py
```

O dashboard abre em `http://localhost:8501`.

O `train_model.py` é opcional: se os artefatos não existirem, o próprio `app.py`
executa o pipeline no primeiro acesso e mostra o progresso na tela. Rodar pelo
terminal só é mais conveniente porque imprime a Tabela 3 diretamente.

---

## O que cada aba faz

| Aba | Conteúdo |
|---|---|
| **🎯 Preditor de traço** | Sliders para os 8 constituintes + idade. Retorna fck previsto, intervalo de 95%, **probabilidade de atingir o fck alvo**, curva de evolução com a idade, análise de sensibilidade univariada e comparação entre os 5 modelos. |
| **🔍 Explicabilidade (SHAP)** | Waterfall local do traço atual, importância global (Figura 19), beeswarm (Figura 20) e dependência parcial com variável de interação selecionável (Figuras 21–23). |
| **📊 Comparação de modelos** | Tabela 3 completa, Figura 12 com barras de erro, predito×observado e diagnóstico de resíduos (histograma + QQ-plot) por modelo, distribuição das métricas entre os 10 folds. |
| **📈 Exploração dos dados** | Figuras 2 a 6, matriz de correlação, explorador livre X/Y/cor e estatísticas descritivas. |

---

## Estrutura

```
dashboard_fck/
├── app.py                 dashboard Streamlit (4 abas)
├── train_model.py         pipeline de treino e validação cruzada k=10
├── requirements.txt
├── src/
│   └── pipeline.py        dados, limpeza, feature engineering, definição dos modelos
├── data/                  gerado — concrete_raw.csv, concrete_processed.csv
└── models/                gerado — 5 modelos .pkl, metrics.json, oof_predictions.csv,
                           ols_summary.txt, clean_log.json
```

---

## Fidelidade ao TCC

**Engenharia de variáveis** — as cinco derivadas da seção "Processamento dos dados":

| Variável | Fórmula |
|---|---|
| `W_B` | `WATER / (CEM + SLAG + FLY_ASH)` |
| `LOG_AGE` | `ln(AGE)` |
| `BINDER` | `CEM + SLAG + FLY_ASH` |
| `FRAC_SLAG` | `SLAG / BINDER` |
| `FRAC_FLYASH` | `FLY_ASH / BINDER` |

**Hiperparâmetros** — exatamente os da Tabela 2:

| Modelo | Configuração |
|---|---|
| Random Forest | `n_estimators=500, max_depth=20, max_features=0.5, min_samples_leaf=2` |
| XGBoost | `n_estimators=500, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=1.0` |
| MLP | `hidden_layer_sizes=(128,64), learning_rate_init=0.01, alpha=0.001` |

**Avaliação** — `KFold(n_splits=10, shuffle=True, random_state=42)` com predições
out-of-fold, exatamente como descrito na Fase III. RMSE, MAE e R² reportados como
média ± desvio-padrão entre os folds.

**Probabilidade de conformidade** — `P(fck ≥ alvo) = 1 − Φ((alvo − ŷ)/σ)`, com
σ = RMSE de validação cruzada do modelo ativo. É a operacionalização direta do
problema levantado na Introdução: *"estimar a probabilidade de que um lote não
atinja o fck especificado"*.

---

## Verificação das métricas

O `train_model.py` imprime a Tabela 3 no terminal ao final da execução. Confira
contra o TCC:

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost | 3,614 | 2,409 | 0,949 |
| Random Forest | 4,633 | 3,300 | 0,916 |
| MLP | 5,489 | 4,069 | 0,882 |
| ElasticNet | 10,412 | 8,247 | 0,604 |
| OLS | 10,498 | 8,316 | 0,597 |

Pequenas diferenças (na 2ª casa decimal) são esperadas e vêm da semente aleatória
do k-fold e da versão das bibliotecas. Se algum modelo divergir de forma
expressiva, o ponto mais provável é o critério de remoção de duplicatas em
`src/pipeline.py::clean` — o dataset UCI original tem 25 linhas duplicadas.

---

## Publicar no Streamlit Cloud

1. Suba a pasta para um repositório no GitHub (inclua `data/concrete_raw.csv` e
   `models/*.pkl`, ou deixe o app baixar e treinar no primeiro acesso).
2. Em [share.streamlit.io](https://share.streamlit.io), aponte para `app.py`.
3. O `requirements.txt` já contém tudo o que é necessário.

---

## Notas técnicas

- O carregador de SHAP tem um *fallback* para o TreeSHAP nativo do XGBoost
  (`pred_contribs=True`), que produz valores idênticos e evita a incompatibilidade
  conhecida entre `shap < 0.50` e `xgboost >= 3.0`.
- `train_model.py` tenta três fontes para o dataset (UCI e dois espelhos). Se
  todas falharem — rede corporativa, por exemplo — baixe o `Concrete_Data.xls`
  manualmente, converta para CSV mantendo a ordem original das 9 colunas e salve
  em `data/concrete_raw.csv`.
- `models/ols_summary.txt` traz os coeficientes do OLS com erros-padrão robustos
  HC3, úteis para a discussão de sinais e significância da Fase II.

---

## Referência dos dados

Yeh, I. C. *Modeling of strength of high-performance concrete using artificial
neural networks*. Cement and Concrete Research, v. 28, n. 12, p. 1797–1808, 1998.
[UCI ML Repository — Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
