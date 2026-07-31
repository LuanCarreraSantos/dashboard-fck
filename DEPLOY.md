# Publicar o dashboard no Streamlit Community Cloud

Roteiro sem instalar nada na máquina — tudo pelo navegador. Ao final você terá
um link público do tipo `https://seu-app.streamlit.app` para citar no TCC e abrir
na apresentação.

Tempo estimado: 15 minutos.

---

## Passo 1 — Conta no GitHub

Se ainda não tiver: [github.com/signup](https://github.com/signup). Só e-mail e senha.

---

## Passo 2 — Criar o repositório

1. Vá em [github.com/new](https://github.com/new).
2. **Repository name**: `dashboard-fck` (ou o nome que preferir).
3. Deixe em **Public** — o Streamlit Cloud gratuito exige repositório público.
4. **Não** marque "Add a README file". O repositório precisa nascer vazio.
5. Clique em **Create repository**.

---

## Passo 3 — Enviar os arquivos

Na página do repositório recém-criado, clique em **uploading an existing file**
(o link no meio da tela).

Abra a pasta `dashboard_fck` no Explorer e arraste para a área de upload:

- `app.py`
- `train_model.py`
- `requirements.txt`
- `README.md`
- a pasta `src` inteira

Confira que aparece `src/pipeline.py` na lista de arquivos a enviar — se aparecer
só `pipeline.py`, apague e arraste a **pasta** `src`, não o arquivo de dentro dela.

Escreva qualquer coisa em *Commit changes* e clique em **Commit changes**.

> As pastas `data` e `models` não precisam ir: o app baixa o dataset e treina os
> modelos sozinho no primeiro acesso. A pasta `.streamlit` também é opcional —
> ela só define as cores do tema.

---

## Passo 4 — Publicar

1. Acesse [share.streamlit.io](https://share.streamlit.io) e entre com **Continue with GitHub**.
2. Autorize o acesso quando o GitHub pedir.
3. Clique em **Create app** → **Deploy a public app from GitHub**.
4. Preencha:
   - **Repository**: `seu-usuario/dashboard-fck`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Em **Advanced settings**, escolha **Python 3.12**.
6. Clique em **Deploy**.

A primeira construção leva de 5 a 10 minutos (instalação das bibliotecas). Depois
o app abre e mostra a tela *"Primeira execução"*, treinando os cinco modelos por
mais 1 a 4 minutos. A partir daí o dashboard carrega direto.

---

## O que esperar depois

**Hibernação.** Apps gratuitos hibernam após alguns dias sem acesso. O primeiro
visitante depois disso espera o servidor acordar e refazer o treino. Antes da
apresentação, abra o link com uns 10 minutos de antecedência para "esquentar".

**Atualizar o app.** Edite o arquivo direto no GitHub (ícone de lápis) e faça
commit — o Streamlit Cloud redeploya sozinho em ~1 minuto.

**Ver as métricas da Tabela 3.** No canto inferior direito do app, *Manage app* →
o painel de logs mostra a saída do `train_model.py`, incluindo RMSE, MAE e R² de
cada modelo. É ali que você confere se o XGBoost bate os 3,614 MPa do TCC.

---

## Se algo falhar

| Sintoma | Causa provável |
|---|---|
| `ModuleNotFoundError: pipeline` | A pasta `src` não subiu como pasta. Reenvie arrastando o diretório inteiro. |
| Erro de instalação nas dependências | Python errado nas *Advanced settings*. Use 3.12. |
| Falha no download do dataset | Raro — o pipeline tenta três fontes. Nesse caso, envie o CSV manualmente como `data/concrete_raw.csv` (9 colunas, ordem original do UCI). |
| App fica em "Your app is in the oven" | Só espere; a primeira construção é lenta mesmo. |

Copie a mensagem de erro do painel de logs se travar em algum ponto.
