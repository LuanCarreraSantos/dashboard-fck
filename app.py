"""
app.py — Dashboard interativo do TCC

"Modelos de Machine Learning para Predição e Avaliação da Resistência
 à Compressão do Concreto"
Luan Carrera Santos — MBA em Data Science and Analytics, USP/ESALQ, 2026

Executar:
    python train_model.py     (uma única vez)
    streamlit run app.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parent
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import (  # noqa: E402
    DERIVED_FEATURES, FEATURES, LABELS, MODEL_DIR, PROCESSED_CSV,
    RAW_FEATURES, TARGET, UNITS, build_features, load_json,
)

# ============================================================================
# Configuração
# ============================================================================
st.set_page_config(
    page_title="Predição de fck — TCC USP/ESALQ",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#1f6feb"
PALETTE = ["#1f6feb", "#e8590c", "#2f9e44", "#9c36b5", "#868e96"]

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
      div[data-testid="stMetricValue"] {font-size: 2.0rem;}
      .small-note {font-size: 0.82rem; color: #6b7280; line-height: 1.45;}
      h1, h2, h3 {letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)

SLUG = {
    "OLS": "ols",
    "ElasticNet": "elasticnet",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "MLP": "mlp",
}

# Compatibilidade de largura entre versões do Streamlit:
# `use_container_width` foi substituído por `width="stretch"`.
_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
W = {"width": "stretch"} if _HAS_WIDTH else {"use_container_width": True}


# ============================================================================
# Carregamento de artefatos
# ============================================================================
@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return load_json(MODEL_DIR / "metrics.json")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_CSV)


@st.cache_data(show_spinner=False)
def load_oof() -> pd.DataFrame:
    return pd.read_csv(MODEL_DIR / "oof_predictions.csv")


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return joblib.load(MODEL_DIR / f"{SLUG[name]}.pkl")


def _shap_explanation(model, X: pd.DataFrame):
    """TreeSHAP robusto a incompatibilidades de versão entre shap e xgboost.

    Caminho principal: shap.TreeExplainer.
    Fallback: TreeSHAP nativo do XGBoost (`pred_contribs=True`), que devolve
    exatamente os mesmos valores sem depender do parser interno do shap.
    """
    import shap
    try:
        return shap.TreeExplainer(model)(X)
    except Exception:  # noqa: BLE001
        import xgboost as xgb
        booster = model.get_booster()
        contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)
        return shap.Explanation(
            values=contribs[:, :-1],
            base_values=contribs[:, -1],
            data=X.values,
            feature_names=list(X.columns),
        )


@st.cache_resource(show_spinner="Calculando valores SHAP...")
def load_shap(_model, X: pd.DataFrame):
    return _shap_explanation(_model, X)


def artifacts_ready() -> bool:
    needed = [MODEL_DIR / "metrics.json", MODEL_DIR / "xgboost.pkl",
              MODEL_DIR / "oof_predictions.csv", PROCESSED_CSV]
    return all(p.exists() for p in needed)


if not artifacts_ready():
    # Primeira execução (ou primeiro boot do contêiner no Streamlit Cloud):
    # baixa o dataset UCI, limpa, treina os 5 modelos e grava os artefatos.
    st.title("🧱 Dashboard de predição de fck")
    st.info(
        "**Primeira execução.** Baixando o dataset UCI e treinando os cinco "
        "modelos com validação cruzada k = 10. Leva de 1 a 4 minutos e só "
        "acontece uma vez — depois o dashboard carrega instantaneamente."
    )

    with st.status("Executando o pipeline...", expanded=True) as status:
        try:
            import train_model
            train_model.main(on_step=lambda m: st.write(m))
        except Exception as exc:  # noqa: BLE001
            status.update(label="Falha no pipeline", state="error")
            st.error(f"**{type(exc).__name__}:** {exc}")
            st.markdown(
                "Se a falha foi no download, baixe o `Concrete_Data.xls` em "
                "[UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), "
                "converta para CSV mantendo a ordem original das 9 colunas e "
                "salve como `data/concrete_raw.csv`."
            )
            st.stop()
        status.update(label="Pipeline concluído", state="complete")

    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

META = load_meta()
DF = load_data()
OOF = load_oof()
METRICS = META["metrics"]
RANGES = META["feature_ranges"]

# Modelo de referência do TCC (Fase III): XGBoost. Cai para o melhor
# modelo por RMSE caso o artefato não exista.
REFERENCE = "XGBoost" if "XGBoost" in METRICS else META["best_model"]
SHAP_MODEL = REFERENCE if REFERENCE in ("XGBoost", "Random Forest") else None


# ============================================================================
# Sidebar — entrada do traço
# ============================================================================
PRESETS = {
    "Traço convencional (C30)": dict(CEM=320, SLAG=0, FLY_ASH=0, WATER=175,
                                     SP=2.0, COARSE_AGG=1000, FINE_AGG=780,
                                     AGE=28),
    "Alto desempenho (baixo w/b)": dict(CEM=450, SLAG=100, FLY_ASH=0,
                                        WATER=150, SP=12.0, COARSE_AGG=950,
                                        FINE_AGG=720, AGE=28),
    "Com adições minerais (eco)": dict(CEM=250, SLAG=120, FLY_ASH=90,
                                       WATER=175, SP=8.0, COARSE_AGG=960,
                                       FINE_AGG=760, AGE=28),
    "Idade elevada (90 dias)": dict(CEM=300, SLAG=90, FLY_ASH=60, WATER=170,
                                    SP=6.0, COARSE_AGG=980, FINE_AGG=770,
                                    AGE=90),
}

with st.sidebar:
    st.markdown("### 🧱 Traço candidato")
    preset = st.selectbox("Ponto de partida", list(PRESETS), index=0)
    if st.button("Aplicar preset", **W):
        for k, v in PRESETS[preset].items():
            st.session_state[f"in_{k}"] = float(v)

    st.divider()

    mix: dict = {}
    for feat in RAW_FEATURES:
        lo = float(RANGES[feat]["min"])
        hi = float(RANGES[feat]["max"])
        default = float(PRESETS[preset][feat])
        default = min(max(default, lo), hi)
        step = 1.0 if hi - lo > 100 else 0.1
        mix[feat] = st.slider(
            f"{LABELS[feat]} ({UNITS[feat]})",
            min_value=round(lo, 1), max_value=round(hi, 1),
            value=st.session_state.get(f"in_{feat}", round(default, 1)),
            step=step, key=f"in_{feat}",
        )

    st.divider()
    fck_alvo = st.number_input(
        "fck alvo (MPa)", min_value=5.0, max_value=90.0, value=30.0, step=1.0,
        help="Resistência característica especificada em projeto.",
    )

    _opts = list(METRICS)
    active = st.selectbox(
        "Modelo ativo", _opts, index=_opts.index(REFERENCE),
        help="XGBoost é o modelo de referência adotado na Fase III do TCC.",
    )

    st.divider()
    st.markdown(
        f"<div class='small-note'>{META['n_obs']} observações · "
        f"{META['n_features']} variáveis · CV k={META['n_folds']}<br>"
        f"Treinado em {META['trained_at']}</div>",
        unsafe_allow_html=True,
    )

X_user = build_features(mix)
model_best = load_model(active)
y_hat = float(model_best.predict(X_user)[0])

# Desvio-padrão preditivo do modelo ativo (usado nas probabilidades)
SIGMA = METRICS[active]["rmse_mean"]


# ============================================================================
# Cabeçalho
# ============================================================================
st.title("Predição e avaliação da resistência à compressão do concreto")
st.markdown(
    "<div class='small-note'>Luan Carrera Santos · Orientação: Anna Carolina "
    "Martins · MBA em Data Science and Analytics, USP/ESALQ · 2026<br>"
    "Base: UCI <i>Concrete Compressive Strength</i> (Yeh, 1998)</div>",
    unsafe_allow_html=True,
)
st.write("")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Preditor de traço",
    "🔍 Explicabilidade (SHAP)",
    "📊 Comparação de modelos",
    "📈 Exploração dos dados",
])


# ============================================================================
# TAB 1 — Preditor de traço
# ============================================================================
with tab1:
    prob = float(1 - stats.norm.cdf(fck_alvo, loc=y_hat, scale=SIGMA))
    lo95, hi95 = stats.norm.interval(0.95, loc=y_hat, scale=SIGMA)

    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1, 1])
    c1.metric("fck previsto", f"{y_hat:.1f} MPa",
              delta=f"{y_hat - fck_alvo:+.1f} vs alvo")
    c2.metric(f"P(fck ≥ {fck_alvo:.0f} MPa)", f"{prob * 100:.1f}%")
    c3.metric("Razão w/b", f"{X_user['W_B'].iloc[0]:.3f}")
    c4.metric("Aglomerante", f"{X_user['BINDER'].iloc[0]:.0f} kg/m³")

    if prob >= 0.95:
        st.success(f"Traço **conforme**: probabilidade de {prob * 100:.1f}% de "
                   f"atingir {fck_alvo:.0f} MPa (≥ 95%).")
    elif prob >= 0.80:
        st.warning(f"Traço **limítrofe**: {prob * 100:.1f}% de probabilidade. "
                   "Considere reduzir w/b ou aumentar a idade de controle.")
    else:
        st.error(f"Risco elevado de não conformidade: apenas "
                 f"{prob * 100:.1f}% de probabilidade de atingir o alvo.")

    st.write("")
    left, right = st.columns([1, 1])

    # ---- Gauge ------------------------------------------------------------
    with left:
        st.markdown("#### Resistência prevista")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=y_hat,
            number={"suffix": " MPa", "font": {"size": 42}},
            gauge={
                "axis": {"range": [0, max(90, float(RANGES[TARGET]["max"]))]},
                "bar": {"color": ACCENT, "thickness": 0.75},
                "steps": [
                    {"range": [0, 20], "color": "#f1f3f5"},
                    {"range": [20, 40], "color": "#e7f5ff"},
                    {"range": [40, 60], "color": "#d0ebff"},
                    {"range": [60, 90], "color": "#a5d8ff"},
                ],
                "threshold": {"line": {"color": "#e03131", "width": 4},
                              "thickness": 0.85, "value": fck_alvo},
            },
        ))
        fig.update_layout(height=290, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig, **W)
        st.markdown(
            f"<div class='small-note'>Intervalo de 95%: "
            f"<b>{lo95:.1f} – {hi95:.1f} MPa</b> "
            f"(σ = RMSE de validação cruzada = {SIGMA:.2f} MPa). "
            "A linha vermelha marca o fck alvo.</div>",
            unsafe_allow_html=True,
        )

    # ---- Distribuição preditiva ------------------------------------------
    with right:
        st.markdown("#### Distribuição preditiva e risco")
        grid = np.linspace(y_hat - 4 * SIGMA, y_hat + 4 * SIGMA, 400)
        dens = stats.norm.pdf(grid, y_hat, SIGMA)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=dens, mode="lines",
                                 line=dict(color=ACCENT, width=2),
                                 name="densidade"))
        below = grid <= fck_alvo
        fig.add_trace(go.Scatter(
            x=grid[below], y=dens[below], fill="tozeroy", mode="none",
            fillcolor="rgba(224,49,49,0.28)", name="não conformidade"))
        above = grid >= fck_alvo
        fig.add_trace(go.Scatter(
            x=grid[above], y=dens[above], fill="tozeroy", mode="none",
            fillcolor="rgba(47,158,68,0.25)", name="conformidade"))
        fig.add_vline(x=fck_alvo, line=dict(color="#e03131", dash="dash"))
        fig.update_layout(
            height=290, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="fck (MPa)", yaxis_title="densidade",
            legend=dict(orientation="h", y=1.12, x=0),
        )
        st.plotly_chart(fig, **W)
        st.markdown(
            "<div class='small-note'>A área vermelha é a probabilidade de o "
            "lote não atingir o fck especificado — a métrica de risco de "
            "não conformidade discutida na Introdução do TCC.</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ---- Curva de evolução com a idade ------------------------------------
    st.markdown("#### Evolução da resistência com a idade de cura")
    ages = np.unique(np.round(np.logspace(np.log10(1), np.log10(365), 60)))
    rows = []
    for a in ages:
        m = dict(mix)
        m["AGE"] = float(a)
        rows.append(build_features(m).iloc[0])
    curve_X = pd.DataFrame(rows)[FEATURES]
    curve_y = model_best.predict(curve_X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ages, y=curve_y, mode="lines",
                             line=dict(color=ACCENT, width=3),
                             name="fck previsto"))
    fig.add_hline(y=fck_alvo, line=dict(color="#e03131", dash="dash"),
                  annotation_text=f"alvo {fck_alvo:.0f} MPa")
    fig.add_trace(go.Scatter(x=[mix["AGE"]], y=[y_hat], mode="markers",
                             marker=dict(size=13, color="#e8590c",
                                         line=dict(width=2, color="white")),
                             name="traço atual"))
    fig.update_layout(height=330, xaxis_type="log",
                      xaxis_title="Idade de cura (dias, escala log)",
                      yaxis_title="fck previsto (MPa)",
                      margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, **W)

    # ---- Análise de sensibilidade ----------------------------------------
    st.markdown("#### Sensibilidade univariada")
    sens_feat = st.selectbox(
        "Variar", RAW_FEATURES, index=RAW_FEATURES.index("WATER"),
        format_func=lambda f: f"{LABELS[f]} ({UNITS[f]})",
    )
    grid_f = np.linspace(RANGES[sens_feat]["min"], RANGES[sens_feat]["max"], 60)
    rows = []
    for v in grid_f:
        m = dict(mix)
        m[sens_feat] = float(v)
        rows.append(build_features(m).iloc[0])
    sens_y = model_best.predict(pd.DataFrame(rows)[FEATURES])

    fig = px.line(x=grid_f, y=sens_y,
                  labels={"x": f"{LABELS[sens_feat]} ({UNITS[sens_feat]})",
                          "y": "fck previsto (MPa)"})
    fig.update_traces(line=dict(color=ACCENT, width=3))
    fig.add_vline(x=mix[sens_feat], line=dict(color="#e8590c", dash="dot"))
    fig.add_hline(y=fck_alvo, line=dict(color="#e03131", dash="dash"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, **W)
    st.markdown(
        "<div class='small-note'>Mantém os demais constituintes fixos e varia "
        "apenas a variável escolhida. Linha laranja: valor atual do traço.</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Predição por modelo + composição do traço ------------------------
    a, b = st.columns([1.15, 1])
    with a:
        st.markdown("#### Predição por modelo")
        preds = {}
        for name in METRICS:
            try:
                preds[name] = float(load_model(name).predict(X_user)[0])
            except Exception:  # noqa: BLE001
                continue
        pred_df = pd.DataFrame({
            "Modelo": list(preds),
            "fck previsto (MPa)": [round(v, 2) for v in preds.values()],
            "RMSE CV (MPa)": [round(METRICS[m]["rmse_mean"], 3) for m in preds],
        }).sort_values("RMSE CV (MPa)")
        st.dataframe(pred_df, hide_index=True, **W)
        st.markdown(
            "<div class='small-note'>A convergência entre os modelos de árvore "
            "é um indicativo de robustez da estimativa.</div>",
            unsafe_allow_html=True,
        )

    with b:
        st.markdown("#### Composição do traço")
        comp = {k: mix[k] for k in
                ["CEM", "SLAG", "FLY_ASH", "WATER", "SP",
                 "COARSE_AGG", "FINE_AGG"]}
        fig = px.pie(values=list(comp.values()),
                     names=[LABELS[k] for k in comp], hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="inside", textinfo="percent")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(font=dict(size=11)))
        st.plotly_chart(fig, **W)

    with st.expander("Variáveis derivadas deste traço"):
        st.dataframe(
            pd.DataFrame({
                "Variável": [LABELS[c] for c in DERIVED_FEATURES],
                "Símbolo": DERIVED_FEATURES,
                "Valor": [round(float(X_user[c].iloc[0]), 4)
                          for c in DERIVED_FEATURES],
                "Unidade": [UNITS[c] for c in DERIVED_FEATURES],
            }), hide_index=True, **W)


# ============================================================================
# TAB 2 — Explicabilidade SHAP
# ============================================================================
with tab2:
    st.markdown("### Interpretabilidade via SHAP")
    st.markdown(
        "<div class='small-note'>Valores SHAP (Lundberg & Lee, 2017) aplicados "
        f"ao modelo <b>{SHAP_MODEL or '—'}</b>, selecionado no TCC como modelo "
        "de referência. Decompõem cada predição em contribuições aditivas de "
        "cada variável de entrada.</div>", unsafe_allow_html=True)
    st.write("")

    if SHAP_MODEL is None:
        st.info("A análise SHAP em árvore requer XGBoost ou Random Forest.")
    else:
        import matplotlib.pyplot as plt
        import shap

        shap_model = load_model(SHAP_MODEL)
        X_bg = DF[FEATURES]
        sv = load_shap(shap_model, X_bg)
        y_hat_shap = float(shap_model.predict(X_user)[0])

        # ---- Explicação local ---------------------------------------------
        st.markdown("#### Explicação local — traço definido na barra lateral")
        sv_user = _shap_explanation(shap_model, X_user)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        shap.plots.waterfall(sv_user[0], max_display=11, show=False)
        plt.tight_layout()
        st.pyplot(fig, **W)
        plt.close(fig)
        st.markdown(
            f"<div class='small-note'>Partindo do valor esperado "
            f"E[f(x)] = {float(sv.base_values[0]):.2f} MPa, cada barra mostra "
            f"quanto a variável empurra a predição até {y_hat_shap:.2f} MPa. "
            "Vermelho aumenta o fck, azul reduz.</div>",
            unsafe_allow_html=True)

        st.divider()

        # ---- Importância global -------------------------------------------
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Importância global (Figura 19)")
            imp = pd.DataFrame({
                "Variável": FEATURES,
                "SHAP médio |·| (MPa)": np.abs(sv.values).mean(axis=0),
            }).sort_values("SHAP médio |·| (MPa)", ascending=True)
            fig = px.bar(imp, x="SHAP médio |·| (MPa)", y="Variável",
                         orientation="h",
                         color="SHAP médio |·| (MPa)",
                         color_continuous_scale="Blues")
            fig.update_layout(height=440, coloraxis_showscale=False,
                              margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, **W)

        with c2:
            st.markdown("#### Distribuição SHAP — beeswarm (Figura 20)")
            fig, ax = plt.subplots(figsize=(7, 5.6))
            shap.plots.beeswarm(sv, max_display=13, show=False)
            plt.tight_layout()
            st.pyplot(fig, **W)
            plt.close(fig)

        st.markdown(
            "<div class='small-note'>Cada ponto é uma observação. O eixo "
            "horizontal indica o impacto na predição; a cor indica o valor "
            "original da variável (azul = baixo, vermelho = alto).</div>",
            unsafe_allow_html=True)

        st.divider()

        # ---- Dependência parcial ------------------------------------------
        st.markdown("#### Dependência parcial SHAP (Figuras 21–23)")
        d1, d2 = st.columns(2)
        feat = d1.selectbox("Variável", FEATURES,
                            index=FEATURES.index("W_B"),
                            format_func=lambda f: f"{LABELS[f]}")
        inter = d2.selectbox("Colorir por (interação)", ["automático"] + FEATURES,
                             index=0,
                             format_func=lambda f: LABELS.get(f, f))

        idx = FEATURES.index(feat)
        color_vals = (sv.data[:, FEATURES.index(inter)]
                      if inter != "automático" else None)
        dep = pd.DataFrame({
            LABELS[feat]: sv.data[:, idx],
            "Contribuição SHAP (MPa)": sv.values[:, idx],
        })
        if color_vals is not None:
            dep[LABELS[inter]] = color_vals
            fig = px.scatter(dep, x=LABELS[feat], y="Contribuição SHAP (MPa)",
                             color=LABELS[inter],
                             color_continuous_scale="RdYlBu_r", opacity=0.8)
        else:
            fig = px.scatter(dep, x=LABELS[feat], y="Contribuição SHAP (MPa)",
                             opacity=0.7,
                             color_discrete_sequence=[ACCENT])
        fig.add_hline(y=0, line=dict(color="#adb5bd", dash="dot"))
        fig.add_vline(x=float(X_user[feat].iloc[0]),
                      line=dict(color="#e8590c", dash="dot"),
                      annotation_text="traço atual")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, **W)

        if feat == "W_B":
            st.markdown(
                "<div class='small-note'>O TCC identifica um limiar prático em "
                "<b>w/b ≈ 0,42</b>: acima desse valor a contribuição SHAP "
                "torna-se sistematicamente negativa.</div>",
                unsafe_allow_html=True)


# ============================================================================
# TAB 3 — Comparação de modelos
# ============================================================================
with tab3:
    st.markdown("### Desempenho comparativo (validação cruzada k = 10)")

    table = pd.DataFrame([
        {
            "Modelo": m,
            "RMSE (média)": round(v["rmse_mean"], 3),
            "RMSE (±dp)": round(v["rmse_std"], 3),
            "MAE (média)": round(v["mae_mean"], 3),
            "MAE (±dp)": round(v["mae_std"], 3),
            "R² (média)": round(v["r2_mean"], 3),
            "R² (±dp)": round(v["r2_std"], 3),
        }
        for m, v in METRICS.items()
    ]).sort_values("RMSE (média)").reset_index(drop=True)

    try:  # o gradiente de cor exige jinja2
        shown = table.style.background_gradient(subset=["R² (média)"],
                                                cmap="Blues")
    except (ImportError, AttributeError):
        shown = table
    st.dataframe(shown, hide_index=True, **W)
    st.markdown(
        "<div class='small-note'>Tabela 3 do TCC. Predições out-of-fold: cada "
        "observação é prevista por um modelo que não a incluiu no "
        "treinamento.</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Figura 12 — comparação visual das métricas")
    m1, m2, m3 = st.columns(3)
    for col, (metric, label, asc) in zip(
        (m1, m2, m3),
        [("rmse", "RMSE (MPa)", True), ("mae", "MAE (MPa)", True),
         ("r2", "R²", False)],
    ):
        d = pd.DataFrame({
            "Modelo": list(METRICS),
            "valor": [METRICS[m][f"{metric}_mean"] for m in METRICS],
            "dp": [METRICS[m][f"{metric}_std"] for m in METRICS],
        }).sort_values("valor", ascending=asc)
        fig = px.bar(d, x="Modelo", y="valor", error_y="dp",
                     color="Modelo", color_discrete_sequence=PALETTE,
                     labels={"valor": label})
        fig.update_layout(height=330, showlegend=False,
                          margin=dict(l=10, r=10, t=30, b=10),
                          title=dict(text=label, font=dict(size=14)))
        col.plotly_chart(fig, **W)

    st.markdown(
        "<div class='small-note'>Barras de erro = desvio-padrão entre os 10 "
        "folds. Menor RMSE/MAE e maior R² indicam melhor desempenho.</div>",
        unsafe_allow_html=True)

    st.divider()

    # ---- Observado vs predito + resíduos ----------------------------------
    st.markdown("#### Diagnóstico por modelo (Figuras 7–18)")
    sel = st.selectbox("Modelo", list(table["Modelo"]), index=0)
    obs = OOF["observado"].values
    pred = OOF[sel].values
    resid = obs - pred

    c1, c2 = st.columns(2)
    with c1:
        lim = [min(obs.min(), pred.min()) - 2, max(obs.max(), pred.max()) + 2]
        fig = px.scatter(x=obs, y=pred, opacity=0.55,
                         labels={"x": "fck observado (MPa)",
                                 "y": "fck predito (MPa)"},
                         color_discrete_sequence=[ACCENT])
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="identidade",
                                 line=dict(color="#e03131", dash="dash")))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                          title=dict(text=f"{sel} — predito vs observado",
                                     font=dict(size=14)),
                          showlegend=False)
        st.plotly_chart(fig, **W)

    with c2:
        fig = px.scatter(x=pred, y=resid, opacity=0.55,
                         labels={"x": "fck predito (MPa)",
                                 "y": "resíduo (obs − pred, MPa)"},
                         color_discrete_sequence=["#e8590c"])
        fig.add_hline(y=0, line=dict(color="#495057", dash="dash"))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                          title=dict(text=f"{sel} — resíduos vs predito",
                                     font=dict(size=14)))
        st.plotly_chart(fig, **W)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.histogram(x=resid, nbins=50, histnorm="probability density",
                           labels={"x": "resíduo (MPa)"},
                           color_discrete_sequence=[ACCENT])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                          title=dict(text="Distribuição dos resíduos",
                                     font=dict(size=14)), yaxis_title="densidade")
        st.plotly_chart(fig, **W)
    with c4:
        theo = stats.norm.ppf(
            (np.arange(1, len(resid) + 1) - 0.5) / len(resid),
            loc=resid.mean(), scale=resid.std(ddof=1))
        fig = px.scatter(x=theo, y=np.sort(resid), opacity=0.6,
                         labels={"x": "quantis teóricos",
                                 "y": "quantis amostrais"},
                         color_discrete_sequence=["#9c36b5"])
        fig.add_trace(go.Scatter(x=theo, y=theo, mode="lines",
                                 line=dict(color="#e03131", dash="dash")))
        fig.update_layout(height=350, showlegend=False,
                          margin=dict(l=10, r=10, t=30, b=10),
                          title=dict(text="QQ-plot dos resíduos",
                                     font=dict(size=14)))
        st.plotly_chart(fig, **W)

    st.markdown(
        f"<div class='small-note'>Resíduo médio: {resid.mean():+.3f} MPa · "
        f"desvio-padrão: {resid.std(ddof=1):.3f} MPa · "
        f"|resíduo| > 10 MPa em {(np.abs(resid) > 10).mean() * 100:.1f}% dos "
        "casos.</div>", unsafe_allow_html=True)

    with st.expander("Distribuição das métricas entre os 10 folds"):
        rows = []
        for m, v in METRICS.items():
            for i, (r, a, q) in enumerate(zip(v["folds"]["rmse"],
                                              v["folds"]["mae"],
                                              v["folds"]["r2"]), start=1):
                rows.append({"Modelo": m, "Fold": i, "RMSE": r,
                             "MAE": a, "R²": q})
        folds_df = pd.DataFrame(rows)
        met = st.radio("Métrica", ["RMSE", "MAE", "R²"], horizontal=True)
        fig = px.box(folds_df, x="Modelo", y=met, color="Modelo",
                     points="all", color_discrete_sequence=PALETTE)
        fig.update_layout(height=400, showlegend=False,
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, **W)


# ============================================================================
# TAB 4 — Exploração dos dados
# ============================================================================
with tab4:
    st.markdown("### Análise descritiva e padrões bivariados (Fase 1)")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Observações", f"{len(DF):,}".replace(",", "."))
    k2.metric("fck médio", f"{DF[TARGET].mean():.1f} MPa")
    k3.metric("fck (min–máx)",
              f"{DF[TARGET].min():.1f}–{DF[TARGET].max():.1f}")
    k4.metric("w/b médio", f"{DF['W_B'].mean():.3f}")

    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Figura 2 — Distribuição de fck")
        fig = px.histogram(DF, x=TARGET, nbins=45, marginal="box",
                           labels={TARGET: "fck (MPa)"},
                           color_discrete_sequence=[ACCENT])
        fig.add_vline(x=DF[TARGET].mean(), line=dict(color="#e03131",
                                                     dash="dash"),
                      annotation_text="média")
        fig.update_layout(height=370, margin=dict(l=10, r=10, t=20, b=10),
                          yaxis_title="frequência")
        st.plotly_chart(fig, **W)

    with c2:
        st.markdown("#### Figura 3 — Idade vs fck (escala log)")
        fig = px.scatter(DF, x="AGE", y=TARGET, opacity=0.5,
                         log_x=True, trendline="lowess",
                         labels={"AGE": "Idade de cura (dias, log)",
                                 TARGET: "fck (MPa)"},
                         color_discrete_sequence=[ACCENT],
                         trendline_color_override="#e03131")
        fig.update_layout(height=370, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, **W)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Figura 4 — Água vs fck")
        fig = px.scatter(DF, x="WATER", y=TARGET, opacity=0.5,
                         trendline="ols",
                         labels={"WATER": "Água (kg/m³)", TARGET: "fck (MPa)"},
                         color_discrete_sequence=["#e8590c"],
                         trendline_color_override="#212529")
        fig.update_layout(height=370, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, **W)

    with c4:
        st.markdown("#### Figura 5 — Razão água/aglomerante vs fck")
        fig = px.scatter(DF, x="W_B", y=TARGET, opacity=0.5, trendline="lowess",
                         color="AGE", color_continuous_scale="Viridis",
                         labels={"W_B": "w/b (adimensional)",
                                 TARGET: "fck (MPa)", "AGE": "idade"},
                         trendline_color_override="#e03131")
        fig.add_vline(x=0.42, line=dict(color="#e03131", dash="dot"),
                      annotation_text="w/b ≈ 0,42")
        fig.update_layout(height=370, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, **W)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### Figura 6 — Aglomerante total vs fck")
        fig = px.scatter(DF, x="BINDER", y=TARGET, opacity=0.5,
                         trendline="lowess",
                         labels={"BINDER": "Aglomerante (kg/m³)",
                                 TARGET: "fck (MPa)"},
                         color_discrete_sequence=["#2f9e44"],
                         trendline_color_override="#212529")
        fig.update_layout(height=370, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, **W)

    with c6:
        st.markdown("#### Matriz de correlação (Pearson)")
        corr = DF[FEATURES + [TARGET]].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10),
                          font=dict(size=9))
        st.plotly_chart(fig, **W)

    st.divider()
    st.markdown("#### Explorador livre")
    e1, e2, e3 = st.columns(3)
    xv = e1.selectbox("Eixo X", FEATURES, index=FEATURES.index("W_B"),
                      format_func=lambda f: LABELS[f])
    yv = e2.selectbox("Eixo Y", FEATURES + [TARGET],
                      index=len(FEATURES), format_func=lambda f: LABELS[f])
    cv = e3.selectbox("Cor", FEATURES + [TARGET],
                      index=FEATURES.index("AGE"),
                      format_func=lambda f: LABELS[f])
    fig = px.scatter(DF, x=xv, y=yv, color=cv, opacity=0.65,
                     color_continuous_scale="Viridis",
                     labels={xv: f"{LABELS[xv]} ({UNITS[xv]})",
                             yv: f"{LABELS[yv]} ({UNITS[yv]})",
                             cv: LABELS[cv]})
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, **W)

    with st.expander("Estatísticas descritivas"):
        desc = DF[FEATURES + [TARGET]].describe().T.round(3)
        desc.insert(0, "Variável", [LABELS[i] for i in desc.index])
        desc.insert(1, "Unidade", [UNITS[i] for i in desc.index])
        st.dataframe(desc, **W)

    st.download_button(
        "⬇️ Baixar dataset processado (CSV)",
        DF.to_csv(index=False).encode("utf-8"),
        file_name="concrete_processed.csv", mime="text/csv",
    )
