"""
pipeline.py — Núcleo reprodutível do TCC
"Modelos de Machine Learning para Predição e Avaliação da Resistência
 à Compressão do Concreto" — Luan Carrera Santos (MBA USP/ESALQ, 2026)

Responsabilidades:
  1. Aquisição do dataset UCI Concrete Compressive Strength (Yeh, 1998)
  2. Limpeza e verificação de consistência física
  3. Engenharia de variáveis (W_B, LOG_AGE, BINDER, FRAC_SLAG, FRAC_FLYASH)
  4. Definição dos 5 modelos com os hiperparâmetros da Tabela 2 do TCC

Este módulo é importado tanto por train_model.py quanto por app.py.
"""

from __future__ import annotations

import io
import json
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

RAW_CSV = DATA_DIR / "concrete_raw.csv"
PROCESSED_CSV = DATA_DIR / "concrete_processed.csv"

# ----------------------------------------------------------------------------
# Esquema de variáveis (Tabela 1 do TCC)
# ----------------------------------------------------------------------------
RAW_FEATURES = [
    "CEM",         # Cimento (component 1)          kg/m3
    "SLAG",        # Escória de alto-forno (c. 2)   kg/m3
    "FLY_ASH",     # Cinza volante (component 3)    kg/m3
    "WATER",       # Água (component 4)             kg/m3
    "SP",          # Superplastificante (c. 5)      kg/m3
    "COARSE_AGG",  # Agregado graúdo (c. 6)         kg/m3
    "FINE_AGG",    # Agregado miúdo (c. 7)          kg/m3
    "AGE",         # Idade de cura                  dias
]

DERIVED_FEATURES = ["W_B", "LOG_AGE", "BINDER", "FRAC_SLAG", "FRAC_FLYASH"]
FEATURES = RAW_FEATURES + DERIVED_FEATURES
TARGET = "FCK"

LABELS = {
    "CEM": "Cimento",
    "SLAG": "Escória de alto-forno",
    "FLY_ASH": "Cinza volante",
    "WATER": "Água",
    "SP": "Superplastificante",
    "COARSE_AGG": "Agregado graúdo",
    "FINE_AGG": "Agregado miúdo",
    "AGE": "Idade de cura",
    "W_B": "Razão água/aglomerante (w/b)",
    "LOG_AGE": "log(Idade)",
    "BINDER": "Aglomerante total",
    "FRAC_SLAG": "Fração de escória",
    "FRAC_FLYASH": "Fração de cinza volante",
    "FCK": "Resistência à compressão (fck)",
}

UNITS = {
    "CEM": "kg/m³", "SLAG": "kg/m³", "FLY_ASH": "kg/m³", "WATER": "kg/m³",
    "SP": "kg/m³", "COARSE_AGG": "kg/m³", "FINE_AGG": "kg/m³", "AGE": "dias",
    "W_B": "—", "LOG_AGE": "—", "BINDER": "kg/m³", "FRAC_SLAG": "—",
    "FRAC_FLYASH": "—", "FCK": "MPa",
}

RANDOM_STATE = 42
N_FOLDS = 10

# ----------------------------------------------------------------------------
# 1. Aquisição dos dados
# ----------------------------------------------------------------------------
_SOURCES = [
    ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
     "concrete/compressive/Concrete_Data.xls", "xls"),
    ("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/"
     "master/concrete.csv", "csv_r"),
    ("https://raw.githubusercontent.com/gchoi/Dataset/master/"
     "Concrete_Data.csv", "csv_uci"),
]

_R_COLS = ["cement", "slag", "ash", "water", "superplastic",
           "coarseagg", "fineagg", "age", "strength"]


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Renomeia qualquer variante conhecida do dataset para o esquema do TCC."""
    if df.shape[1] != 9:
        raise ValueError(f"Esperadas 9 colunas, recebidas {df.shape[1]}")
    df = df.copy()
    df.columns = RAW_FEATURES + [TARGET]
    return df.astype(float)


def download_dataset(force: bool = False) -> pd.DataFrame:
    """Baixa o dataset UCI. Usa cache local em data/concrete_raw.csv."""
    if RAW_CSV.exists() and not force:
        return _standardise(pd.read_csv(RAW_CSV))

    last_err = None
    for url, kind in _SOURCES:
        try:
            print(f"  -> tentando {url[:72]}...")
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (TCC-USP-ESALQ)"}
            )
            payload = urllib.request.urlopen(req, timeout=45).read()

            if kind == "xls":
                df = pd.read_excel(io.BytesIO(payload))
            else:
                df = pd.read_csv(io.BytesIO(payload))
                if kind == "csv_r":
                    df = df[_R_COLS]

            df = _standardise(df)
            df.to_csv(RAW_CSV, index=False)
            print(f"  [ok] {len(df)} observacoes salvas em {RAW_CSV}")
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"     falhou: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Nao foi possivel baixar o dataset automaticamente.\n"
        f"Ultimo erro: {last_err}\n\n"
        "Solucao manual: baixe 'Concrete_Data.xls' em\n"
        "  https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength\n"
        f"converta para CSV (9 colunas, ordem original) e salve em:\n  {RAW_CSV}"
    )


# ----------------------------------------------------------------------------
# 2. Limpeza e verificação de consistência
# ----------------------------------------------------------------------------
def clean(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Limpeza conforme a seção 'Processamento dos dados' do TCC."""
    log: dict = {"n_inicial": int(len(df))}

    # (a) registros essenciais incompletos
    df = df.dropna(subset=RAW_FEATURES + [TARGET])
    log["apos_dropna"] = int(len(df))

    # (b) duplicidades exatas
    df = df.drop_duplicates()
    log["apos_dedup"] = int(len(df))

    # (c) plausibilidade física
    mask = (
        (df["CEM"] > 0)
        & (df["WATER"] > 0)
        & (df["AGE"] > 0)
        & (df[TARGET] > 0)
        & (df[RAW_FEATURES] >= 0).all(axis=1)
    )
    log["removidos_implausiveis"] = int((~mask).sum())
    df = df[mask]

    # (d) tratamento conservador de discrepantes: w/b fisicamente impossível
    binder = df["CEM"] + df["SLAG"] + df["FLY_ASH"]
    wb = df["WATER"] / binder
    keep = (wb > 0.10) & (wb < 2.00)
    log["removidos_wb_extremo"] = int((~keep).sum())
    df = df[keep]

    log["n_final"] = int(len(df))
    df = df.reset_index(drop=True)

    if verbose:
        print(f"  Limpeza: {log['n_inicial']} -> {log['n_final']} observacoes "
              f"({log['n_inicial'] - log['apos_dedup']} duplicadas removidas)")
    return df, log


# ----------------------------------------------------------------------------
# 3. Engenharia de variáveis
# ----------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as 5 variáveis derivadas com interpretação físico-empírica."""
    df = df.copy()
    df["BINDER"] = df["CEM"] + df["SLAG"] + df["FLY_ASH"]           # (iii)
    df["W_B"] = df["WATER"] / df["BINDER"]                          # (i)
    df["LOG_AGE"] = np.log(df["AGE"])                               # (ii)
    df["FRAC_SLAG"] = df["SLAG"] / df["BINDER"]                     # (iv)
    df["FRAC_FLYASH"] = df["FLY_ASH"] / df["BINDER"]                # (v)
    return df


def build_features(mix: dict) -> pd.DataFrame:
    """Converte um traço informado pelo usuário em um vetor de features."""
    row = {k: float(mix[k]) for k in RAW_FEATURES}
    binder = max(row["CEM"] + row["SLAG"] + row["FLY_ASH"], 1e-9)
    row["BINDER"] = binder
    row["W_B"] = row["WATER"] / binder
    row["LOG_AGE"] = float(np.log(max(row["AGE"], 1e-9)))
    row["FRAC_SLAG"] = row["SLAG"] / binder
    row["FRAC_FLYASH"] = row["FLY_ASH"] / binder
    return pd.DataFrame([row])[FEATURES]


def load_processed(force_download: bool = False) -> pd.DataFrame:
    """Pipeline completo de dados: download -> limpeza -> feature engineering."""
    if PROCESSED_CSV.exists() and not force_download:
        return pd.read_csv(PROCESSED_CSV)
    raw = download_dataset(force=force_download)
    clean_df, _ = clean(raw)
    out = engineer(clean_df)
    out.to_csv(PROCESSED_CSV, index=False)
    return out


# ----------------------------------------------------------------------------
# 4. Definição dos modelos (Tabela 2 do TCC)
# ----------------------------------------------------------------------------
def get_models() -> dict:
    """Retorna os 5 modelos com os hiperparâmetros otimizados no TCC."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    return {
        "OLS": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5,
                                 max_iter=10000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=20, max_features=0.5,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.1, max_depth=4,
            subsample=0.8, colsample_bytree=1.0,
            objective="reg:squarederror", random_state=RANDOM_STATE,
            n_jobs=-1, tree_method="hist",
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 64), learning_rate_init=0.01,
                alpha=0.001, max_iter=2000, early_stopping=True,
                n_iter_no_change=25, random_state=RANDOM_STATE)),
        ]),
    }


BEST_MODEL = "XGBoost"


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
