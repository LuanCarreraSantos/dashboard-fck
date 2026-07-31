"""
train_model.py — Treino e avaliação reprodutíveis (Fases I, II e III do TCC)

Execute UMA vez antes de abrir o dashboard:

    python train_model.py

Gera em models/:
    xgboost.pkl            modelo final (ajustado em 100% dos dados)
    random_forest.pkl      "
    mlp.pkl / ols.pkl / elasticnet.pkl
    metrics.json           Tabela 3 (RMSE, MAE, R², média ± dp em k=10)
    oof_predictions.csv    predições out-of-fold de todos os modelos
    ols_summary.txt        coeficientes OLS com erros-padrão robustos (HC3)
    clean_log.json         auditoria da limpeza
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline import (  # noqa: E402
    DATA_DIR, FEATURES, MODEL_DIR, N_FOLDS, PROCESSED_CSV, RANDOM_STATE,
    RAW_FEATURES, TARGET, clean, download_dataset, engineer, get_models,
    save_json,
)

SLUG = {
    "OLS": "ols",
    "ElasticNet": "elasticnet",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "MLP": "mlp",
}


def cross_validate(name, model, X, y, kf):
    """Validação cruzada k-fold com predições out-of-fold e métricas por fold."""
    oof = np.zeros(len(y))
    per_fold = {"rmse": [], "mae": [], "r2": []}

    for train_idx, test_idx in kf.split(X):
        est = clone(model)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = est.predict(X.iloc[test_idx])
        oof[test_idx] = pred

        yt = y.iloc[test_idx]
        per_fold["rmse"].append(float(np.sqrt(mean_squared_error(yt, pred))))
        per_fold["mae"].append(float(mean_absolute_error(yt, pred)))
        per_fold["r2"].append(float(r2_score(yt, pred)))

    summary = {
        "rmse_mean": float(np.mean(per_fold["rmse"])),
        "rmse_std": float(np.std(per_fold["rmse"], ddof=1)),
        "mae_mean": float(np.mean(per_fold["mae"])),
        "mae_std": float(np.std(per_fold["mae"], ddof=1)),
        "r2_mean": float(np.mean(per_fold["r2"])),
        "r2_std": float(np.std(per_fold["r2"], ddof=1)),
        "folds": per_fold,
    }
    return oof, summary


def fit_ols_inference(X, y):
    """OLS com statsmodels e erros-padrão robustos HC3 (Fase II do TCC)."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return None
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit(cov_type="HC3")
    return res


def main(force_download: bool = False, on_step=None) -> dict:
    """Executa o pipeline completo.

    on_step: callback opcional `f(mensagem: str)` para reportar progresso em
             interfaces gráficas (usado pelo bootstrap do app.py).
    """
    def report(msg: str) -> None:
        if on_step is not None:
            on_step(msg)

    t0 = time.time()
    print("=" * 74)
    print("TCC — Predição da resistência à compressão do concreto (fck)")
    print("Treino e validação cruzada k = 10")
    print("=" * 74)

    # ---------------------------------------------------------------- Fase 0
    print("\n[1/5] Aquisição e limpeza dos dados")
    report("Baixando o dataset UCI Concrete Compressive Strength...")
    raw = download_dataset(force=force_download)
    clean_df, log = clean(raw)
    df = engineer(clean_df)
    df.to_csv(PROCESSED_CSV, index=False)
    save_json(log, MODEL_DIR / "clean_log.json")
    print(f"  Dataset processado: {df.shape[0]} obs × {df.shape[1]} colunas")
    report(f"Dados prontos: {df.shape[0]} observações após a limpeza.")

    X = df[FEATURES]
    y = df[TARGET]

    # ---------------------------------------------------------------- Fases II/III
    print(f"\n[2/5] Validação cruzada de 5 modelos (k = {N_FOLDS})")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()

    metrics, oof_table = {}, {"observado": y.values}
    for i, (name, model) in enumerate(models.items(), start=1):
        t = time.time()
        report(f"Validação cruzada {i}/{len(models)} — {name}...")
        oof, summary = cross_validate(name, model, X, y, kf)
        metrics[name] = summary
        oof_table[name] = oof
        line = (f"  {name:<14} RMSE {summary['rmse_mean']:6.3f} ± "
                f"{summary['rmse_std']:.3f} | MAE {summary['mae_mean']:6.3f} | "
                f"R² {summary['r2_mean']:.3f}   ({time.time() - t:.1f}s)")
        print(line)
        report(f"{name}: RMSE {summary['rmse_mean']:.3f} MPa · "
               f"R² {summary['r2_mean']:.3f}")

    pd.DataFrame(oof_table).to_csv(MODEL_DIR / "oof_predictions.csv", index=False)

    # ---------------------------------------------------------------- Ajuste final
    print("\n[3/5] Ajuste final em 100% dos dados")
    report("Ajustando os modelos finais em 100% dos dados...")
    for name, model in models.items():
        est = clone(model)
        est.fit(X, y)
        joblib.dump(est, MODEL_DIR / f"{SLUG[name]}.pkl")
        print(f"  salvo models/{SLUG[name]}.pkl")

    # ---------------------------------------------------------------- Inferência OLS
    print("\n[4/5] Inferência estatística do OLS (HC3)")
    res = fit_ols_inference(X, y)
    if res is not None:
        (MODEL_DIR / "ols_summary.txt").write_text(str(res.summary()),
                                                   encoding="utf-8")
        print("  salvo models/ols_summary.txt")
    else:
        print("  statsmodels não instalado — etapa ignorada")

    # ---------------------------------------------------------------- Metadados
    print("\n[5/5] Consolidação das métricas")
    ranking = sorted(metrics, key=lambda k: metrics[k]["rmse_mean"])
    payload = {
        "metrics": metrics,
        "ranking": ranking,
        "best_model": ranking[0],
        "n_obs": int(len(df)),
        "n_features": len(FEATURES),
        "features": FEATURES,
        "raw_features": RAW_FEATURES,
        "n_folds": N_FOLDS,
        "random_state": RANDOM_STATE,
        "clean_log": log,
        "feature_ranges": {
            c: {"min": float(df[c].min()), "max": float(df[c].max()),
                "mean": float(df[c].mean()), "p50": float(df[c].median())}
            for c in FEATURES + [TARGET]
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(payload, MODEL_DIR / "metrics.json")

    elapsed = time.time() - t0
    print("\n" + "=" * 74)
    print(f"Concluído em {elapsed:.1f}s. Melhor modelo: {ranking[0]} "
          f"(RMSE {metrics[ranking[0]]['rmse_mean']:.3f} MPa, "
          f"R² {metrics[ranking[0]]['r2_mean']:.3f})")
    print("Agora rode:  streamlit run app.py")
    print("=" * 74)
    report(f"Concluído em {elapsed:.0f}s. Melhor modelo: {ranking[0]}.")
    return payload


if __name__ == "__main__":
    main(force_download="--force-download" in sys.argv)
