"""
Seleciona o melhor modelo treinado com base em ROC AUC.

Este script:
1. Carrega os dados de avaliação salvos pelo train.py
2. Carrega os modelos salvos em Model/artifacts
3. Calcula ROC AUC para cada modelo
4. Salva o ranking atualizado
5. Salva o nome do modelo vencedor em best_model.txt
"""

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "Model" / "config.yml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_file(model_folder: Path) -> Path:
    model_files = list(model_folder.glob("*_model.pkl"))

    if not model_files:
        raise FileNotFoundError(f"Nenhum modelo encontrado em: {model_folder}")

    return model_files[0]


def select_best_model():
    cfg = load_config()

    artifacts_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
    eval_dir = artifacts_dir / "evaluation_data"

    X_test = joblib.load(eval_dir / "X_test.pkl")
    y_test = joblib.load(eval_dir / "y_test.pkl")

    resultados = []

    for model_folder in artifacts_dir.iterdir():
        if not model_folder.is_dir():
            continue

        if model_folder.name == "evaluation_data":
            continue

        model_path = get_model_file(model_folder)
        model = joblib.load(model_path)

        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        resultados.append({
            "Modelo": model_folder.name,
            "AUC": auc,
            "model_path": str(model_path.relative_to(PROJECT_ROOT))
        })

    df_resultados = (
        pd.DataFrame(resultados)
        .sort_values(by="AUC", ascending=False)
        .reset_index(drop=True)
    )

    ranking_path = artifacts_dir / "ranking_modelos_select_best.csv"
    df_resultados.to_csv(ranking_path, index=False)

    best_model = df_resultados.loc[0, "Modelo"]

    with open(artifacts_dir / "best_model.txt", "w", encoding="utf-8") as f:
        f.write(best_model)

    print("Ranking gerado:")
    print(df_resultados)

    print(f"\nMelhor modelo selecionado: {best_model}")
    print(f"Arquivo salvo em: {artifacts_dir / 'best_model.txt'}")


if __name__ == "__main__":
    select_best_model()