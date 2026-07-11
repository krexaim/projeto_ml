"""
Módulo de Treinamento e Persistência Dinâmica de Modelos.
"""

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "Model" / "config.yml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_feature_importance(model, feature_names, save_path, model_name):
    """Gera gráfico de importância e salva dinamicamente."""
    try:
        est = model.named_steps['model'] if isinstance(model, Pipeline) else model
        if hasattr(est, 'feature_importances_'):
            imp = pd.Series(est.feature_importances_, index=feature_names).nlargest(15)
            plt.figure(figsize=(8, 6))
            imp.sort_values().plot(kind='barh', color='skyblue')
            plt.title(f'Top 15 Features: {model_name}')
            plt.tight_layout()
            plt.savefig(save_path / "feature_importance.png")
            plt.close()
    except Exception as e:
        print(f"Feature importance não disponível para {model_name}: {e}")

def build_models(cfg, y_train=None):
    random_state = cfg["project"]["random_state"]
    models_cfg = cfg["models"]

    models = {}

    for model_name, model_info in models_cfg.items():
        if not model_info.get("enabled", True):
            continue

        params = model_info.get("params", {}).copy()
        params["random_state"] = random_state

        if model_info["class"] == "XGBClassifier":
            spw = params.get("scale_pos_weight", "auto")
            if isinstance(spw, str) and spw.lower() == "auto":
                if y_train is None:
                    raise ValueError(
                        "scale_pos_weight='auto' requer y_train: "
                        "build_models(cfg, y_train=y_train)"
                    )
                neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
                params["scale_pos_weight"] = neg / pos
                print(
                    f"scale_pos_weight calculado automaticamente: "
                    f"{params['scale_pos_weight']:.2f} (neg={neg}, pos={pos})"
                )
            model = xgb.XGBClassifier(**params)

        elif model_info["class"] == "RandomForestClassifier":
            model = RandomForestClassifier(**params)

        elif model_info["class"] == "LogisticRegression":
            base_model = LogisticRegression(**params)

            if model_info.get("use_scaler", False):
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", base_model)
                ])
            else:
                model = base_model

        else:
            raise ValueError(f"Modelo não suportado: {model_info['class']}")

        models[model_name] = model

    return models

def train_and_evaluate():
    cfg = load_config()
    target = cfg["project"]["target"]
    abt_dir = PROJECT_ROOT / cfg["paths"]["abt_dir"]
    model_base_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]
    abt_files = cfg["abt_files"]

    # Carregamento
    train_df = pd.read_parquet(abt_dir / abt_files["train"])
    val_df = pd.read_parquet(abt_dir / abt_files["val"])
    test_df = pd.read_parquet(abt_dir / abt_files["test"])

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    eval_dir = model_base_dir / "evaluation_data"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # X_test/y_test = holdout de verdade. Nunca usado em fit(), early
    # stopping ou seleção de threshold. É isso que evaluation.ipynb e
    # best_model.py devem carregar.
    joblib.dump(X_test, eval_dir / "X_test.pkl")
    joblib.dump(y_test, eval_dir / "y_test.pkl")
    joblib.dump(list(X_test.columns), eval_dir / "feature_names.pkl")
    joblib.dump(X_train.median(numeric_only=True), eval_dir / "medianas.pkl")

    # Guardamos o val separado, pra quem quiser auditar se o threshold
    # ficou colado demais nos dados em que foi tunado.
    joblib.dump(X_val, eval_dir / "X_val.pkl")
    joblib.dump(y_val, eval_dir / "y_val.pkl")

    modelos = build_models(cfg, y_train=y_train)

    resultados = []

    for nome, model in modelos.items():
        print(f"\n>>> Treinando: {nome}")
        
        # Treino específico
        if nome == 'XGBOOST':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        # Avaliação
        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        
        # Threshold 
        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        min_recall = cfg.get("evaluation", {}).get("min_recall_target", 0.70)

        # precision/recall têm 1 elemento a mais que thresholds
        idx_validos = np.where(recall[:-1] >= min_recall)[0]

        if len(idx_validos) > 0:
            melhor_idx = idx_validos[np.argmax(precision[idx_validos])]
            best_thresh = thresholds[melhor_idx]
        else:
            best_thresh = 0.5
            print(f"AVISO ({nome}): nenhum threshold atingiu recall >= {min_recall}; usando 0.5")
        
        resultados.append({'Modelo': nome, 'AUC': auc, 'Threshold': best_thresh})     
          
        # PERSISTÊNCIA DINÂMICA
        # O nome do arquivo .pkl agora usa o nome do modelo (ex: XGBOOST.pkl)
        save_path = model_base_dir / nome
        save_path.mkdir(parents=True, exist_ok=True)
        
        save_feature_importance(model, X_train.columns, save_path, nome)
        
        # Salvamento dinâmico: usa a variável 'nome' para criar o nome do arquivo
        joblib.dump(model, save_path / f"{nome.lower()}_model.pkl")
        
        with open(save_path / "threshold.txt", "w") as f:
            f.write(str(best_thresh))
            
    # Rankeamento
    pd.DataFrame(resultados).sort_values(by='AUC', ascending=False).to_csv(model_base_dir / "ranking_modelos.csv", index=False)
    print("\nProcesso finalizado. Modelos salvos dinamicamente em suas pastas.")

if __name__ == "__main__":
    train_and_evaluate()