import joblib
import pandas as pd
from pathlib import Path

def carregar_modelo(model_dir: Path, nome_modelo: str):
    model_path = model_dir / nome_modelo / f"{nome_modelo.lower()}_model.pkl"
    medianas_path = model_dir / "evaluation_data" / "medianas.pkl"
    threshold_path = model_dir / nome_modelo / "threshold.txt"
    features_path = model_dir / "evaluation_data" / "feature_names.pkl"
    
    if nome_modelo is None:
        with open(model_dir / "best_model.txt", encoding="utf-8") as f:
            nome_modelo = f.read().strip()
        
    model = joblib.load(model_path)
    medianas = joblib.load(medianas_path)
    feature_names = joblib.load(features_path)
    
    # --- AJUSTE AQUI: Acessar o passo do classificador dentro do Pipeline ---
    # Geralmente o nome do passo no Pipeline é 'model' ou 'classifier'
    # Verifique qual nome você usou no train.py (ex: Pipeline([('scaler', ...), ('model', XGBoost())]))
    try:
        xgb_model = model.named_steps['model'] 
    except KeyError:
        # Se você não nomeou o passo, tente pegar o último passo do pipeline
        xgb_model = list(model.named_steps.values())[-1]

    # Agora extraímos as importâncias do objeto XGBoost real
    importancias = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("--- Top 10 Features Mais Importantes ---")
    print(importancias.head(10))
    
    with open(threshold_path, "r") as f:
        threshold = float(f.read())
            
    return model, threshold, medianas, feature_names

def prever_risco(model, threshold, medianas, feature_names, dados_input):
    df_modelo = pd.DataFrame([medianas], columns=feature_names).copy()
    
    for col in dados_input.columns:
        if col in df_modelo.columns:
            df_modelo[col] = float(dados_input[col].iloc[0])
    
    probs = model.predict_proba(df_modelo)
    prob_final = float(probs[0, 1])
    return prob_final, bool(prob_final >= threshold)