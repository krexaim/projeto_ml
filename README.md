# Projeto final do módulo de AI/ML da LabData FIA: Home Credit

- **Contexto**: O Home Credit, empresa que oferece empréstimos para clientes com pouco ou nenhum histórico de crédito, enfrenta o desafio de equilibrar a concessão de crédito com a minimização da inadimplência.
- A "Dor" do Negócio: A análise manual ou baseada em regras simples de crédito é lenta e falha em capturar padrões complexos de risco, resultando em perda de receita (clientes bons negados) ou aumento de prejuízo (clientes inadimplentes aprovados).
- **Objetivo**: Desenvolver um modelo de Machine Learning preditivo capaz de classificar a probabilidade de inadimplência de um solicitante, permitindo uma tomada de decisão automatizada, mais rápida e mais precisa.
- **Impacto Esperado**: Aumentar a rentabilidade da carteira de crédito e promover a inclusão financeira de forma sustentável para o negócio.


## Como treinar o modelo

### 1. Ambiente virtual

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Dados

Baixe a base do Kaggle (<https://www.kaggle.com/competitions/home-credit-default-risk/overview>)
e coloque os CSVs **brutos** na pasta `Dados/`:
`application_train.csv`, `bureau.csv`, `previous_application.csv`.

### 3. Pipeline (do dado bruto ao modelo)

Execute a partir da raiz do projeto, nesta ordem:
```bash
python DataPipeline/data_sanitization.py   # raw -> Dados/clean_data.csv (+ auxiliares)
python DataPipeline/abt_transform.py       # clean -> Dados/abt.csv
python Model/train.py                       # abt -> Model/artifacts/modelo_risco_credito.pkl
```

Os parâmetros (caminhos, colunas, hiperparâmetros, threshold) ficam nos arquivos de
configuração `DataPipeline/config.yml` e `Model/config.yml` — não é preciso editar os scripts.
