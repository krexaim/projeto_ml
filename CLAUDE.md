# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este projeto

Projeto Final (Capstone) do MBA em Big Data & Analytics (LABDATA / FIA). Vale como avaliação da disciplina de Inteligência Artificial **e** como etapa do TCC. Simula o ciclo real de um projeto de Machine Learning numa empresa, seguindo o framework **CRISP-DM**. A base escolhida é **Home Credit Default Risk** (Kaggle — <https://www.kaggle.com/competitions/home-credit-default-risk/overview>): prever a probabilidade de um candidato pagar um empréstimo (variável alvo `TARGET`), usando dados alternativos para ampliar a inclusão financeira de pessoas sem histórico de crédito.

Premissa norteadora da avaliação: **"O modelo é apenas o meio, não o fim."** Cada decisão técnica precisa estar justificada pelo problema de negócio e pela ação que a empresa tomará a partir do resultado. A entrega final inclui um pitch executivo de 15 min (Demoday) focado em problema de negócio, insights da EDA e se o modelo ficou bom/ruim **e o porquê** — não apenas a métrica.

Métrica oficial de avaliação do modelo: **ROC AUC**.

## Estado atual

O repositório é hoje um **scaffold**: a árvore de pastas exigida já existe, mas os scripts (`.py`), notebooks (`.ipynb`) e arquivos de configuração estão **vazios** (placeholders). Não há ainda `requirements.txt`, suíte de testes ou comandos de build/lint configurados. Ao implementar, preencha estes arquivos respeitando as regras abaixo — não reorganize a estrutura.

Alguns placeholders têm nomes provisórios que **não** são os nomes finais e existem só para marcar a pasta no Git: `Dados/brutecleanabt` (será desdobrado em `raw_data.csv` / `clean_data.csv` / `abt.csv`) e `MLOps/app/fastAPIorstreamlit` (será o serviço FastAPI **ou** Streamlit). Renomeie-os ao implementar; não os trate como artefatos reais.

## Arquitetura: fluxo de dados (a regra central)

O coração do projeto é um pipeline em camadas. Cada estágio lê a saída do anterior; entender qualquer script exige conhecer essa cadeia:

```
raw_data.csv  --(DataPipeline/data_sanitization.py)-->  clean_data.csv
clean_data.csv  --(DataPipeline/abt_transform.py)----->  abt.csv  (ABT = Analytical Base Table)
abt.csv  --(Model/train.py)--------------------------->  modelo treinado
modelo + dados novos  --(Model/predict.py)------------>  predições
```

Os três artefatos de dados (`raw_data.csv`, `clean_data.csv`, `abt.csv`) vivem em `/Dados`. A base original tem múltiplas tabelas — `application_train.csv` é a principal (uma linha por empréstimo); `bureau.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv` e `installments_payments.csv` são históricas e precisam ser agregadas para uma linha por empréstimo durante a construção da ABT.

Mapa de diretórios:

- **`/Dados`** — artefatos de dados (`raw_data.csv` → `clean_data.csv` → `abt.csv`). Não versionar CSVs pesados; usar `.gitignore`.
- **`/DataPipeline`** — manipulação de dados em **scripts `.py`** (não notebooks):
  - `data_sanitization.py` — limpeza (raw → clean)
  - `abt_transform.py` — construção da ABT (clean → abt)
  - `config.yml` — configuração do pipeline (caminhos, metadados, variável alvo)
  - `exp_analysis.ipynb` — **único notebook permitido aqui**; EDA feita **sobre o dado limpo** (`clean_data.csv`), não sobre o bruto.
- **`/Model`** — `train.py` (treinamento), `evaluation.ipynb` (notebook de avaliação + interpretabilidade) e seu **próprio arquivo de configuração** (o brief exige config em `/DataPipeline` **e** em `/Model`). `predict.py` aqui é entregável da etapa individual.
- **`/MLOps`** — etapa individual (deploy): `pipeline_orchestration.py` (orquestração), `app/` (serviço Streamlit ou API) e `Readme.md` próprio com o desenho da arquitetura. `docker-compose.yml` na raiz.

## Regras inegociáveis do projeto

Estas regras vêm da especificação de entrega e têm precedência sobre conveniência de código:

1. **Proibido hardcoding.** Nenhum caminho, parâmetro, nome de coluna, variável alvo ou metadado pode ficar "chumbado" nos scripts. Tudo deve ser lido de arquivos de configuração (`config.yml` já reservado em `DataPipeline/`; também são aceitos `.json`/`.env`). Ao escrever qualquer script, primeiro defina o parâmetro no config e leia-o.

2. **Pipeline em scripts `.py`, não em notebooks.** Toda manipulação de dados (limpeza e ABT) é código `.py` modular, para ser orquestrável depois. Notebooks existem **apenas** para EDA (`exp_analysis.ipynb`) e avaliação (`evaluation.ipynb`).

3. **EDA é sobre o dado limpo.** A análise exploratória roda em cima de `clean_data.csv`, depois do `data_sanitization.py` — nunca sobre o raw.

4. **Código pensado para MLOps desde já.** A fase seguinte fará deploy com Docker, Airflow (orquestrando do dado bruto até a ABT) e FastAPI/Streamlit para servir o modelo. Por isso o código de pipeline precisa ser modular, parametrizado e executável de forma independente (cada estágio como entrypoint chamável), não acoplado a estado de notebook.

## Cronograma, entregáveis e avaliação (oficial — brief LABDATA FIA)

Datas no formato DD/MM (edição 2026).

### Cronograma do trabalho em grupo
- **Dia 1 — Kickoff (22/06):** apresentação do contexto, objetivos e aula-exemplo.
- **Dia 2 — Dados (29/06):** definir o problema, o impacto no negócio e as métricas de sucesso; EDA (padrões, qualidade da base, nulos, inconsistências, comportamento das variáveis); estruturar a **ABT**.
- **Dia 3 — Modelo (06/07):** desenvolver a modelagem e avaliar em cenário de teste.
- **Dia 4 — Narrativa de negócio:** análise crítica do desempenho (limitações, erros, vieses, cenários de falha) + storytelling + **Demoday** (pitch de 15 min para banca/empresa).
- **Entrega final do grupo: 13/07.**
- **Entrega final individual: 15/07.**

### Entregáveis do grupo
- **A) Pitch da solução** (Demoday).
- **B) PowerPoint com 5 slides**, nesta ordem exata: (1) problema de negócio; (2) análise exploratória; (3) ABT; (4) modelo — técnica do ciclo de desenvolvimento (modelo de ML, controle de overfitting, hiperparâmetros); (5) avaliação — performance do algoritmo + explicabilidade.
- **C) Código no Git** seguindo a estrutura obrigatória (ver "Arquitetura"). Cada integrante mantém o **seu próprio repositório**. `/DataPipeline` e `/Model` precisam cada um do seu arquivo de configuração. `README.md` deve conter: descrição do projeto + objetivo de negócio + resumo da metodologia + instruções de como treinar o modelo.

### Entregável individual (deploy / arquitetura)
Desenvolvimentos adicionais no mesmo repositório:
- `/Model/predict.py`.
- `/MLOps/`: `Readme.md` (desenho da arquitetura com componentes + próximos passos de monitoramento e automação), `docker-compose`, `pipeline_orchestration.py` e `/app` (resultado servido via Streamlit ou API). Acrescentar ao README as instruções de execução do serviço de predição.
- Propor e demonstrar: (a) arquitetura funcional completa, da origem dos dados ao deploy do modelo como serviço de predição; (b) infraestrutura via `docker-compose` (a orquestração `bruta → clean → abt → Model` roda em Airflow); (c) **monitoramento** de dados e modelo em produção — falhas, perda de performance e mudança de comportamento dos dados (*data drift*); (d) **ações automatizadas** acionadas pelas previsões do modelo, conectando ML + automação + agentes de IA.
- **Esteja preparado para modificar o projeto ao vivo na banca** (a professora sorteia e pede para alterar parâmetros do modelo ou da orquestração na hora).

### Critérios de avaliação
- **Grupo** — *BUSINESS KNOWLEDGE* (alinhamento estratégico; viabilidade operacional) e *RESULTS* (análise e diagnóstico; métricas e governança). RESULTS é a defesa do **"podemos confiar que vai funcionar?"**: apresente métricas de rastreabilidade da confiança do modelo, de resultado de negócio e de conformidade de governança.
- **Individual** — *FUNDAMENTALS* (fundamentação teórica; mecanismos técnicos) e *CODING* (qualidade de código; tratamento de dados; construção de pipeline).
- **Pesos (TCC):** Pitch 20% (nota do grupo) · Entregável técnico 20% · Banca final 60% (notas individuais). O foco maior da avaliação está na entrega final e na defesa da solução.

## Dicas e Diretrizes da Professora para o Projeto Final (MBA)

Compilado das exigências e conselhos da professora para o sucesso nas etapas em grupo e individuais.

### 1. Contexto de Negócio e Storytelling
- **O modelo é o meio, não o fim:** a solução não é o modelo pelo modelo. Defina bem o contexto de negócio — qual a dor ou oportunidade da empresa e como a saída do modelo resolve isso.
- **Visão executiva:** na apresentação do *Demoday*, foco executivo. Explique o "porquê" (o problema), o "que" e os "resultados". Os detalhes técnicos minuciosos (o "como") são avaliados nos repositórios e na etapa individual.
- **Ciclo completo:** mostre o discurso de ponta a ponta. Se o modelo prevê *churn*, o que o negócio fará com isso? (ex: acionar gerente comercial, ligar para o cliente).

### 2. Dados, Modelagem e Avaliação
- **Sustente seus resultados:** não importa se a acurácia foi 99% ou "uma porcaria" — o fundamental é saber explicar o **porquê** o resultado foi bom ou ruim.
- **A EDA justifica o modelo:** é a análise exploratória que sustenta os resultados, cria hipóteses e explica as métricas finais.
- **Resultados negativos são válidos:** se as variáveis não correlacionam ou os dados são insuficientes e o modelo falha, isso é um resultado válido — desde que você prove o motivo pela EDA.

### 3. Qualidade de Código e Repositório (GitHub)
- **Repositório individual:** mesmo com o código do grupo sendo igual, **cada aluno** deve ter o próprio repositório Git com o trabalho do grupo.
- **Sem caminhos/variáveis "chumbados" (hardcoded):** nunca engesse nomes de variáveis ou paths no script. Use arquivos de configuração (`.json`, `.csv`, `.yml` ou módulo `.py`) para passar parâmetros e metadados.
- **Scripts vs. Notebooks:** limpeza, padronização, criação da ABT e deploy usam **scripts** `.py`, não notebooks. Jupyter fica restrito à EDA e à avaliação do modelo.
- **Reprodutibilidade:** o repositório precisa ter `requirements.txt` e `README.md` com instruções claras para a professora treinar e reproduzir o modelo do zero na máquina dela.

### 4. Apresentação em Grupo (Demoday)
- **Gestão de tempo:** exatos 15 minutos. Ensaie. Se o tempo acabar e ainda estiverem no primeiro slide, a apresentação é cortada no meio.
- **Divisão de falas:** o grupo define quem apresenta (uma pessoa ou todas) e quem responde dúvidas. A nota desta etapa é coletiva.

### 5. Etapa Individual (Banca e Deploy)
- **Demonstração funcional:** na banca, você executa o projeto de ponta a ponta — orquestração rodando no Docker com Airflow até a resposta gerada via FastAPI ou Streamlit. Não são só slides.
- **Modificações ao vivo:** esteja preparado para alterar o código na hora. A professora sorteia e pede para mudar algum parâmetro do modelo ou da orquestração ao vivo, para ver como o sistema se comporta.
- **Evite "copiar e colar" cego:** o pipeline pode ser semelhante entre membros do grupo, mas cópias sem sentido (ex: deixar o path da máquina do colega no seu código) somadas à incapacidade de explicar o próprio código resultam em reprovação sumária (Sub).

## Idioma

Comunicação, comentários e documentação do projeto em **português (Brasil)**.
