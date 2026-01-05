# Análise Preditiva de Evasão Escolar (Dropout)

Projeto de **Machine Learning para classificação** com foco na **previsão de evasão escolar**, utilizando **CatBoost** e técnicas de **explicabilidade integradas a LLMs**.

---

## Objetivo

O objetivo deste projeto é desenvolver um modelo de classificação capaz de prever a **evasão acadêmica (dropout)** de estudantes, a partir de dados acadêmicos e socioeconômicos.  
Além da predição, o projeto se preocupa com **interpretabilidade**, utilizando o método nativo do CatBoost (**PredictionValuesChange**) e um **LLM** para gerar explicações compreensíveis sobre a importância das features.

---

## Dataset

O projeto utiliza o dataset “Dropout or Academic Success”, disponível no Kaggle:

Autor: ankanhore545

Objetivo: Classificar estudantes em: Dropout, Enrolled, Graduate

## Etapas

Pré-processamento: Encoding de variáveis categóricas, Padronização de dados numéricos, Separação treino/teste

Modelo: Algoritmo (CatBoostClassifier)

Avaliação: Matriz de confusão, Precision, Recall e F1-score por classe

**Observação**: Análise focada na classe Dropout

----

## Explicabilidade

Utiliza o método **PredictionValuesChange**, nativo do CatBoost, que mede o impacto médio de cada feature na predição do modelo. Os valores de importância são enviados a um LLM (gemma:2b), que gera explicações em linguagem natural, tornando os resultados mais acessíveis para públicos não técnicos.

---

## Executando o projeto

O projeto inclui uma aplicação interativa com Streamlit, que permite visualizar as métricas do modelo.

### Passos para executar:

```bash
# 1. Clonar o repositório
git clone https://github.com/Andressa-Evelyn/analise_preditiva.git

# 2. Entrar na pasta
cd analise_preditiva

# 3. Criar ambiente virtual
python -m venv venv

# 4. Ativar ambiente virtual
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 5. Instalar dependências
pip install -r requirements.txt

# 6. Executar aplicação Streamlit
streamlit run streamlit_app.py
