# 📦 Stock Price Predictor — MLOps Pipeline com DVC, LSTM e Flask

Este repositório contém um projeto focado em aprendizado prático de pipelines de Machine Learning, MLOps e versionamento com DVC.

> ⚠️ **Atenção:**  
> O objetivo deste projeto **não** é treinar o melhor modelo preditivo, nem obter as melhores métricas.  
> O foco principal é aprender a construir uma pipeline completa de Machine Learning, organizada, reprodutível e próxima de um cenário real de produção.

---

## 🎯 Objetivo do Projeto

- Construir uma pipeline completa de Machine Learning
- Aplicar conceitos fundamentais de MLOps
- Utilizar DVC para versionamento de dados e estágios da pipeline
- Criar uma arquitetura modular e reutilizável
- Disponibilizar o modelo treinado por meio de uma API Flask
- Praticar boas práticas de separação entre:
  - código
  - dados
  - modelos
  - métricas

> Este projeto deve ser entendido como um **exercício de engenharia**, não como uma solução final de previsão financeira.

---

## 🧠 Escopo do Modelo

- Modelo baseado em **LSTM**
- Previsão do **Volume de negociações**
- Entrada baseada em **janelas temporais**
- Features simples e intencionalmente limitadas

> 📌 **A escolha do modelo e das features é didática, não otimizada.**

---

## 🔄 Pipeline de Machine Learning (DVC)

A pipeline é composta pelas seguintes etapas:

1. **Data Loading**  
   Leitura e organização dos dados brutos

2. **Data Preprocessing**  
   Separação treino/teste  
   Normalização

3. **Feature Engineering**  
   Criação de janelas temporais para séries temporais

4. **Model Training**  
   Treinamento do modelo LSTM  
   Salvamento do modelo e do scaler do target

5. **Model Evaluation**  
   Avaliação em escala real (MAE, MSE, RMSE, R²)

6. **Model Serving**  
   API Flask para inferência

---

## 🚀 API Flask

A API permite realizar previsões utilizando o modelo treinado.

- O endpoint recebe um CSV com as features já processadas
- Retorna o Volume previsto em escala real
- A API é apenas para fins educacionais

**Para iniciar a aplicação:**
```bash
python -m app.main

```

## 🚀 API Flask

A API permite realizar previsões utilizando o modelo treinado.

- O endpoint recebe um CSV com as features já processadas
- Retorna o Volume previsto em escala real
- A API é apenas para fins educacionais

**Para iniciar a aplicação:**
```bash
python -m app.main

```

**Acesse no navegador:**
```bash
http://localhost:5001

```

## 📊 Métricas

As métricas de treino e avaliação são salvas automaticamente em:

- `metrics/training.json`
- `metrics/evaluation.json`

> Essas métricas existem apenas para validar o fluxo da pipeline, **não como benchmark de qualidade do modelo**.

---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- DVC
- Flask
- Git

---

## 📚 Motivação

Este projeto foi criado para:

- Aprender como estruturar pipelines reais de Machine Learning
- Entender o papel do DVC no versionamento de dados
- Praticar conceitos fundamentais de MLOps
- Construir um fluxo completo de:  
  `dados → modelo → avaliação → API`
- Simular um ambiente profissional de Data Science

> 👉 O foco está na **arquitetura e no processo**, não na performance do modelo.

---

## 📝 Observação Final

> Este projeto **não** deve ser utilizado para decisões financeiras reais.  
> Ele existe exclusivamente como material de estudo e prática em **MLOps e pipelines de Machine Learning**.