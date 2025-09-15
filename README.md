# Analise-Preditiva-Enem-2023
Projeto de machine learning utilizando dados fornecidos pelo INEP do ENEM 2023, com objetivo de desenvolver um algoritmo capaz de predizer as notas dos participantes a partir de variáveis socioeconômicas como renda familiar, escolaridade dos pais e tipo de escola frequentada.

Os dados são oficiais e foram disponibilizados pelo INEP, como o próprio site informa: "Os microdados reúnem um conjunto de informações detalhadas sobre o Enem 2023, permitindo que gestores, pesquisadores e instituições realizem análises de dados para subsidiar diagnósticos, estudos e pesquisas, bem como o acompanhamento de estatísticas e informações educacionais."

## O projeto foi dividido em 2 partes: Limpeza/Exploração dos dados e Modelagem.

## 1. Limpeza e Exploração dos Dados

Este notebook apresenta uma análise completa dos microdados do ENEM 2023, incluindo:

**Pré-processamento dos Dados:**
- Análise de valores nulos e investigação de padrões de ausência
- Identificação e tratamento de candidatos ausentes/eliminados
- Criação da variável target (NOTA_ENEM) como média das cinco áreas avaliadas
- Remoção de colunas irrelevantes e com alta porcentagem de valores ausentes
  
**Análise Exploratória (EDA):**
- **Análise de Abstenção**: Investigação dos fatores socioeconômicos que influenciam a decisão de não comparecer às provas
- **Análise Univariada**: Distribuição de candidatos por características demográficas e socioeconômicas
- **Análise Bivariada**: Relação entre variáveis socioeconômicas e desempenho no ENEM

**Principais Descobertas:**
- Identificação de padrões claros entre renda familiar e desempenho
- Correlação entre escolaridade dos pais e notas dos filhos
- Impacto do acesso à tecnologia (internet, computador) no desempenho
- Diferenças regionais e demográficas significativas

Criando o dataset final que vai para a modelagem contendo informações de candidatos que realizaram todas as provas, totalizando 2.4 milhões de registros.

## 2. Modelagem 

Este notebook implementa uma abordagem completa de machine learning para predição das notas do ENEM, incluindo:

**Pipeline de Pré-processamento:**
- Criação de pipelines automatizados com encoders específicos para diferentes tipos de variáveis
- Aplicação de **OneHotEncoder** para variáveis categóricas nominais
- Implementação de **OrdinalEncoder** para variáveis categóricas ordinais (escolaridade, renda)

**  Diferentes Modelos Modelos Implementados:**


**Otimização e Validação:**
- **Hyperparameter Tuning** com Otimização bayesiana
- Validação cruzada estratificada para robustez dos resultados
- Comparação sistemática entre diferentes algoritmos
- Análise de métricas: MAE, MSE, RMSE, R²

**Interpretabilidade e Análise:**
- **SHAP (SHapley Additive exPlanations)** para explicabilidade do modelo
- Identificação das variáveis mais importantes para predição
- Análise do impacto individual de cada feature nas predições
- Visualizações interativas dos valores SHAP

**Resultados:**
- Comparação de performance entre todos os modelos testados
- Seleção do melhor modelo baseado em métricas de validação
- Análise detalhada dos fatores socioeconômicos mais preditivos

O pipeline final permite predizer com precisão as notas do ENEM baseado apenas em informações socioeconômicas dos candidatos, oferecendo insights valiosos sobre desigualdades educacionais no Brasil.
