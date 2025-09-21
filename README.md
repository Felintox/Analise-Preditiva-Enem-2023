# Analise-Preditiva-Enem-2023
Projeto de machine learning utilizando dados fornecidos pelo INEP do ENEM 2023, com objetivo de desenvolver um algoritmo capaz de predizer as notas dos participantes a partir de variáveis socioeconômicas como renda familiar, escolaridade dos pais e tipo de escola frequentada.

Os dados são oficiais e foram disponibilizados pelo INEP, como o próprio site informa: "Os microdados reúnem um conjunto de informações detalhadas sobre o Enem 2023, permitindo que gestores, pesquisadores e instituições realizem análises de dados para subsidiar diagnósticos, estudos e pesquisas, bem como o acompanhamento de estatísticas e informações educacionais." <br>
<br>
link : https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem
#
## O projeto foi dividido em 2 partes: Limpeza/Exploração dos dados e Modelagem.

## 1. Limpeza e Exploração dos Dados

Este notebook apresenta uma análise completa dos microdados do ENEM 2023, incluindo:

**Pré-processamento dos Dados:**
- Análise de valores nulos e investigação de padrões de ausência
- Identificação e tratamento de candidatos ausentes/eliminados
- Criação da variável target (NOTA_ENEM) como média das cinco áreas avaliadas
- Remoção de colunas irrelevantes e com alta porcentagem de valores ausentes
#
  
# **Análise Exploratória (EDA):**

## **Análise de Abstenção**: Investigação dos fatores socioeconômicos que influenciam a decisão de não comparecer às provas
## Cada barra representa a porcentagem de abstenção de cada categoria
<img width="1198" height="502" alt="1" src="https://github.com/user-attachments/assets/e872e626-0ade-4071-ab21-936afd08ddb1" />

## Podemos observar claramente que com o aumento da renda há uma diminuição da porcentagem de abstenção de candidatos: quanto maior a renda, menor a abstenção

<img width="1208" height="502" alt="2" src="https://github.com/user-attachments/assets/6157a04f-65b0-498c-932c-08c49c5a3353" />

## Comportamento similar ao da renda: quanto maior o grau de escolaridade do pai, menor a taxa de abstenção, chegando a 24% para pais que completaram o ensino médio e 42% para pais que não estudaram.
<img width="1144" height="503" alt="807e7e2e-c1aa-479f-94a1-e30360be8962" src="https://github.com/user-attachments/assets/e707a67b-7415-4e1f-8ff2-fa969fbd941b" />

## Pessoas com menos de 18 anos e entre 18 e 24 anos apresentam as menores taxas de abstenção, o que é esperado, já que nessa faixa etária muitos estão em busca de ingressar na faculdade por meio do Enem.



# **Análise Univariada**: Distribuição de candidatos por características demográficas e socioeconômicas 



<img width="788" height="490" alt="935a6220-f172-4ef8-aa37-11eb07bd5b7f" src="https://github.com/user-attachments/assets/a69877aa-2a9a-4a26-9007-da35848bd1a4" />

## O estado de São Paulo lidera a quantidade de candidatos, seguido de Minas Gerais e Bahia.

<img width="791" height="490" alt="a82c5e7e-c9c4-4e3d-9d32-8504f00f938c" src="https://github.com/user-attachments/assets/d6645ed0-a40c-47d9-96f7-ac103a1c3194" />

## 87% dos candidados do ano de 2023 são adolescentes (<18) e jovem adulto (18-24), 0,2% são pessoas idosas.



# **Análise Bivariada**: Relação entre variáveis socioeconômicas e desempenho no ENEM

<img width="1184" height="503" alt="88bd9f5b-403a-4056-bffe-81358f6be0bc" src="https://github.com/user-attachments/assets/9c89e721-1a78-4149-aba8-49e0e0609684" />
<img width="987" height="590" alt="3d7989e8-f6de-4d4f-b104-cdbae012f91b" src="https://github.com/user-attachments/assets/c1d2b06b-d308-4395-a9e8-4621789433c4" />

# As médias confirmam melhor desempenho nas classes de maior renda. Contudo, a alta variabilidade observada nos boxplots mostra que o desempenho individual pode variar consideravelmente dentro de cada faixa socioeconômica.

<img width="1194" height="503" alt="54dc2ca0-7633-4144-83e0-a5e00f9dfc99" src="https://github.com/user-attachments/assets/fb770a6a-0f51-40ee-9123-30fcc261f4f4" />
<img width="983" height="590" alt="6919c193-6857-4686-a34c-1f2d4a4f2456" src="https://github.com/user-attachments/assets/24ce4241-a05a-4899-8488-8ec4a6993564" />

# As notas médias crescem conforme a escolaridade paterna: de 479,72 (nunca estudou) até 612,50 (pós-graduação). Apesar da tendência clara, a alta variabilidade observada nos boxplots mostra ampla distribuição de desempenho em todos os níveis educacionais.

<img width="1132" height="503" alt="070dd082-d317-4320-aa7e-791fe131236a" src="https://github.com/user-attachments/assets/83838bb4-e4af-4d93-877c-ca70393c8d11" />
<img width="989" height="590" alt="0cecb3f6-b385-4bf3-9a2c-bd45e55a539f" src="https://github.com/user-attachments/assets/3ec95c45-2567-415b-a992-1c54ed851bd0" />

# O desempenho mantém-se estável até a faixa 'Adulto Jovem' (548,18 a 524,00 pontos), declinando a partir da 'Meia-idade' (488,45) até 'Idosos' (463,75). Os boxplots mostram presença de outliers em todas as faixas etárias.

# A análise completa esta disponivel no notebook 'tratamento_EDA.py'

# Finalizamos com um dataset com 2.6 milhões de registros e 35 colunas, com 735 MB de uso.

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
