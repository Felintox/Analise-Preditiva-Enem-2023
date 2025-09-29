# Analise-Preditiva-Enem-2023
Projeto de machine learning utilizando dados fornecidos pelo INEP do ENEM 2023, com objetivo de desenvolver um algoritmo capaz de predizer as notas dos participantes a partir de variáveis socioeconômicas como renda familiar, escolaridade dos pais e tipo de escola frequentada. Além de uma Análise Exploratória de Dados

Os dados são oficiais e foram disponibilizados pelo INEP, como o próprio site informa: "Os microdados reúnem um conjunto de informações detalhadas sobre o Enem 2023, permitindo que gestores, pesquisadores e instituições realizem análises de dados para subsidiar diagnósticos, estudos e pesquisas, bem como o acompanhamento de estatísticas e informações educacionais." <br>
<br>
link : https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem
#
## Deploy

O projeto é finalizado com a realização do deploy de um modelo simplificado (debatido no topico 2.5) deploy este feito atraves do streamlit:

link:

## O projeto foi dividido em 2 partes: Limpeza/Exploração dos dados e Modelagem.

## 1. Limpeza e Exploração dos Dados

Este notebook apresenta uma análise completa dos microdados do ENEM 2023, incluindo:

**Pré-processamento dos Dados:**
- 1. Análise de valores nulos e investigação de padrões.
- 2. Exploração de perfis de candidados que se absteram.
- 3. Criação da variável target (NOTA_ENEM) como média das cinco áreas avaliadas.
- 4. Remoção de colunas irrelevantes e com alta porcentagem de valores ausentes.
#
## Principais Descobertas

Quando consideramos valores médios de Nota para cada classe das categorias, observamos uma relação positiva consistente entre as variáveis socioeconômicas e o desempenho: quanto melhores os indicadores (como renda familiar, nível educacional dos pais, entre outros), maior a nota média no ENEM.

Contudo, ao analisar os boxplots, verifica-se uma grande quantidade de valores discrepantes e alta variabilidade dos dados em todas as categorias.

Isso demonstra que, apesar das tendências observadas nos valores médios, existe considerável dispersão no desempenho individual dentro de cada grupo socioeconômico.

Tecnologia e notas médias: O acesso à internet está associado a diferença de 62,79 pontos na nota média entre quem tem (545,86) e quem não tem acesso (483,07). Similarmente, ter computador está associado a diferença de 107,46 pontos (611,04 vs 503,58).

Em contraste, a posse de celular apresenta menor diferença: 32,67 pontos (549,98 vs 507,31). Os boxplots evidenciam presença de outliers em todas as categorias, indicando alta variabilidade no desempenho independentemente do acesso a essas tecnologias.

Observação: Como se trata de uma análise meramente descritiva, não foram realizados testes estatísticos para comprovar se as médias de cada categoria são estatisticamente diferentes entre si, apenas em algumas categorias como renda.
  
# **Análise Exploratória (EDA):**

**Análise de Abstenção**: Investigação dos fatores socioeconômicos que influenciam a decisão de não comparecer às provas, cada classe da variavel categorica será a proporção de pessoas que se abstiveram em relação ao total de pessoas daquela classe.
#
<img width="1198" height="502" alt="1" src="https://github.com/user-attachments/assets/e872e626-0ade-4071-ab21-936afd08ddb1" />

### Podemos observar claramente que com o aumento da renda há uma diminuição da porcentagem de abstenção de candidatos: quanto maior a renda, menor a abstenção

<img width="1208" height="502" alt="2" src="https://github.com/user-attachments/assets/6157a04f-65b0-498c-932c-08c49c5a3353" />

### Comportamento similar ao da renda: quanto maior o grau de escolaridade do pai, menor a taxa de abstenção, chegando a 24% para pais que completaram o ensino médio e 42% para pais que não estudaram.
<img width="1144" height="503" alt="807e7e2e-c1aa-479f-94a1-e30360be8962" src="https://github.com/user-attachments/assets/e707a67b-7415-4e1f-8ff2-fa969fbd941b" />

### Pessoas com menos de 18 anos e entre 18 e 24 anos apresentam as menores taxas de abstenção, o que é esperado, já que nessa faixa etária muitos estão em busca de ingressar na faculdade por meio do Enem.



# **Análise Univariada**: Distribuição de candidatos por características demográficas e socioeconômicas 



<img width="788" height="490" alt="935a6220-f172-4ef8-aa37-11eb07bd5b7f" src="https://github.com/user-attachments/assets/a69877aa-2a9a-4a26-9007-da35848bd1a4" />

 ### O estado de São Paulo lidera a quantidade de candidatos, seguido de Minas Gerais e Bahia.

<img width="791" height="490" alt="a82c5e7e-c9c4-4e3d-9d32-8504f00f938c" src="https://github.com/user-attachments/assets/d6645ed0-a40c-47d9-96f7-ac103a1c3194" />

 ### 87% dos candidados do ano de 2023 são adolescentes (<18) e jovem adulto (18-24), 0,2% são pessoas idosas.



# **Análise Bivariada**: Relação entre variáveis socioeconômicas e desempenho no ENEM

<img width="1184" height="503" alt="88bd9f5b-403a-4056-bffe-81358f6be0bc" src="https://github.com/user-attachments/assets/9c89e721-1a78-4149-aba8-49e0e0609684" />
<img width="987" height="590" alt="3d7989e8-f6de-4d4f-b104-cdbae012f91b" src="https://github.com/user-attachments/assets/c1d2b06b-d308-4395-a9e8-4621789433c4" />

### As médias confirmam melhor desempenho nas classes de maior renda. Contudo, a alta variabilidade observada nos boxplots mostra que o desempenho individual pode variar consideravelmente dentro de cada faixa socioeconômica.

<img width="1194" height="503" alt="54dc2ca0-7633-4144-83e0-a5e00f9dfc99" src="https://github.com/user-attachments/assets/fb770a6a-0f51-40ee-9123-30fcc261f4f4" />
<img width="983" height="590" alt="6919c193-6857-4686-a34c-1f2d4a4f2456" src="https://github.com/user-attachments/assets/24ce4241-a05a-4899-8488-8ec4a6993564" />

### As notas médias crescem conforme a escolaridade paterna: de 479,72 (nunca estudou) até 612,50 (pós-graduação). Apesar da tendência clara, a alta variabilidade observada nos boxplots mostra ampla distribuição de desempenho em todos os níveis educacionais.

<img width="1132" height="503" alt="070dd082-d317-4320-aa7e-791fe131236a" src="https://github.com/user-attachments/assets/83838bb4-e4af-4d93-877c-ca70393c8d11" />
<img width="989" height="590" alt="0cecb3f6-b385-4bf3-9a2c-bd45e55a539f" src="https://github.com/user-attachments/assets/3ec95c45-2567-415b-a992-1c54ed851bd0" />

### O desempenho mantém-se estável até a faixa 'Adulto Jovem' (548,18 a 524,00 pontos), declinando a partir da 'Meia-idade' (488,45) até 'Idosos' (463,75). Os boxplots mostram presença de outliers em todas as faixas etárias.

### A análise completa esta disponivel no notebook 'tratamento_EDA.py'.

### Finalizamos com um dataset com 2.6 milhões de registros e 35 colunas, com 735 MB de uso.
#

# 2. Modelagem 

Este notebook implementa uma abordagem completa de machine learning para predição das notas do ENEM, incluindo:

# 2.0 **Pipeline de Pré-processamento:**
- Criação de pipelines automatizados com encoders específicos para diferentes tipos de variáveis.
- Aplicação de **OneHotEncoder** para variáveis categóricas nominais.
- Implementação de **OrdinalEncoder** para variáveis categóricas ordinais.

# 2.1 **Diferentes Modelos Modelos Implementados:**
  Modelos Lineares com Regularização:
  
  - Ridge Regression (alpha=1.0)
  - Lasso Regression (alpha=0.1)
  - ElasticNet (alpha=0.1, l1_ratio=0.5)

  Modelos Baseados em Árvores:

  - Decision Tree Regressor
  - Histogram Gradient Boosting Regressor (scikit-learn)
  - LightGBM Regressor 
  - XGBoost Regressor 
  - CatBoost Regressor 

 **Melhores Resultados:**
 - **LightGBM**: 76.2062 RMSE (melhor modelo)
 - **Hist Gradient Boosting**: 76.2110 RMSE
 - **XGBoost**: 76.2891 RMSE
 - **Decision Tree**: 78.54920 RMSE
   
 Os algoritmos de ensemble (LightGBM, XGBoost, HistGB) tiveram performance similares.

 LightGBM foi escolhido para otimização de hiperparâmetros.

 
 <img width="1389" height="790" alt="grafico1" src="https://github.com/user-attachments/assets/54924899-7bae-48e4-8f38-c4416062600b" />


A diferença entre CV e teste final indica:

- • Capacidade de generalização: Performance em dados totalmente não vistos
- • Validação robusta: Cross-validation prediz bem a performance real
- • Ausência de overfitting: Modelos generalizam para novos dados
- • Pipeline confiável: Preprocessing consistente entre conjuntos

Essa consistência entre CV e teste final é o melhor indicador de
que os modelos terão boa performance em dados de produção.

 
 # 2.2 **Otimização e Validação:  Tunagem de Hiperparâmetros com Optuna**

 A tunagem de hiperparâmetros é um processo crucial para otimizar o desempenho de modelos de machine learning.
 Em vez de usar métodos tradicionais como Grid Search ou Random Search, utilizamos otimização bayesiana através da biblioteca Optuna.

  
RESULTADO FINAL NO TESTE:

MAE: 59.1802

MAPE (%): inf

MSE: 5725.1384

RMSE: 75.6646

Melhoria: 76.2062 → 75.6646

Após a tunagem o melhor algoritmo (Lightgbm) melhorou.

# 2.3 **Interpretabilidade e Análise:**
- **SHAP (SHapley Additive exPlanations)** para explicabilidade do modelo
- Identificação das variáveis mais importantes para predição
<img width="732" height="609" alt="image" src="https://github.com/user-attachments/assets/09c8c505-fb33-4698-9c4a-72b14ec1340c" />

A análise de importância do LightGBM revelou que renda familiar (Q006) e faixa etária são as variáveis mais frequentemente utilizadas pelo algoritmo para criar divisões nas árvores de decisão, indicando sua importância estrutural no modelo. As variáveis de escolaridade dos pais (Q001, Q002) e remainder_Q005 ocupam posições intermediárias, confirmando que fatores socioeconômicos são os principais critérios algorítmicos para segmentação dos dados e predição das notas do ENEM.

SHAP Summary Plot - Análise
Este gráfico mostra o impacto individual de cada variável nas predições do modelo, onde cada ponto representa um exemplo e as cores indicam valores altos (rosa) ou baixos (azul) da variável. A posição horizontal revela se a variável aumenta (direita) ou diminui (esquerda) a nota predita.
<br>

A renda familiar (Q006) apresenta o maior range de impacto (-100 a +50 pontos) com padrão claro: renda alta aumenta as notas, renda baixa diminui. A faixa etária mostra padrão inverso: candidatos mais jovens (azul) têm impacto positivo, enquanto mais velhos (rosa) têm impacto negativo, confirmando que jovens performam melhor no ENEM.

SHAP Feature Importance - Análise
Este gráfico apresenta a importância média de cada variável calculada pela média dos valores SHAP absolutos, oferecendo uma visão simplificada de qual variável tem maior impacto médio nas predições do modelo. Diferente do Summary Plot, remove a complexidade das distribuições e cores, focando apenas na magnitude do impacto.
<br>

As duas variáveis mais importantes são cat_ord_Q006 (renda familiar) com aproximadamente 22 pontos de impacto médio, dominando significativamente todas as outras variáveis, e cat_ord_TP_ST_CONCLUSAO com cerca de 13 pontos, relacionada ao status de conclusão do ensino médio. Esta hierarquia confirma que fatores socioeconômicos fundamentais são os principais determinantes das predições do modelo.




# 2.4  **Resultados:** Comparação do nosso modelo final com o valor médio dos dados completo:



COMPARAÇÃO:
<br>
- Baseline (Média):        95.33 RMSE
- LightGBM Otimizado:      75.66 RMSE
- Redução do erro:         19.62 pontos
- Melhoria percentual:     20.6%

CONCLUSÃO:
<br>
 - Baseline (predição pela média): RMSE = 95.33
 - LightGBM otimizado com Optuna:   RMSE = 75.66

 O modelo ML é 20% melhor que simplesmente chutar a média as features realmente agregam valor preditivo significativo.

# 2.5 Deploy com Streamlit:

 Este modelo utiliza apenas **12 das mais de 30 features** disponíveis no modelo completo, selecionadas com base na **análise SHAP** que identificou as variáveis de maior impacto nas predições.

### Justificativa da Simplificação:

 **Redução de Complexidade:**
 - **Modelo Original**: 30+ features → Pipeline complexo
 - **Modelo Simplificado**: 12 features → Deploy otimizado

 **Benefícios para Deploy:**
 - **Performance aceitável**: Pequena degradação no RMSE (~2-3 pontos)
  - **Velocidade**: Predições mais rápidas no Streamlit

- **Manutenibilidade**: Interface mais simples e intuitiva
- **Robustez**: Menos dependências e transformações

**Seleção Baseada em SHAP:**
As 12 features foram escolhidas pela análise de importância SHAP, garantindo que os **fatores mais determinantes** (renda familiar, escolaridade, idade) sejam mantidos, preservando a capacidade preditiva essencial do modelo.
 
**Trade-off Consciente:**

Embora apresente performance ligeiramente inferior ao modelo completo, este modelo simplificado é **otimizado para produção**, priorizando usabilidade e eficiência no ambiente de deploy via Streamlit.


## O pipeline final permite predizer com precisão as notas do ENEM baseado apenas em informações socioeconômicas dos candidatos, oferecendo insights valiosos sobre desigualdades educacionais no Brasil.
