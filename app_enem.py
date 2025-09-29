#%%
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Classes necessárias para o modelo (devem ser definidas antes de carregar o modelo)
class TratadorNaoSei(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        if 'Q001' in X_.columns:
            X_['Q001_naosei'] = np.where(X_['Q001'] == 'H', 1, 0)
            X_.loc[X_['Q001'] == 'H', 'Q001'] = np.nan
        if 'Q002' in X_.columns:
            X_['Q002_naosei'] = np.where(X_['Q002'] == 'H', 1, 0)
            X_.loc[X_['Q002'] == 'H', 'Q002'] = np.nan
        if 'Q003' in X_.columns:
            X_['Q003_naosei'] = np.where(X_['Q003'] == 'F', 1, 0)
            X_.loc[X_['Q003'] == 'F', 'Q003'] = np.nan
        return X_

class AgrupadorCategorias(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapa_grupos_idade = {
            1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2,
            10: 3, 11: 3, 12: 3, 13: 3, 14: 4, 15: 4, 16: 4, 17: 4,
            18: 5, 19: 5, 20: 5
        }
        self.renda_abep = {
            "A": 0, "B": 1, "C": 2, "D": 2, "E": 3, "F": 3,
            "G": 3, "H": 3, "I": 4, "J": 4, "K": 4, "L": 4,
            "M": 4, "N": 5, "O": 5, "P": 5, "Q": 5
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        if 'TP_FAIXA_ETARIA' in X_.columns:
            X_['TP_FAIXA_ETARIA'] = X_['TP_FAIXA_ETARIA'].map(self.mapa_grupos_idade)
        if 'Q006' in X_.columns:
            X_['Q006'] = X_['Q006'].map(self.renda_abep)
        return X_

# Configuração da página
st.set_page_config(
    page_title="Preditor de Nota ENEM 2023",
    page_icon="📚",
    layout="wide"
)

# Título da aplicação
st.title("📚 Preditor de Nota ENEM 2023")
st.write("Utilize dados socioeconômicos para prever a nota média do ENEM")

# Carregar o modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load('data/modelo_enem_deploy.pkl')
        return model
    except FileNotFoundError:
        st.error("Modelo não encontrado! Certifique-se de que o arquivo 'modelo_enem_deploy.pkl' está na pasta 'data/'")
        return None

model = load_model()

if model is not None:
    st.success("Modelo carregado com sucesso!")

    # Criar colunas para organizar inputs
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Informações Socioeconômicas")

        # Q006 - Renda familiar
        renda_options = {
            'A': 'Nenhuma Renda',
            'B': 'Até R$ 1.320,00',
            'C': 'De R$ 1.320,01 até R$ 1.980,00',
            'D': 'De R$ 1.980,01 até R$ 2.640,00',
            'E': 'De R$ 2.640,01 até R$ 3.300,00',
            'F': 'De R$ 3.300,01 até R$ 3.960,00',
            'G': 'De R$ 3.960,01 até R$ 5.280,00',
            'H': 'De R$ 5.280,01 até R$ 6.600,00',
            'I': 'De R$ 6.600,01 até R$ 7.920,00',
            'J': 'De R$ 7.920,01 até R$ 9.240,00',
            'K': 'De R$ 9.240,01 até R$ 10.560,00',
            'L': 'De R$ 10.560,01 até R$ 11.880,00',
            'M': 'De R$ 11.880,01 até R$ 13.200,00',
            'N': 'De R$ 13.200,01 até R$ 15.840,00',
            'O': 'De R$ 15.840,01 até R$19.800,00',
            'P': 'De R$ 19.800,01 até R$ 26.400,00',
            'Q': 'Acima de R$ 26.400,00'
        }
        Q006 = st.selectbox("💰 Renda mensal familiar:", list(renda_options.keys()),
                           format_func=lambda x: f"{x} - {renda_options[x]}")

        # Q005 - Pessoas na residência
        Q005 = st.selectbox("🏠 Quantas pessoas moram na residência?", list(range(1, 21)))

        # Q024 - Computador
        computador_options = {
            'A': 'Não',
            'B': 'Sim, um',
            'C': 'Sim, dois',
            'D': 'Sim, três',
            'E': 'Sim, quatro ou mais'
        }
        Q024 = st.selectbox("💻 Possui computador?", list(computador_options.keys()),
                           format_func=lambda x: computador_options[x])

        # Q013 - Freezer
        freezer_options = {
            'A': 'Não',
            'B': 'Sim, um',
            'C': 'Sim, dois',
            'D': 'Sim, três',
            'E': 'Sim, quatro ou mais'
        }
        Q013 = st.selectbox("❄️ Possui freezer?", list(freezer_options.keys()),
                           format_func=lambda x: freezer_options[x])

        # Q021 - TV por assinatura
        tv_options = {'A': 'Não', 'B': 'Sim'}
        Q021 = st.selectbox("📺 Possui TV por assinatura?", list(tv_options.keys()),
                           format_func=lambda x: tv_options[x])

    with col2:
        st.subheader("🎓 Informações Educacionais")

        # TP_ST_CONCLUSAO - Situação ensino médio
        conclusao_options = {
            1: 'Já concluí o Ensino Médio',
            2: 'Estou cursando e concluirei o Ensino Médio em 2023',
            3: 'Estou cursando e concluirei o Ensino Médio após 2023',
            4: 'Não concluí e não estou cursando o Ensino Médio'
        }
        TP_ST_CONCLUSAO = st.selectbox("📚 Situação do Ensino Médio:", list(conclusao_options.keys()),
                                      format_func=lambda x: conclusao_options[x])

        # TP_ESCOLA - Tipo de escola
        escola_options = {
            1: 'Não Respondeu',
            2: 'Pública',
            3: 'Privada'
        }
        TP_ESCOLA = st.selectbox("🏫 Tipo de escola do Ensino Médio:", list(escola_options.keys()),
                                format_func=lambda x: escola_options[x])

        # TP_LINGUA - Língua estrangeira
        lingua_options = {0: 'Inglês', 1: 'Espanhol'}
        TP_LINGUA = st.selectbox("🌍 Língua estrangeira:", list(lingua_options.keys()),
                                format_func=lambda x: lingua_options[x])

        # TP_FAIXA_ETARIA - Idade
        idade_options = {
            1: 'Menor de 17 anos', 2: '17 anos', 3: '18 anos', 4: '19 anos', 5: '20 anos',
            6: '21 anos', 7: '22 anos', 8: '23 anos', 9: '24 anos', 10: '25 anos',
            11: 'Entre 26 e 30 anos', 12: 'Entre 31 e 35 anos', 13: 'Entre 36 e 40 anos',
            14: 'Entre 41 e 45 anos', 15: 'Entre 46 e 50 anos', 16: 'Entre 51 e 55 anos',
            17: 'Entre 56 e 60 anos', 18: 'Entre 61 e 65 anos', 19: 'Entre 66 e 70 anos',
            20: 'Maior de 70 anos'
        }
        TP_FAIXA_ETARIA = st.selectbox("👤 Faixa etária:", list(idade_options.keys()),
                                      format_func=lambda x: idade_options[x])

    # Seção separada para informações dos pais
    st.subheader("👨‍👩‍👧‍👦 Informações dos Pais/Responsáveis")

    col3, col4 = st.columns(2)

    with col3:
        # Q001 - Escolaridade do pai
        escolaridade_options = {
            'A': 'Nunca estudou',
            'B': 'Não completou a 4ª série/5º ano do Ensino Fundamental',
            'C': 'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental',
            'D': 'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio',
            'E': 'Completou o Ensino Médio, mas não completou a Faculdade',
            'F': 'Completou a Faculdade, mas não completou a Pós-graduação',
            'G': 'Completou a Pós-graduação',
            'H': 'Não sei'
        }
        Q001 = st.selectbox("👨 Escolaridade do pai:", list(escolaridade_options.keys()),
                           format_func=lambda x: escolaridade_options[x])

        # Q003 - Ocupação do pai
        ocupacao_options = {
            'A': 'Grupo 1: Lavrador, agricultor, pescador, etc.',
            'B': 'Grupo 2: Empregado doméstico, vendedor, auxiliar administrativo, etc.',
            'C': 'Grupo 3: Padeiro, operário, pedreiro, motorista, etc.',
            'D': 'Grupo 4: Professor, técnico, policial, pequeno empresário, etc.',
            'E': 'Grupo 5: Médico, engenheiro, advogado, diretor, etc.',
            'F': 'Não sei'
        }
        Q003 = st.selectbox("💼 Ocupação do pai:", list(ocupacao_options.keys()),
                           format_func=lambda x: ocupacao_options[x])

    with col4:
        # Q002 - Escolaridade da mãe
        Q002 = st.selectbox("👩 Escolaridade da mãe:", list(escolaridade_options.keys()),
                           format_func=lambda x: escolaridade_options[x])

    # Botão para predição
    if st.button("🎯 Predizer Nota do ENEM", type="primary"):
        # Criar DataFrame com os dados de entrada
        dados_entrada = pd.DataFrame({
            'Q006': [Q006],
            'TP_ST_CONCLUSAO': [TP_ST_CONCLUSAO],
            'TP_LINGUA': [TP_LINGUA],
            'Q024': [Q024],
            'TP_FAIXA_ETARIA': [TP_FAIXA_ETARIA],
            'Q002': [Q002],
            'Q013': [Q013],
            'Q001': [Q001],
            'TP_ESCOLA': [TP_ESCOLA],
            'Q021': [Q021],
            'Q003': [Q003],
            'Q005': [Q005]
        })

        try:
            # Fazer predição
            predicao = model.predict(dados_entrada)[0]

            # Mostrar resultado
            st.success("✅ Predição realizada com sucesso!")

            # Mostrar resultado centralizado
            st.metric(
                label="📊 Nota Média Predita ENEM 2023",
                value=f"{predicao:.2f} pontos"
            )

            # Informações adicionais e limitações
            st.warning("""
            **⚠️ IMPORTANTE - Sobre esta predição:**

            **O que este modelo faz:**
            - Prediz a nota média baseada **apenas** em dados socioeconômicos do ENEM 2023
            - Utiliza padrões estatísticos encontrados em dados históricos
            - Representa uma **estimativa baseada em tendências populacionais**

            **O que este modelo NÃO faz:**
            - **NÃO determina seu desempenho individual**
            - **NÃO considera** preparação específica, cursos, tempo de estudo
            - **NÃO inclui** qualidade da escola, acesso a materiais, suporte familiar
            - **NÃO reflete** suas habilidades e conhecimentos específicos

            **Esta é apenas uma predição estatística:**
            - Baseada em padrões gerais observados nos dados
            - Seu resultado real pode ser muito diferente desta predição
            - Muitos fatores individuais importantes não estão incluídos no modelo
            """)

            st.info("""
            **📊 Este é um projeto acadêmico/científico** que demonstra como dados
            socioeconômicos se relacionam estatisticamente com performance no ENEM.
            """)

        except Exception as e:
            st.error(f"❌ Erro ao fazer predição: {str(e)}")

else:
    st.error("⚠️ Não foi possível carregar o modelo. Verifique se o arquivo existe na pasta 'data/'")

# Rodapé
st.markdown("---")
st.markdown("🤖 **Modelo desenvolvido com LightGBM e otimização bayesiana usando Optuna**")
# %%
