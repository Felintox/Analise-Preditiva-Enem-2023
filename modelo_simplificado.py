#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from lightgbm import LGBMRegressor

#%% [markdown]
## Modelo Simplificado para Deploy
# Este modelo utiliza apenas **12 das mais de 30 features** disponíveis no modelo completo, selecionadas com base na **análise SHAP** que identificou as variáveis de maior impacto nas predições.

### Justificativa da Simplificação:

# **Redução de Complexidade:**
# - **Modelo Original**: 30+ features → Pipeline complexo
# - **Modelo Simplificado**: 12 features → Deploy otimizado
# # 
# **Benefícios para Deploy:**
# - **Performance aceitável**: Pequena degradação no RMSE (~2-3 pontos)
# - **Velocidade**: Predições mais rápidas no Streamlit
# - **Manutenibilidade**: Interface mais simples e intuitiva
# - **Robustez**: Menos dependências e transformações

# **Seleção Baseada em SHAP:**
# As 12 features foram escolhidas pela análise de importância SHAP, garantindo que os **fatores mais determinantes** (renda familiar, escolaridade, idade) sejam mantidos, preservando a capacidade preditiva essencial do modelo.
#  
# **Trade-off Consciente:**
# Embora apresente performance ligeiramente inferior ao modelo completo, este modelo simplificado é **otimizado para produção**, priorizando usabilidade e eficiência no ambiente de deploy via Streamlit.
#%%
TOP_FEATURES_SHAP = [
    'Q006',           # Renda familiar (maior impacto SHAP)
    'TP_ST_CONCLUSAO', # Situação conclusão EM
    'TP_LINGUA',      # Língua estrangeira
    'Q024',           # Possui computador
    'TP_FAIXA_ETARIA', # Faixa etária
    'Q002',           # Escolaridade da mãe
    'Q013',           # Possui freezer
    'Q001',           # Escolaridade do pai
    'TP_ESCOLA',      # Tipo de escola
    'Q021',           # TV por assinatura
    'Q003',           # Ocupação do pai
    'Q005'            # Pessoas na residência
]

# Carregando dados
X_train = joblib.load('data/X_train.pkl')
X_test = joblib.load('data/X_test.pkl')
y_train = joblib.load('data/y_train.pkl')
y_test = joblib.load('data/y_test.pkl')

# Remover features sensíveis
features_sensiveis = ['TP_SEXO', 'TP_COR_RACA']
X_train = X_train.drop(columns=features_sensiveis, errors='ignore')
X_test = X_test.drop(columns=features_sensiveis, errors='ignore')

# Filtrar features e combinar train + val
X_train_simples = X_train[TOP_FEATURES_SHAP].copy()
X_test_simples = X_test[TOP_FEATURES_SHAP].copy()

# Usar apenas dados de treino
X_train_full = X_train_simples.copy()
y_train_full = y_train.copy()

print(f"Dados de treino: {X_train_full.shape}")
print(f"Dados de teste: {X_test_simples.shape}")

#%%
# Classes de feature engineering (iguais ao modelo original)
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

#%%
def criar_pipeline_simplificado(modelo):
    col_numericas = ['Q005']
    categorica_nominal_pequena = ['TP_ESCOLA']
    categorica_ordinal = [col for col in TOP_FEATURES_SHAP
                         if col not in col_numericas + categorica_nominal_pequena]

    transformador_ordinal = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='constant', fill_value='Desconhecido')),
        ("ordinal", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    transformador_onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat_onehot", transformador_onehot, categorica_nominal_pequena),
            ("cat_ord", transformador_ordinal, categorica_ordinal)
        ],
        remainder='passthrough'
    )

    pipeline_final = Pipeline([
        ("tratador_nao_sei", TratadorNaoSei()),
        ("agrupamento", AgrupadorCategorias()),
        ("preprocess", preprocessor),
        ("modelo", modelo)
    ])

    return pipeline_final

#%%
def avaliar_modelo(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "MAPE (%)": mape, "MSE": mse, "RMSE": rmse}

#%%
# Modelo com hiperparâmetros otimizados do Optuna
modelo_otimizado = LGBMRegressor(
    learning_rate=0.08584715614794079,
    num_leaves=150,
    n_estimators=853,
    random_state=42,
    verbose=-1
)

pipeline_deploy = criar_pipeline_simplificado(modelo_otimizado)

# Treinar com dados de treino
print("Treinando modelo simplificado com hiperparâmetros otimizados...")
pipeline_deploy.fit(X_train_full, y_train_full)

# TESTE FINAL - dados nunca vistos
y_pred_test = pipeline_deploy.predict(X_test_simples)
metricas_test = avaliar_modelo(y_test, y_pred_test)

print(f"\nRESULTADO FINAL DO MODELO SIMPLIFICADO:")
for metrica, valor in metricas_test.items():
    print(f"  {metrica}: {valor:.4f}")

X_processed = pipeline_deploy[:-1].transform(X_train_full)
print(f"\nFeatures finais: {X_processed.shape[1]}")
print(f"Redução: {X_train.shape[1]} → {len(TOP_FEATURES_SHAP)} → {X_processed.shape[1]} features")

#%%
# Salvar modelo final para deploy
joblib.dump(pipeline_deploy, 'data/modelo_enem_deploy.pkl')
print("\nModelo simplificado salvo como 'modelo_enem_deploy.pkl'")


#%%