#%%
"""
Funções de visualização para análise exploratória de dados do ENEM
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def barplot_horizontal(dado, coluna, titulo):
    plt.figure(figsize=(8, 5))
    # cria gráfico ordenando por contagem percentual (decrescente)
    ax = sns.countplot(
        data=dado,
        y=coluna,
        stat='percent',
        order=dado[coluna].value_counts(normalize=True).index  # ordem decrescente
    )
    # adiciona texto ao lado das barras
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.5,                    # um pouco à direita da barra
            p.get_y() + p.get_height() / 2, # centralizado verticalmente
            f'{width:.1f}%',                # formatação
            ha='left', va='center'
        )
    # remove "bordas" do gráfico
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(titulo)
    plt.tight_layout()
    plt.show()


def barplot(dado,coluna,titulo):
    plt.figure(figsize=(8, 4))
    ax = sns.countplot(data=dado, x=coluna, stat='percent')
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.,  # Posição X (centralizado na barra)
            height + 1,                     # Posição Y (um pouco acima da barra)
            f'{height:.1f}%',               # O texto a ser exibido, formatado
            ha='center'                     # Alinhamento horizontal
        )
    #Remove as "bordas" do grafico.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.set_ylabel('') 
    ax.set_xlabel('')
    ax.set_title(titulo)
    plt.show()

def plot_media_nota(df, coluna, notas, titulo=None,figsize=(12,6)):
    # Cria tabela: coluna de interesse x nota média
    tabela = pd.DataFrame({
        coluna: df[coluna],
        "Nota": notas
    }).groupby(coluna)["Nota"].mean().reset_index()
    # Cria o gráfico
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(y=coluna, x="Nota", data=tabela,ax=ax)
    # Valores sobre as barras
    for i, v in enumerate(tabela["Nota"]):
        ax.text(v + 0.5, i, f'{v:.2f}', ha='left', va='center')
    # Ajustes visuais
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.set_xlabel('')
    ax.set_ylabel(coluna.replace('_', ' ').title())
    ax.set_title(titulo if titulo else f'Nota média por {coluna.replace("_"," ").lower()}')
    ax.grid(False)
    plt.show()

def plot_boxplot_nota(df, coluna_cat, coluna_num, titulo=None, figsize=(10, 6), order=None):
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=coluna_cat, y=coluna_num, data=df, ax=ax, order=order)
    ax.set_xlabel(coluna_cat.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Nota', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right') # Adicionado para melhor visualização de rótulos longos
    plt.tight_layout() # Adicionado para garantir que os rótulos não sejam cortados
    plt.show()

def plot_taxa_abstencao(df, coluna, titulo=None,figsize=(12,6)):
    # Cruzamento GRUPO x coluna
    tabela = pd.crosstab(df[coluna], df['GRUPO'])
    # Calcula taxa de abstenção
    tabela['Taxa_Abstencao_%'] = tabela['abstencao'] / (tabela['abstencao'] + tabela['presente_dois_dias']) * 100
    # Cria o gráfico
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(y=tabela.index, x='Taxa_Abstencao_%', data=tabela.reset_index(), ax=ax)
    # Valores sobre as barras
    for i, v in enumerate(tabela['Taxa_Abstencao_%']):
        ax.text(v + 0.5, i, f'{v:.2f}%', ha='left', va='center')
    # Ajustes visuais
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.set_xlabel('')
    ax.set_ylabel(coluna.replace('_', ' ').title())
    ax.set_title(titulo if titulo else f'Taxa de abstenção por {coluna.replace("_"," ").lower()}')
    ax.grid(False)
    plt.show()
# %%
