import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid
import pandas as pd
import pickle

# Carregando o modelo
Arquivo = open('RandomForestClassifier.p', 'rb')
modelo = pickle.load(Arquivo)

st.set_page_config(layout="wide")

st.title("Previsão de status de aprovaçao ou reprovação de alunos.")

st.subheader("**Classificação usando Random Forest**")
st.markdown("Este aplicativo realiza a previsão do status final de alunos: Aprovado/Reprovado.")

st.sidebar.subheader('***Selecione as modalidades:***')

# Carrega do dataframe de forma dinâmica
dados = st.file_uploader('Escolha o arquivo de dados (.csv) que deseja usar para a análise: ', type='csv')
if dados is not None:
    dfBase = pd.read_csv(dados)
    #AgGrid(dfBase)  # interativo

    st.write(f'O arquivo contém {dfBase.shape[1]} atributos e {dfBase.shape[0]} registros.')
    st.markdown("Percentual de alunos por status:")
    dados = dfBase['status'].value_counts(normalize=True) * 100
    st.dataframe(dados)

    df1 = dfBase.groupby(["status"]).count().reset_index()
    grfStatus = px.bar(df1,
                       y=dfBase[['status']].groupby(['status'])['status'].count() / dfBase.shape[0] * 100, x="status", color='status')
    st.plotly_chart(grfStatus)

    # Inicio do Random Forest
    previsores = dfBase

    previsores = previsores.drop(['status', 'cursonome'], axis=1)

    previsao = modelo.predict(previsores)

    Aprovados = 0
    Reprovados = 0
    for status in previsao:
        if status == 0:
            Reprovados += 1
        else:
            Aprovados += 1

    # Juntando valores preditos com base original
    prev = previsao
    prev = pd.DataFrame(prev, columns=['Status Predito'])
    cont = 0
    for i in prev['Status Predito']:
        if i == 0:
            prev['Status Predito'][cont] = "Reprovado"
        else:
            prev['Status Predito'][cont] = "Aprovado"
        cont += 1
    base_concat = pd.concat([dfBase, prev], verify_integrity=True, axis=1)

    st.markdown("Quantidade de alunos por status predito.")
    st.markdown(f'Aprovados: {Aprovados}')
    st.markdown(f'Reprovados: {Reprovados}')

    st.markdown("Porcentagem")
    T = len(previsao)
    A = (Aprovados / T) * 100
    R = (Reprovados / T) * 100

    st.markdown(f'Aprovados: {round(A, 4):.2f} %')
    st.markdown(f'Reprovados: {round(R, 4):.2f} %')
    st.subheader("**Comparação entre o fato e o predito.**")
    st.markdown("Verifique na tabela abaixo as colunas no final que informam o que de fato aconteceu com o aluno e o que foi predito para o o mesmo.")
    AgGrid(base_concat)
#%%
