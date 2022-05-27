import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid
import pandas as pd
import pickle

# Carregando o modelo
Arquivo = open('RandomForestClassifier.p', 'rb')
modelo = pickle.load(Arquivo)

st.set_page_config(layout="wide")

st.title("Prevendo o Status dos Alunos")

st.subheader("**Classificação com Random Forest**")
st.markdown("Este app faz a classificação do status do aluno: Aprovado/Reprovado.")

st.sidebar.subheader('***Selecione as modalidades:***')

# Carrega do dataframe de forma dinâmica
dados = st.file_uploader('Escolha o dataset (.csv) para continuarmos com as análises: ', type='csv')
if dados is not None:
    dfBase = pd.read_csv(dados)
    #AgGrid(dfBase)  # interativo

    st.write("Shape: ", dfBase.shape)

    st.markdown("Quantidade de alunos por status")
    dados = dfBase['status'].value_counts(normalize=True) * 100
    st.dataframe(dados)

    df1 = dfBase.groupby(["status"]).count().reset_index()
    grfStatus = px.bar(df1,
                       y=dfBase[['status']].groupby(['status'])['status'].count() / dfBase.shape[0] * 100,
                       x="status",
                       color='status')
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
    prev = pd.DataFrame(prev, columns=['Status Predict'])
    cont = 0
    for i in prev['Status Predict']:
        if i == 0:
            prev['Status Predict'][cont] = "Reprovado"
        else:
            prev['Status Predict'][cont] = "Aprovado"
        cont += 1
    base_concat = pd.concat([dfBase, prev], verify_integrity=True, axis=1)

    st.markdown("Quantidade Predict de alunos por status")
    st.markdown(f'Aprovados: {Aprovados}')
    st.markdown(f'Reprovados: {Reprovados}')

    st.markdown("Porcentagem")
    T = len(previsao)
    A = (Aprovados / T) * 100
    R = (Reprovados / T) * 100

    st.markdown(f'Aprovados: {round(A, 4)} %')
    st.markdown(f'Reprovados: {round(R, 4)} %')
    AgGrid(base_concat)