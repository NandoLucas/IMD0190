import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from collections import Counter
from pydantic import BaseModel, Field
from typing import Union, List, Literal

# insert your gemini api key here
gemini_api_key = ""
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_api_key,
    temperature=0.0
    )
class SentimentAnalysisResponse(BaseModel):
    """The response of a function that performs sentiment analysis on text."""

    # The sentiment label assigned to the text
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        default_factory=str,
        description="The sentiment label assigned to the text. You can only have 'positive', 'neutral' or 'negative' as values.",
    )
model_sentiment_gemini = gemini.with_structured_output(SentimentAnalysisResponse)
@tool
def analyze_sentiment(texts: Union[str, List[str]]) -> dict:
    """
    Recebe um único texto ou uma lista de textos e retorna a classificação de sentimento de cada um.
    Se for uma lista, também retorna a contagem total de cada classificação.
    """

    if isinstance(texts, str):
        # Caso seja um único texto, retorna apenas sua classificação
        return model_sentiment_gemini.invoke(texts)

    elif isinstance(texts, list):
        # Caso seja uma lista, processa cada um e gera um resumo da contagem
        sentiment_counts = Counter()
        individual_results = []

        for text in texts:
            result = model_sentiment_gemini.invoke(text)
            sentiment_counts[result.sentiment] += 1
            individual_results.append(result)

        return {
            "individual_results": individual_results,
            "total_counts": dict(sentiment_counts)
        }

# 🔹 Configuração do Agente
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente amigável e faz análise de sentimentos com textos e arquivos. "),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 🔹 Lista de ferramentas disponíveis
tools = [analyze_sentiment]

# 🔹 Criando o agente do LangChain
agent = create_tool_calling_agent(
    tools=tools,
    llm=gemini,  # Substituir pelo modelo correto
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

# 🔹 Interface no Streamlit
st.title("Análise de Sentimentos 🚀 com LLM")
st.write("Digite um texto, uma lista de textos ou faça upload de um CSV.")

# 📌 **Entrada de texto do usuário**
user_input = st.text_area("Digite um comando para o agente:")

# 📌 **Upload do CSV**
uploaded_file = st.file_uploader("Ou carregue um arquivo CSV", type=["csv"])

df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file, usecols=[14])
    st.subheader("📊 Prévia da coluna alvo presente no arquivo")
    st.write(df.head())

if st.button("Executar Análise"):
    with st.spinner("O agente está processando sua solicitação..."):
        try:
            if user_input and uploaded_file:
                reviews_filtered = df.dropna()
                reviews = reviews_filtered['Unnamed: 14'].head().tolist()  # Converter DataFrame para lista
                print(reviews)
                result = agent_executor.invoke({"input": reviews})
            elif user_input:
                result = agent_executor.invoke({"input": user_input})
            else:
                result = {"error": "Nenhum input fornecido"}

            #st.subheader("🔍 Debug: Resposta Bruta do Agente")
            st.write(result)  # Mostra a resposta bruta para depuração

        except Exception as e:
            st.error(f"Erro ao executar o agente: {str(e)}")

    if "error" in result:
        st.error(result["error"])
    else:
        if "individual_results" in result:
            st.subheader("📌 Classificação Individual")
            st.write((result["individual_results"]))

        if "total_counts" in result:
            st.subheader("📊 Resumo da Análise")
            st.write(result["total_counts"])

            # Criando o gráfico de barras
            st.subheader("📊 Distribuição dos Sentimentos")
            fig, ax = plt.subplots()
            ax.bar(result["total_counts"].keys(), result["total_counts"].values(), color=["green", "gray", "red"])
            ax.set_xlabel("Sentimentos")
            ax.set_ylabel("Quantidade")
            ax.set_title("Distribuição dos Sentimentos no Dataset")
            st.pyplot(fig)
