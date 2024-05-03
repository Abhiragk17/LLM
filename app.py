from langchain_openai import ChatOpenAI
import streamlit as st
from urllib.parse import urlparse
import requests
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
import re
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_community.vectorstores import LanceDB
from langchain_community.embeddings import OpenAIEmbeddings

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# DB_FAISS_PATH = 'vectorstore/db_faiss'

@st.cache_resource
def get_url_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def Get_formatted_chat_history(chat_history) -> str:
    chat_history_formatted = ""
    for message in chat_history[-3:][::-1]:  # Iterate over last 3 messages in reverse order
        if isinstance(message, HumanMessage):
            content = message.content
            chat_history_formatted += "User : " + content + "\n"
        elif isinstance(message, AIMessage):
            content = message.content
            chat_history_formatted += "AI : " + content + "\n"
    return chat_history_formatted

def preprocess_data(data : str) -> str:

    # data = re.sub(r'(\n)\1+','\n', data)
    data = re.sub(r'\n{3,}', '\n', data)
    data = re.sub(r'\t{3,}', '\t', data)
    data = f"""\
    {data}
    """
    return data 

def similarity_search_elastic(index_name,query,k):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    Vectordb = ElasticsearchStore(es_url=ES_URL, index_name=index_name, embedding=embeddings, es_api_key=ES_API_KEY)
    docs = Vectordb.similarity_search(k=k, query=query)
    # docs_content = ''
    # i = 1
    # for doc in docs:
    #     docs_content += doc.page_content
    #     i += 1
    # return docs_content
    return docs

def ExtractData(filepath):
    loader = WebBaseLoader(web_path=filepath)
    splits = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=50,
            ))
    print(f'data: {splits}')
    print(f'Length: {len(splits)}')
    # data_m = markdownify.markdownify(data[0].page_content, heading_style="ATX") 
    # final_data = preprocess_data(data_m)
    # print(f'Data Extracted : {final_data}')
    return splits

@st.cache_resource
def create_embeddings(index_name,file_path):
    try:
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        es_client = Elasticsearch(api_key=ES_API_KEY,hosts=[ES_URL])
        print(f"es client : {es_client}")
        if es_client.indices.exists(index=index_name):
            print(f"Index Exists ")
            return True
        else:
            splits = ExtractData(file_path)
            # text_splitter = RecursiveCharacterTextSplitter(
            #     chunk_size=1500,
            #     chunk_overlap=0,
            # )
            # splits = text_splitter.split_text(data)
            # print(f'Splits : {splits}')
            # print(f'Length : {len(splits)}')
            # Vectordb = ElasticsearchStore(es_connection=es_client,index_name=index_name, embedding=embeddings)
            # print(f'db : {Vectordb}')
            db = ElasticsearchStore.from_documents(
                    splits,
                    embeddings,
                    es_url=ES_URL,
                    index_name=index_name,
                    es_api_key=ES_API_KEY,
                    bulk_kwargs={"request_timeout" : 600}
                )
            if db:
                    print(f'embeddings created with index name : {index_name}')
                    return True
            else:
                return False
    except Exception as e:  
        print("An unexpected error occurred while creating embeddings:", e)
        return False
    
def _run(question: str,chat_history : str,index_name : str) -> str:
    docs = similarity_search_elastic(index_name,question,5)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f'Context : {context}')
    # chain = load_qa_chain(, chain_type="stuff")
    prompt = ChatPromptTemplate.from_messages([
        ("system","""
    You are a ChatBot created for answering questions of the users. You will recieve a prompt that includes retreived content from the vectorDB based on the user's question and the source along with the Chat history of the user.
    Your task is to respond to the user's new question using the information from the vectorDB without relying on your own knowledge.

    Chat history: {chat_history}

    VectorDB content - {context}
    """),
        ("human", """User question: {question}
        note - Do not mention the VectorDB in your answer"""),
    ])
    # prompt = prompt.format_messages(context=context,question=question,chat_history=chat_history)
    # llm = ChatGoogleGenerativeAI(model="gemini-pro")
    llm = ChatOpenAI()
    retrieval_chain = (prompt
        | llm
        | StrOutputParser()
    )
    result = retrieval_chain.stream({"context": context, "question": question,"chat_history" : chat_history})
    print(f'Result : {result}')
    return result


st.set_page_config(page_title="💬 WebChat -AI")
# Taking input URL
index_name = ""
with st.sidebar:
    input_url = st.text_input("Enter the Website URL", key="url_input")
    if len(input_url) > 0:
        url_name = get_url_name(input_url)
        st.info("Your URL is: 👇")
        st.write(url_name)
        index_name = url_name.replace(".","_")
        res_embeddings = create_embeddings(index_name,input_url)
        if res_embeddings: st.success(f"Emeddings created with index name: {index_name}")
        else: st.error(f"Error creating embeddings")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a WebChatAI bot. How can I help you?"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

#chat history
chat_history = st.session_state.chat_history
chat_history_str = Get_formatted_chat_history(chat_history)
print(f'{len(chat_history)}')
print(f"chat history : {chat_history_str}")

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        response = st.write_stream(_run(user_query, chat_history_str,index_name))
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
