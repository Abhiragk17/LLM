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
import re
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import chromadb

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

@st.cache_resource
def get_url_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def Get_formatted_chat_history(chat_history) -> str:
    chat_history_formatted = ""
    for message in chat_history[-4:][::-1]:  # Iterate over last 3 messages in reverse order
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

@st.cache_resource
def similarity_search_chroma(index_name,query,k):
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device':'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
    )
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_or_create_collection(index_name)
    langchain_chroma = Chroma(
    client=persistent_client,
    collection_name=index_name,
    embedding_function=embeddings,
    )
    retriever = langchain_chroma.as_retriever(search_kwargs = {"k": k})
    return retriever

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
        modelPath = "sentence-transformers/all-MiniLM-l6-v2"
        model_kwargs = {'device':'cpu'}
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,     # Provide the pre-trained model's path
            model_kwargs=model_kwargs, # Pass the model configuration options
        )
        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_or_create_collection(index_name)
        langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=index_name,
        embedding_function=embeddings,
        )
        print("There are", langchain_chroma._collection.count(), "in the collection")
        if langchain_chroma._collection.count() > 0:
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
            db = langchain_chroma.from_documents(splits,embeddings)
            if db:
                    print(f'embeddings created with index name : {index_name}')
                    return True
            else:
                return False
    except Exception as e:  
        print("An unexpected error occurred while creating embeddings:", e)
        return False
    
def _run(question: str,chat_history : str,index_name : str) -> str:
    retriever = similarity_search_chroma(index_name,question,5)
    #docs = "\n".join([f"{doc.page_content}" for doc in docs])
    # chain = load_qa_chain(, chain_type="stuff")
    prompt = ChatPromptTemplate.from_messages([
        ("system","""
    You are a ChatBot created for answering questions of the users. You will recieve a prompt that includes retreived content from the vectorDB based on the user's question and the source along with the Chat history of the user.
    Your task is to respond to the user's new question using the information from the vectorDB without relying on your own knowledge.

    Chat history: {chat_history}

    VectorDB content - {context}
    """),
        ("human", """User question: {user_question}
        note - Do not mention the VectorDB in your answer"""),
    ])
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    chain = ({"context" : retriever,"user_question" : RunnablePassthrough(), "chat_history" : chat_history} | prompt | llm | StrOutputParser())
    result = chain.invoke(question)
    print(result)
    return result


st.set_page_config(page_title="💬 AI-Chat")
# Taking input URL
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
        AIMessage(content="Hello, I am a bot. How can I help you?"),
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
chat_history = st.session_state.chat_history[-3]
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