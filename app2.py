import streamlit as st
from urllib.parse import urlparse
from langchain.chat_models import ChatOpenAI
import os
import openai
from langchain.tools.base import BaseTool
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores.redis import Redis
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key





@st.cache_resource
def get_url_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def _run(url: str, question: str) -> str:
    loaders = UnstructuredURLLoader(urls=[url])
    data = loaders.load()
    print(data) 
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    print(len(docs))
    embeddings = OpenAIEmbeddings()

    # r = rd.Redis(
    # host='redis-11818.c301.ap-south-1-1.ec2.cloud.redislabs.com',
    # port=11818,
    # password='Zw569N4lfVIA8mILjyGNIR6WK0tXdrHF')


    rds = Redis.from_texts(
    docs,
    embeddings,
    redis_url="redis://localhost:6379",
    index_name="users",
    )

    docs = rds.similarity_search(query= question,k=3)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    return chain.run(input_documents=docs, question=question)


st.set_page_config(page_title="💬 AI-Chat")
# Taking input URL
with st.sidebar:
    input_url = st.text_input("Enter the Website URL", key="url_input")

    if len(input_url) > 0:
        url_name = get_url_name(input_url)
        st.info("Your URL is: 👇")
        st.write(url_name)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input("Enter Your Query", key="query_input"):
    print(type(prompt))
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_answer = _run(url=input_url, question=prompt)
            message = {"role": "assistant", "content": final_answer}
            st.session_state.messages.append(message)
            st.rerun()