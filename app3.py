import streamlit as st
from urllib.parse import urlparse
from langchain.chat_models import ChatOpenAI
import os
import openai
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain
from langchain.tools.base import BaseTool
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredURLLoader

os.environ["OPENAI_API_KEY"] = "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"

@st.cache_resource
def get_url_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def _run(url: str, question: str) -> str:
    urls = [url]
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()
    print(data)
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    print(len(docs))
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(docs, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = docsearch.similarity_search(question)
    return chain.run(input_documents=docs, question=question)

styl = f"""
<style>
    .stTextArea {{
      position: fixed;
      bottom: 3rem; /* Adjust the width as needed */
    }}
</style>
"""

st.set_page_config(page_title="💬 AI-Chat")
st.markdown(styl, unsafe_allow_html=True)
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
prompt = st.text_input("Enter Your Query", key="query_input")
if st.button("Get Answers"):
    if len(prompt) > 0:
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

