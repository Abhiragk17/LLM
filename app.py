import streamlit as st
from urllib.parse import urlparse
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import openai
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain
from langchain.tools.base import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


@st.cache_resource
def get_url_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = "Browse a webpage and retrieve the information and answers relevant to the question. Please use bullet points to list the answers"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    qa_chain: BaseCombineDocumentsChain

    def _run(self, url: str, question: str) -> str:
        response = requests.get(url)
        page_content = response.text
        docs = [Document(page_content=page_content, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i+4]
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)

def run_llm(url, query):
    llm = ChatOpenAI(temperature=0.5,request_timeout=120)
    query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))
    result = query_website_tool._run(url, query)  # Pass the URL and query as arguments
    return result


# Define Streamlit UI layout
st.markdown("<h1 style='text-align: center; color: green;'>Info Retrieval from Website 🦜 </h1>", unsafe_allow_html=True)

input_url = st.text_input("Enter the Website URL", key="url_input")

if len(input_url) > 0:
    url_name = get_url_name(input_url)
    st.info("Your URL is: 👇")
    st.write(url_name)

    your_query = st.text_area("Enter Your Query", key="query_input")

    if st.button("Get Answers"):
        if len(your_query) > 0:
            st.info("Your query is: " + your_query)
            st.markdown("---")

            # Add a loading indicator while the model is processing
            with st.spinner("Searching for answers..."):
                final_answer = run_llm(input_url, your_query)

            st.markdown("<h2 style='color: blue;'>Answers:</h2>", unsafe_allow_html=True)
            st.write(final_answer)

# Add some space at the end
st.markdown("<br><br>", unsafe_allow_html=True)


