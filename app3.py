from langchain.vectorstores.redis import Redis
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings()
texts = ["document1", "document2", "document3"]
rds = Redis.from_texts(
    texts = texts,
    embedding = embeddings,
    redis_url="redis://localhost:6379",
    index_name="test_index",
    )

print(rds.similarity_search(query="document4", k=1))
# print(rds.all())