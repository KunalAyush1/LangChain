from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=512)

result = embedding.embed_query("Diffusion based models")

print(str(result)) #using str for better visibility
