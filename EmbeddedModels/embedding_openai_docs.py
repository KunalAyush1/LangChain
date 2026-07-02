from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=512)


documents = [
    "Diffusion based model",
    "Transformers based models",
    "CNN based models"
]

#generates embeddings of a document

result = embedding.embed_documents(documents)

#it will return a 2d list where every element will the embedding of respective document[i]

print(str(result)) #using str for better visibility
