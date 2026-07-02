from langchain_huggingface import HuggingFaceEmbeddings
#downloading the model locally and running it
embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

text = "Diffusion based models"

vector = embedding.embed_query(text)

print(str(vector))