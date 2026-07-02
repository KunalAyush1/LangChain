from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Flash',
    task='text-generation'
    
)

model = ChatHuggingFace(llm=llm)

output = model.invoke("Write a few line about Diffusion based models")

print(output.content)