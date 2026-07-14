from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Flash',
    task='text-generation'
    
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content='You are a helpful research summariser assistant'),
    HumanMessage(content='Tell me about Diffusion models')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)