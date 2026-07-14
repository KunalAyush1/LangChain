from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


st.header('Simple AI Chatbot')

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Flash',
    task='text-generation'
    
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content='You are a simple AI chatbot')
]



user_input = st.text_input('You: ')
chat_history.append(HumanMessage(content=user_input))
if st.button("Send"):
    if user_input.strip():
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        st.write("### AI")
        st.write(result.content)
        st.write(chat_history)