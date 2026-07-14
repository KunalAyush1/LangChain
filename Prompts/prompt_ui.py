from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.header('Research Tool')

paper_input = st.selectbox("Select the Paper", ["Select...", "Attention is All You Need","BERT", "GPT3", "Diffusion Models", "Video Auto Encoders" ])


llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Flash',
    task='text-generation'
    
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="""
    Please summarize the research paper titled "{paper_input}"
    in two lines which cover the overall meaning and impact of the paper""",
    input_variables=['paper_input']
    
)

prompt = template.invoke({
    "paper_input":paper_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)