from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv() #loads environment variables from .env file


llm = OpenAI(model="gpt-3.5-turbo-instruct") #creates an instance of OpenAI model

output = llm.invoke("Do you know what are LLMs") #sends a request with prompt and receives output

print(output)
