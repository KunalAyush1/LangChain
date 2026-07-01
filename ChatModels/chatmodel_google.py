from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', max_output_tokens = 10000)

output = model.invoke("Write a short note on Diffusion models")

print(output.content)