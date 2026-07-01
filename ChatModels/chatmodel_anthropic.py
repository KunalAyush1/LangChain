from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-opus-4-8')

output =model.invoke("Write a paragraph about Diffusion models")

print(output.content)