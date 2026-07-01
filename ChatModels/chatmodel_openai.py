from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=1.2, max_completion_tokens=100)
#temperature is a parameter which determines creativity and determinism of model
#max_completion_tokens is a parameters which limits the no of tokens in output of the model
output = model.invoke("Are you different from LLMs?")

print(output.content)

#ChatModels give a lot of metadata in output along with the content
# to get only the content use <name>.content