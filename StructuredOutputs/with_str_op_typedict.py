from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Flash',
    task='text-generation'
    
)

model = ChatHuggingFace(llm=llm)

#schema 
class Review(TypedDict):
    
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral"]
        
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated.
                                 There are too many pre-installed apps that I can't remove. Also, the UI looks outdated
                                 compared to other brands. Hoping for a software update to fix this.""")

print(type(result))
print(result)