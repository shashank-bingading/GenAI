from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


model = ChatGroq(groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3)

#schema
class Review(TypedDict):
    key_themes:Annotated[list[str],"Write down all the key themes discussed in the review in a list"]
    summary:Annotated[str,"A brief summary of the review"]
    sentiment:Annotated[str,"Return Sentiment of the review either negative,postive or neutral"]
    pros:Annotated[Optional[list[str]],"Write down all pros inside a list"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
Great Microwave but still a really poor design choice""")

print(type(result))
print(result['sentiment'])
print(result['summary'])