from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
import os
from pydantic import BaseModel, Field

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


model = ChatGroq(groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3)

#schema
class Review(BaseModel):
    key_themes:list[str]=Field(description="Write down all the key themes discussed in the review in a list")
    summary:str=Field("A brief summary of the review")
    sentiment:Literal["pos","neg"] = Field(description="Return sentiment which can either be positive,negative or neutral")
    pros:Optional[list[str]]=Field(default=None,description="Write down all pros inside a list")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
Great Microwave but still a really poor design choice""")

print(type(result))
# print(result['sentiment'])
# print(result['summary'])
print(result.sentiment)
print(result.pros)