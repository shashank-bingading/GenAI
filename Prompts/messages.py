from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

messages =[
    SystemMessage("You are the professor at Stanford"),
    HumanMessage("Give me the core idea behind quantum mechanics")
]

result = model.invoke(messages)

messages.append(result.content)

print(messages)
