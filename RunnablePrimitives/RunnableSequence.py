from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Tell me why the joke was so bad: {result}',
    input_variables=['result']
)

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

parser = StrOutputParser()

chain = RunnableSequence(prompt,model,parser,prompt2,model,parser)

print(chain.invoke({'topic':'Cars'}))