from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)
prompt = PromptTemplate(
    template='Write a summary for the following poem -\n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('F1.txt',encoding='utf-8')

docs = loader.load()

print(type(docs))
print(type(docs[0]))

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))
