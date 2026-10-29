from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel
import os

load_dotenv()

prompt = PromptTemplate(
    template='Generate an insta post about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a linkedIN post about: {topic}',
    input_variables=['topic']
)

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'insta':RunnableSequence(prompt,model,parser),
    'linkedIn':RunnableSequence(prompt2,model,parser)
})

print(parallel_chain.invoke({'topic':'AI'}))

