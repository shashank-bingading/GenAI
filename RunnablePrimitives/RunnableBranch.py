from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableBranch
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda X:len(X.split())>500,RunnableSequence(prompt2,model,parser)),
    #default
    RunnablePassthrough()

)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))