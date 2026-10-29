from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

def word_count(text):
    return len(text.split())

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel(
    {
        'joke':RunnablePassthrough(),
        'word_count':RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'Smart watches'})

final_result = """
{} \n word count - {}""".format(result['joke'],result['word_count'])

print(final_result)
