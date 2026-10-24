import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name,age and city of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser
final_result = chain.invoke({})

print(final_result)
print(type(final_result))