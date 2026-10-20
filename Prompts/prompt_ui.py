from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import load_prompt

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')

# prompt = template.invoke(
#     {
#         'paper_input':paper_input,
#         'style_input':style_input,
#         'length_input':length_input
#     }
# )

if st.button('Summarise'):
    #invoke is called only once using chain
    chain = template | model
    result = chain.invoke(
        {
                'paper_input':paper_input,
                'style_input':style_input,
                'length_input':length_input
            }
    )

    # result = model.invoke(prompt)

    st.write(result.content)