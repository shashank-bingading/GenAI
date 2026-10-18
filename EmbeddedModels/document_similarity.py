from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

documents = [
    "Something 1",
    "somethings 2",
    "something 3"
]

query = "tell me about somehitng 2"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)[0]

print(list(enumerate(scores)))
##to not mess up the order while sorting

##-1 for highest
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(documents[index])

print("similarity score",score)