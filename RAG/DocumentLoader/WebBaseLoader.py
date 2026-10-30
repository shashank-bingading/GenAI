from langchain_community.document_loaders import WebBaseLoader

url = 'https://github.com/shashank-bingading'

loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))