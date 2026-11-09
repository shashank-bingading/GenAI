#USING RECURSIVE CHARACTER TEXT SPLITTER
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Wikipedia[e] is a free online encyclopedia written and maintained by a community of volunteer editors, known as Wikipedians, through open collaboration and the wiki software MediaWiki. Founded by Jimmy Wales and Larry Sanger in 2001, Wikipedia has been hosted since 2003 by the Wikimedia Foundation, an American nonprofit organization funded mainly by donations from readers.[1] Wikipedia is the largest and most read reference work in history.[2][3]

Initially available only in English, Wikipedia has grown to over 300 languages and is one of the world's most visited websites as of 2026. The English Wikipedia, with over 7 million articles, remains the largest of the editions, which together comprise more than 68 million articles and attract more than 1.5 billion unique device visits and 13 million edits per month (about five edits per second on average) as of April 2024.[4] As of December 2025, over 25% of Wikipedia's traffic comes from the United States, while Japan accounts for nearly 7%, and the United Kingdom, Germany, and Russia each represent around 5%.[5]

Wikipedia has been praised for its internal structure and culture, enabling the democratization of knowledge, and for its extensive coverage on various topics. The encyclopedia has also been criticized for systemic bias, such as a gender bias against women and a geographical bias against the Global South.[6][7] The reliability of Wikipedia was frequently criticized in the 2000s, but has received greater praise from the late 2010s onward.[2][8] Censorship of Wikipedia by national governments has ranged from censoring specific pages to the entire site, sometimes due to its criticism of the government, or for content considered blasphemous.[9][10] Wikipedia articles on breaking news are accessed as sources for up-to-date information by many.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(chunks)

