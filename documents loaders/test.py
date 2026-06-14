from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=[""],
    chunk_size=10,
    chunk_overlap=1
)

loader = TextLoader("notes.txt")

docs = loader.load()

chunks = text_splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.page_content)
    print("-" * 20)