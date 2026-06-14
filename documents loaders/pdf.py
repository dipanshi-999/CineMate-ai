from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

loader = PyPDFLoader("3.3 (IoT) Reference Architectures and Models copy.pdf")

docs = loader.load()

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)
chunks = text_splitter.split_documents(docs)


print(len(chunks))