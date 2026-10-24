from langchain_community.document_loaders import PyPDFLoader 

from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document loaders/Huzaifa CN Project Report.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
  chunk_size = 100, 
  chunk_overlap = 1
)

#print(docs[4].page_content)

splitter.split_documents(docs)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)