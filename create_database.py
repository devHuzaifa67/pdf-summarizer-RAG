#load pdf
#split into chunks
#create the embeddings
#store in chroma db 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document

load_dotenv()

pdf_path = "D:/Courses/GenAI/RAG Project L2/document loaders/THE 48 LAWS OF POWER - Robert Greene.pdf"
data = PyPDFLoader(pdf_path)
docs = data.load()

# If text is empty (image-based PDF), use OCR
if sum(len(d.page_content.strip()) for d in docs) < 1000:
    print("PDF seems image-based, running OCR...")
    #pages = convert_from_path(pdf_path)
    pages = convert_from_path(pdf_path, first_page=1, last_page=100)
    text = "".join(pytesseract.image_to_string(page) for page in pages)
    docs = [Document(page_content=text)]
    print("OCR extracted text preview:", text[:500])



splitter = RecursiveCharacterTextSplitter(
  chunk_size = 1000, 
  chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
  documents = chunks, 
  embedding = embedding_model,
  persist_directory="chroma_db"
)

print("Chunks stored:", len(chunks))
for d in chunks[:3]:
    print(d.page_content[:200])
