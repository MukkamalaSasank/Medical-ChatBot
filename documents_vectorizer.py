from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


# loading the pdf file

def load_pdf_files(data):
    loader =DirectoryLoader(data,
                            glob = "*.pdf",
                            loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
# print("Lenght of the documents:", len(documents))


# create chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
# print("Lenght of the text chunks:", len(text_chunks))



# Create Vector Embeddings

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Store embedding in FAISS 
db=FAISS.from_documents(text_chunks, embedding=embedding_model)
db.save_local(DB_FAISS_PATH)
