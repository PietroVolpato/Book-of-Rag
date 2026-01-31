import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# CONFIGURAZIONE
OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "local_notebook"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

def ingest_document(file_path):
    print(f"Caricamento file: {file_path}...")
    
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    print(f"   -> Trovate {len(docs)} pagine.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"   -> Documento diviso in {len(splits)} chunks (frammenti).")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL
    )

    client = QdrantClient(url=QDRANT_URL)
    
    if not client.collection_exists(COLLECTION_NAME):
        print("   -> Creazione nuova collezione in Qdrant...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    print("Inizio calcolo Embeddings...")
    
    QdrantVectorStore.from_documents(
        splits,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True
    )
    
    print("Completato! Il documento è nella memoria a lungo termine.")

if __name__ == "__main__":
    pdf_path = "SP - Lezione 01.pdf" 
    
    if os.path.exists(pdf_path):
        ingest_document(pdf_path)
    else:
        print(f"File {pdf_path} non trovato. Metti un PDF nella cartella e riprova.")