import sys
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

# CONFIGURAZIONE (Deve coincidere con ingest.py)
OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "local_notebook"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

def start_chat():
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4}) # Recupera i top 4 documenti rilevanti

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
    )

    template = """Sei un assistente AI che risponde alle domande basandosi SOLO sul contesto fornito qui sotto.
    Se la risposta non è nel contesto, dì "Non ho trovato questa informazione nei documenti caricati".
    Non inventare nulla.

    Contesto:
    {context}

    Domanda Utente: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("BookofRAG è pronto! (Scrivi 'exit' per uscire)")
    print("-" * 50)

    while True:
        query = input("\nTu: ")
        if query.lower() in ["exit", "esci", "quit"]:
            break

        print("\nRicerca nei documenti...", end="", flush=True)
        
        docs = retriever.invoke(query)
        print(f" Trovati {len(docs)} frammenti rilevanti.\n")
        
        print("--- FONTI USATE ---")
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Sconosciuto")
            page = doc.metadata.get("page", "?")
            content_preview = doc.page_content[:100].replace("\n", " ")
            print(f"[{i+1}] {source} (Pag. {page}): {content_preview}...")
        print("-" * 20 + "\n")

        print("AI: ", end="", flush=True)
        
        for chunk in rag_chain.stream(query):
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    start_chat()