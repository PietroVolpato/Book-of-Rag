import sys
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_index(doc_path: str):
    """Crea un indice RAG a partire da un file o directory."""

    # 1. Carica i documenti
    documents = SimpleDirectoryReader(input_files=[doc_path]).load_data()

    # 2. Embedding model leggero
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Vector store (ChromaDB in locale)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("book_of_rag")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Costruisci l’indice
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index


def load_phi3():
    """Carica Phi-3 Mini con tokenizer e modello ufficiale HuggingFace."""

    model_name = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=4000,     # finestra massima di Phi-3 mini
        max_new_tokens=512,      # risposta massima
        generate_kwargs={"temperature": 0.7, "do_sample": True},
    )

    return llm


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py <percorso_documento>")
        sys.exit(1)

    doc_path = sys.argv[1]

    print(f"📚 Carico e indicizzo: {doc_path}")
    index = build_index(doc_path)

    # Carica LLM (Phi-3 mini)
    llm = load_phi3()

    # Costruisci query engine
    query_engine = index.as_query_engine(llm=llm)

    print("\n❓ Inserisci una domanda sul documento (scrivi 'exit' per uscire)")
    while True:
        query = input(">> ")
        if query.lower() == "exit":
            break

        response = query_engine.query(query)
        print("\n💡 Risposta:\n", response, "\n")


if __name__ == "__main__":
    main()
