# src/galoispy/rag/utils/chroma_utils.py
import json
import os
from typing import Iterator, Optional
# Langchain imports
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader, JSONLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import TokenTextSplitter
# HuggingFace imports
from sentence_transformers import CrossEncoder
# GaloisPy utils
from galoispy.utils import ollama_utils
from galoispy.utils import spinner_animation as sp

def load_document_from_path(path: str) -> Iterator[Document]:
    """
    Load document from a given file path.\n
    It supports different file types like PDF, CSV, TXT and JSON.
    
    :param path: The file path or directory to load the document from.
    :type path: str
    :return: An iterator of Document objects.
    :rtype: Iterator[Document]
    """
    # If it is a directory, scan all the files in the directory and load them
    if os.path.isdir(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            # If 'file_path' is a file, load it, otherwise skip it (to avoid loading sub-directories)
            if os.path.isfile(file_path):
                yield from load_document_from_path(file_path)
        return
    
    # Load based on file extension
    if path.lower().endswith('.csv'):
        loader = CSVLoader(path=path, csv_args={"delimiter": ",", "quotechar": '"'}, encoding="utf-8")
    elif path.lower().endswith('.json'):
        loader = JSONLoader(path)
    elif path.lower().endswith('pdf'):
        loader = PyPDFLoader(path)
    elif path.lower().endswith('.txt'):
        loader = TextLoader(path, encoding="utf-8")
    elif path.lower().endswith('.md'):
        loader = UnstructuredMarkdownLoader(path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type {(path.split('.')[-1]).capitalize()!r} for document {path!r}. Available types are CSV, JSON, PDF, TXT, and MD.")
    # Load the documents lazily to save memory
    yield from loader.lazy_load()

def batch_documents_loader(generator: Iterator[Document], batch_size: int) -> Iterator[list[Document]]:
    """
    Returns batches of documents from a generator.
    
    :param generator: An Iterator of Document objects.
    :type generator: Iterator[Document]
    :param batch_size: The number of documents in each batch.
    :type batch_size: int
    :return: An iterator of lists of Document objects.
    :rtype: Iterator[list[Document]]
    """
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    Split documents into smaller chunks with specified size and overlap.
    
    :param documents: The list of Document objects to be split.
    :type documents: list[Document]
    :param chunk_size: The size of each chunk (in tokens).
    :type chunk_size: int
    :param chunk_overlap: The number of overlapping tokens between chunks.
    :type chunk_overlap: int
    :return: A list of Document objects split into chunks.
    :rtype: list[Document]
    """
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)

def _get_embedding(model_name: str) -> OllamaEmbeddings:
    """
    Get the Ollama embedding model.
    
    :param model_name: The name of the Ollama embedding model.
    :type model_name: str
    :return: An OllamaEmbeddings object.
    :rtype: OllamaEmbeddings
    """
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings

def _get_chunks_ids(chunks: list[Document]) -> list:
    """
    Assign unique IDs to each document chunk based on source, page, and chunk index.\n
    A chunk ID is defined as: source:page:chunk_index (e.g., document.pdf:3:0).
    
    :param chunks: The list of document chunks.
    :type chunks: list[Document]
    :return: The list of document chunks with assigned IDs.
    :rtype: list
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def _is_chroma_existing(chroma_path: str) -> bool:
    """
    Check if a Chroma vector store exists at the given path.
    
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :return: True if the Chroma vector store exists, False otherwise.
    :rtype: bool
    """
    # Check if the chroma_path is a relative path or absolute path
    chroma_path = os.path.abspath(chroma_path)
    # Check if the directory exists
    return os.path.exists(chroma_path) and os.path.isdir(chroma_path) and len(os.listdir(chroma_path)) > 0

def load_chroma(chroma_path: str, model_name: str) -> Chroma:
    """
    Load an existing Chroma vector store.\n
    If the vector store does not exist, it creates a new one.
    
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :param model_name: The name of the embedding model.
    :type model_name: str
    :return: A Chroma object.
    :rtype: Chroma
    """
    
    # Check if the chroma_path is a relative path or absolute path
    chroma_path = os.path.abspath(chroma_path)
    config_path = os.path.join(chroma_path, "chroma_config.json")
    if not _is_chroma_existing(chroma_path=chroma_path):
        # Create the directory if it does not exist
        os.makedirs(chroma_path, exist_ok=True)
        config = {"chroma_embedding": model_name}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
    else:
        # Verify that the embedding model matches the existing one
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing chroma_config.json in {chroma_path!r}. Cannot verify embedding model.")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("chroma_embedding") != model_name:
            raise ValueError(f"The embedding model {model_name!r} does not match the one in the existing Chroma vector store ({config.get('chroma_embedding')!r}) at {chroma_path!r}.")
    db = Chroma(persist_directory=chroma_path, embedding_function=_get_embedding(model_name=model_name))
    return db

def is_collection_existing(chroma_path: str, collection_name: str) -> bool:
    """
    Check if a Chroma collection exists.
    
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :param collection_name: The name of the collection to check.
    :type collection_name: str
    :return: True if the collection exists, False otherwise.
    :rtype: bool
    """
    chroma_path = os.path.abspath(chroma_path)
    db = Chroma(persist_directory=chroma_path)
    # Check if the collection already exists
    existing_collections = db._client.list_collections()
    existing_collections = [col.name for col in existing_collections]
    return collection_name in existing_collections

def create_empty_collection(chroma_path: str, collection_name: str) -> None:
    """
    Create an empty Chroma collection.\n
    If drop_existing is True and the collection already exists, it will be cleared.
    
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :param collection_name: The name of the collection to create.
    :type collection_name: str
    """
    chroma_path = os.path.abspath(chroma_path)
    db = Chroma(persist_directory=chroma_path)
    # Check if the collection already exists
    if is_collection_existing(chroma_path=chroma_path, collection_name=collection_name):
        raise ValueError(f"The collection {collection_name!r} already exists in the Chroma vector store at {chroma_path!r}.")
    # Create the empty collection
    Chroma(persist_directory=chroma_path, collection_name=collection_name)

def add_to_chroma(chunks: list[Document], chroma_path: str, model_name: str, collection_name: str) -> None:
    """
    Create a vector store from document chunks.\n
    It loads the chunks into a Chroma vector store using the specified embedding model.
    
    :param chunks: The list of document chunks to add.
    :type chunks: list[Document]
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :param model_name: The name of the embedding model.
    :type model_name: str
    :param collection_name: The name of the collection to add the chunks to.
    :type collection_name: str
    """
    # Check if ollama is running
    if not ollama_utils.is_ollama_active():
        ollama_utils.start_ollama()
    
    # Check if the chroma_path is a relative path or absolute path
    chroma_path = os.path.abspath(chroma_path)
    # Create the Chroma vector store
    db = Chroma(persist_directory=chroma_path, embedding_function=_get_embedding(model_name=model_name), collection_name=collection_name)
    
    # Get the chunk IDs
    chunks_ids = _get_chunks_ids(chunks=chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    
    # Keep only the new chunks
    new_chunks = []
    for chunk in chunks_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    # Add the new document chunks to the vector store
    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)

def remove_from_chroma(chunks: list[Document], chroma_path: str, model_name: str, collection_name: str) -> None:
    """
    Remove document chunks from a Chroma vector store.
    
    :param chunks: The list of document chunks to remove.
    :type chunks: list[Document]
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :param model_name: The name of the embedding model.
    :type model_name: str
    :param collection_name: The name of the collection to remove the chunks from.
    :type collection_name: str
    """
    # Check if ollama is running
    if not ollama_utils.is_ollama_active():
        ollama_utils.start_ollama()

    # Check if the embedding model is already downloaded
    if model_name not in ollama_utils.list_models():
        # Download the embedding model
        try:
            stop_spinner = sp.spinner(message=f"Downloading embedding model {model_name!r}")
            ollama_utils.model_selection(model_name=model_name, download=True)
            # Stop the animation
            stop_spinner.set()
            print("\nDownload completed.")
        except Exception as e:
            stop_spinner.set()
            print("\n")
            raise RuntimeError(f"Error downloading embedding model {model_name!r}: {e}")
        
    # Check if the chroma_path is a relative path or absolute path
    chroma_path = os.path.abspath(chroma_path)
    # Create the Chroma vector store
    db = Chroma(persist_directory=chroma_path, embedding_function=_get_embedding(model_name=model_name), collection_name=collection_name)

    # Get the chunk IDs
    chunks_ids = _get_chunks_ids(chunks=chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    # Keep only the existing chunks
    new_chunks = []
    for chunk in chunks_ids:
        if chunk.metadata["id"] in existing_ids:
            new_chunks.append(chunk)

    # Remove the document chunks from the vector store
    if len(new_chunks):
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.delete(ids=new_chunk_ids)

def list_collections(chroma_path: str) -> list[str]:
    """
    List all collections in a Chroma vector store.
    
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :return: A list of collection names.
    :rtype: list[str]
    """
    chroma_path = os.path.abspath(chroma_path)
    db = Chroma(persist_directory=chroma_path)
    existing_collections = db._client.list_collections()
    existing_collections = [col.name for col in existing_collections]
    return existing_collections

def count_collection_items(chroma_path: str, collection_name: str) -> int:
    """
    Count the number of items in a Chroma collection.
    
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :param collection_name: The name of the collection to count items in.
    :type collection_name: str
    :return: The number of items in the collection.
    :rtype: int
    """
    chroma_path = os.path.abspath(chroma_path)
    # Check if the collection exists
    if not is_collection_existing(chroma_path=chroma_path, collection_name=collection_name):
        raise ValueError(f"The collection {collection_name!r} does not exist in the Chroma vector store at {chroma_path!r}.")
    db = Chroma(persist_directory=chroma_path, collection_name=collection_name)
    items = db.get(include=[])
    return len(items["ids"])

def clear_collection(chroma_path: str, collection_name: str) -> None:
    """
    Delete all documents in a Chroma collection.
    
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :param collection_name: The name of the collection to clear.
    :type collection_name: str
    """
    chroma_path = os.path.abspath(chroma_path)
    db = Chroma(persist_directory=chroma_path)
    # Check if the collection already exists
    if is_collection_existing(chroma_path=chroma_path, collection_name=collection_name):
        db = Chroma(persist_directory=chroma_path, collection_name=collection_name)
        db.reset_collection()

def delete_collection(chroma_path: str, collection_name: str) -> None:
    """
    Delete a Chroma collection.
    
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :param collection_name: The name of the collection to drop.
    :type collection_name: str
    """
    chroma_path = os.path.abspath(chroma_path)
    db = Chroma(persist_directory=chroma_path)
    # Check if the collection already exists
    if is_collection_existing(chroma_path=chroma_path, collection_name=collection_name):
        db = Chroma(persist_directory=chroma_path, collection_name=collection_name)
        db.delete_collection()

def get_relevant_docs(
    chroma_path: str,
    collection_name: str,
    embedding_model: str,
    query: str,
    top_k: int,
    reranking_mode: Optional[bool] = False,
    reranking_model: Optional[str] = None
) -> str:
    """
    Query the Chroma vector store to retrieve relevant document chunks.\n
    It supports querying by `top_k` or `similarity_threshold`.\n
    It returns the text of the relevant document chunks concatenated together.
    
    :param chroma_path: The path to the Chroma vector store.
    :type chroma_path: str
    :param collection_name: The name of the collection to query.
    :type collection_name: str
    :param embedding_model: The name of the embedding model.
    :type embedding_model: str
    :param query: The query string.
    :type query: str
    :param top_k: The number of top similar documents to retrieve.
    :type top_k: int
    :param reranking_mode: Whether to use reranking mode (reranking + similarity threshold) or not (top_k). Defaults to False.
    :type reranking_mode: Optional[bool]
    :param reranking_model: The name of the reranker model (required if reranking_mode is True).
    :type reranking_model: Optional[str]
    :return: The concatenated text of the relevant document chunks.
    :rtype: str
    """
    # Check if ollama is running
    if not ollama_utils.is_ollama_active():
        ollama_utils.start_ollama()

    # Check if the chroma_path is a relative path or absolute path
    chroma_path = os.path.abspath(chroma_path)
    embedding_function = _get_embedding(model_name=embedding_model)
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function, collection_name=collection_name)

    filtered_results = []

    # Reranking
    if reranking_mode:
        # Get 3*top_k documents for reranking
        topk_results = db.similarity_search_with_score(query, k=3*top_k)
        reranker = CrossEncoder(reranking_model)
        # Prepare pairs for reranking
        docs = [doc for doc, _ in topk_results]
        pairs = [[query, doc.page_content] for doc in docs]
        # Perform reranking
        rerank_scores = reranker.predict(pairs)
        # Sort by rerank score (descending)
        reranked = list(zip(docs, rerank_scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        # Filter by similarity threshold
        filtered_results = reranked[:top_k]
    else:
        # Just get top_k
        filtered_results = db.similarity_search_with_score(query, k=top_k)

    formatted_docs = []
    
    for idx, (doc, score) in enumerate(filtered_results, 1):
        if doc.metadata:
            # Document with metadata
            metadata_lines = [f"- {key}: {value}" for key, value in doc.metadata.items()]
            metadata_str = "\n".join(metadata_lines)
            chunk = f"""### Document {idx}\n**Metadata:**\n{metadata_str}\n\n**Content:**{doc.page_content}\n**Score:** {score:.4f}"""
        else:
            # Document without metadata
            chunk = f"""### Document {idx}\n**Content:**{doc.page_content}\n**Score:** {score:.4f}"""
        
        formatted_docs.append(chunk)

    # Join all chunks with separator
    docs_selected = "\n\n---\n\n".join(formatted_docs)

    return docs_selected