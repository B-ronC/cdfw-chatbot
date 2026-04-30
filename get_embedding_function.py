from langchain_ollama import OllamaEmbeddings  

# embedding function
def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embeddings