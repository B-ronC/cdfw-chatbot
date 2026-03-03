from langchain_ollama import OllamaEmbeddings  

# embedding function
def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding",
    )
    return embeddings