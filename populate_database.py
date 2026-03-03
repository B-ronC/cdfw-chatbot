import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

CHROMA_PATH = "./chroma_langchain_db"
DATA_PATH = "./data"

def main():
    # database reset
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

# load documents from the "data" directory
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# split documents into chunks
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

# create unique IDs for each chunk
# return chunks with chunk_id in meta-data
def create_chunk_ids(chunks: list[Document]):
    # ID = Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # get chunk page info
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # calculate chunk index in page
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # create chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # add to chunk meta-data
        chunk.metadata["id"] = chunk_id

    return chunks

def add_to_chroma(chunks: list[Document]):
    # load existing database
    vector_store = Chroma(
        embedding_function=get_embedding_function(),
        persist_directory=CHROMA_PATH,
    )

    # get chunks with IDs
    chunks_with_ids = create_chunk_ids(chunks)

    # get existing documents / ids in database
    existing_items = vector_store.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # only add documents that do not exist in database
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_store.add_documents(documents=new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
if __name__ == "__main__":
    main()