from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "./chroma_langchain_db"

PROMPT_TEMPLATE = """
You are the official CDFW Freshwater Sport Fishing Regulations assistant.

STRICT RULES:
- Use ONLY the provided regulation text.
- Do NOT guess or use outside knowledge.
- Return ONLY one short answer.
- The answer must closely match the wording used in the regulations (extractive style).
- Include relevant details such as specific conditions, exceptions, or limitations mentioned in the regulations.
- Prefer exact phrases or minimal edits from the regulation text.
- Keep answers short and direct.

FORMAT RULES:
- End the answer with a citation in exactly this format:
  (§SECTION, page PAGE)

- If the answer is not explicitly stated, output exactly:
The regulations provided do not specify.

REGULATIONS:
{context}

QUESTION:
{question}

ANSWER:
"""

def main():
    model = OllamaLLM(
    model="llama3.1",
    temperature=0,     
    num_ctx=4096, 
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | model

    # load database
    embedding_function = get_embedding_function()
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # create retriever
    retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,         
        "fetch_k": 40,    
        "lambda_mult": 0.75,  
    }
    )

    # create CLI
    while True:
        print("\n\n--------------------------------------------------------")
        question = input("Ask your question (q to quit): ")
        if question.lower() == "q":
            break

        # retrieve relevant chunks from the database
        docs = retriever.invoke(question)
        context = format_docs(docs)

        result = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )
        print(f"\nAnswer: {result}")

def format_docs(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "n/a")
        chunk_id = d.metadata.get("id", "")
        lines.append(f"[source={src} page={page} id={chunk_id}]\n{d.page_content}")
    return "\n\n---\n\n".join(lines)

if __name__ == "__main__":
    main()