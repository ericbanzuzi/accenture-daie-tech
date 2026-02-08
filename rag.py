from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from dotenv import load_dotenv


SYSTEM_PROMPT = """
You are a helpful assistant for a Nordic client's Customer Service team. 
Use the following context from internal policy documents to answer the user's question. 
Do not use outside knowledge and do not make up new information.
Always maintain a helpful and professional tone. 

Context:
{context}
"""


def load_docs(path: str = "./data/documents/") -> list:
    """Load documents from a text file and return them as a list."""
    p = Path(path)
    docs = []
    for f in p.glob('*.txt'):
        content = ""
        with open(f, 'r') as file:
            # Clean the content of each row by stripping whitespace
            for line in file.readlines():
                content += line.strip() + "\n"
        docs.append(content.strip())

    # FOR LARGER DOCUMENTS, consider chunking the documents into smaller pieces for better retrieval performance
    return docs


def index_docs_to_db(docs: list, model_name: str = "all-MiniLM-L6-v2") -> FAISS:
    """Index the documents using FAISS and return the in-memory vector store database."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_db = FAISS.from_texts(docs, embeddings, distance_strategy=DistanceStrategy.COSINE)
    return vector_db


def run_rag(query: str, vector_db: FAISS, llm: InferenceClient, top_k: int = 1, chat_history: list = [], verbose: bool = False) -> str:
    """Run the RAG pipeline: retrieve relevant context and generate an answer."""
    # Retrieve relevant policy rule using cosine similarity search
    context = vector_db.similarity_search(query, k=top_k)[0].page_content

    if verbose:
        print(f"\nRetrieved context for query '{query}':\n{context}\n")
    
    # Generate answer using the LLM
    llm_prompt = SYSTEM_PROMPT.format(context=context)

    if chat_history:
        messages = [{"role": "system", "content": llm_prompt}] + chat_history + [{"role": "user", "content": query}]
    else:
        messages = [
            {"role": "system", "content": llm_prompt},
            {"role": "user", "content": query},
        ]

    output = llm.chat.completions.create(
        messages=messages, temperature=0.2, max_tokens=300
    )
    return output.choices[0].message.content.strip()


# THIS FUNCTION CAN REPLACE load_docs() IF YOU WANT TO CHUNK THE DOCUMENTS BEFORE INDEXING
# IT IS UNUSED IN THE IMPLEMENTATION
def load_and_chunk_docs(path: str = "./data/documents/") -> list:
    """Load documents and split them into smaller chunks for better retrieval."""
    p = Path(path)
    raw_text_list = []
    for f in p.glob('*.txt'):
        with open(f, 'r') as file:
            raw_text_list.append(file.read().strip())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,  
        chunk_overlap=64,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.create_documents(raw_text_list)
    return [doc.page_content for doc in docs]

if __name__ == "__main__":
    # docs = load_and_chunk_docs()
    # print(f"Total documents after chunking: {len(docs)}")
    # print(docs)

    load_dotenv()
    docs = load_docs()
    vector_db = index_docs_to_db(docs)
    client = InferenceClient(model="meta-llama/Llama-3.2-1B-Instruct")

    chat_history = []
    print("--- Policy Assistant Active (Type 'quit' to stop) ---")
    while True:
        user_input = input("\nUser query: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Closing Assistant. Goodbye!")
            break
        
        if not user_input.strip():
            continue

        # Get the answer and update history inside run_rag
        answer = run_rag(query=user_input, 
                         vector_db=vector_db, 
                         llm=client, 
                         chat_history=chat_history)
        
        print(f"\nLLM answer: {answer}")