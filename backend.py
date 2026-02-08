import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from rag import load_docs, index_docs_to_db, run_rag
from huggingface_hub import InferenceClient
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file


DATA_DIR = Path("data")
app = FastAPI()
llm = InferenceClient(model="meta-llama/Llama-3.2-1B-Instruct")


def init_vector_db():
    """Initialize the vector database by loading documents and indexing them."""
    docs = load_docs()
    vector_db = index_docs_to_db(docs)
    return vector_db


def init_customer_data():
    """Initialize customer feature data."""
    customer_data_file = DATA_DIR / "customer_features_gold.csv"
    if customer_data_file.exists():
        return pd.read_csv(customer_data_file)
    else:
        from etl import run_pipeline
        from features import create_features
        customer_clean, transactions_clean = run_pipeline(save_files=True)
        features_df = create_features(customer_clean, transactions_clean, save_file=True)
        return features_df


def init_server():
    global customer_df, vector_db
    customer_df = init_customer_data()
    vector_db = init_vector_db()


# Load data and initialize vector database at startup
init_server()


@app.get("/")
def hello():
    return "Hello world! The API is up and running :)"
    

@app.get("/query-llm/")
def query_llm(q: str):
    """Serve a query to the LLM and return the response.
       NOTE: A post request would be more appropriate for this endpoint in a production setting, but for simplicity we are using GET here.
    """
    try:
        answer = run_rag(query=q, vector_db=vector_db, llm=llm)
        return {
            "query": q, 
            "llm_response": answer
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/customer/{customer_id}")
def get_customer_features(customer_id: int):
    """Retrieves customer transaction insights for a given customer."""
    result = customer_df[customer_df["customer_id"] == customer_id].to_dict(orient="records")
    if not result:
        raise HTTPException(status_code=404, detail="Customer not found")
    return result[0]
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)