# accenture-daie-tech

## Solution overview

The solution consists of:
- **A Medallion ETL Pipeline:** Cleaning raw data into "Silver" layer.
- **Feature Engineering:** Aggregating "Silver" tables into a "Gold" table to create a feature set for customer transactions.
- **RAG Inference Service:** A retrieval-augmented generation pipeline that grounds an LLM in internal policy documents to reduce compliance risks
- **A Backend Server:** Small server that serves the LLM RAG solution and the feature set for customer transactions.


## Setting up the project

### 1. Clone the repository:
````
git clone https://github.com/ericbanzuzi/accenture-daie-tech
cd accenture-daie-tech
````

### 2. Create env and install the required packages
````
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

## Run the solution

### 1. Data Pipeline & ETL
To run the solution for this part use the following command at the root of the project:
````
python etl.py
````

It runs the ETL by saving the files locally into `./data/` as `{filename}_silver.csv`.

### 2. Feature Engineering + Simple Logic
To run the solution for this part use the following command at the root of the project:
````
python features.py
````

It runs the feature engineering logic and creates a small feature set per customer into `./data/customer_features_gold.csv`.

### 3. Data Pipeline & ETL
To run the solution use thefollowing command at the root of the project:
````
python rag.py
````

It starts an interactive CLI loop, where you can answer questions to an LLM and see the responses.

### 4. Serve / Output
To start the FastAPI backend server that serves the simple RAG and customer level feature, run the following command:
````
python backend.py
````

The server runs in [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The end point `/customer/{customer_id}` will give access to the features for a specific customer. Furthermore, the endoint `/query-llm/` gives access to prompt the LLM rag component. Use it as `/query-llm/?q=WRITE YOUR PROMPT`.


## Key assumptions

1. For feature engineering I assumed that the customer transactions are still relevant for customers with signup dates later than the first transaction. A vast majority of customers had first transaction before the signup. Trade-off: inconsistency and noise in features.
2. I assumed that the documents can be stored in memory locally for RAG in this solution as they are very small. Trade-off: larger scale solution would require chunking and proper vectore database e.g. Pinecone.


## Future directions

I would invest more time into prompt engineering of the LLM. Now it sometimes uses knowledge outside of the decuments. Another thing would be to further explore the data so that the feature set for each customer could become richer and more accurate. 