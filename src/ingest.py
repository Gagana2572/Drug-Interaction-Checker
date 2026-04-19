# src/ingest.py
import requests
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def fetch_fda_label(drug_name: str):
    url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug_name}&limit=1"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        result = data["results"][0]
        interactions = result.get("drug_interactions", ["No interaction data found"])[0]
        return {
            "drug": drug_name,
            "interaction_text": interactions
        }
    except Exception:
        return None

def build_index(drug_list: list, save_path: str = "index"):
    documents = []
    metadatas = []
    for drug in drug_list:
        label = fetch_fda_label(drug)
        if label:
            documents.append(label["interaction_text"])
            metadatas.append({"drug": drug})

    embeddings = get_embeddings()
    os.makedirs(save_path, exist_ok=True)
    vector_store = FAISS.from_texts(documents, embeddings, metadatas=metadatas)
    vector_store.save_local(save_path)
    print(f"Index saved to {save_path}")

    return vector_store
