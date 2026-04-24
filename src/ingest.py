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
        
        interactions = " ".join(result.get("drug_interactions", ["No interaction data found"]))
        warnings = " ".join(result.get("warnings", ["No warnings data found"]))
        set_id = result.get("set_id", "unknown")
        effective_time = result.get("effective_time", "unknown")
        
        return {
            "drug_name": drug_name,
            "drug": drug_name,
            "interaction_text": interactions,
            "warnings": warnings,
            "set_id": set_id,
            "effective_time": effective_time
        }
    except Exception:
        return None

def build_index(drug_list: list, save_path: str = "index"):
    documents = []
    metadatas = []
    
    for drug in drug_list:
        label = fetch_fda_label(drug)
        if label:
            chunk = (
                f"Drug: {label['drug_name']}\n\n"
                f"Drug Interactions:\n{label['interaction_text']}\n\n"
                f"Warnings:\n{label['warnings']}"
            )
            documents.append(chunk)
            metadatas.append({
                "drug_name": label["drug_name"],
                "drug": label["drug_name"],
                "set_id": label["set_id"],
                "last_updated": label["effective_time"],
                "source": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={drug}"
            })
        else:
            print(f"Could not fetch label for: {drug}")

    embeddings = get_embeddings()
    os.makedirs(save_path, exist_ok=True)
    vector_store = FAISS.from_texts(documents, embeddings, metadatas=metadatas)
    vector_store.save_local(save_path)
    print(f"Index saved to {save_path}")

    return vector_store