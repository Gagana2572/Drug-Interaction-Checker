from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def load_index(index_path: str = "index/"):
    embeddings = get_embeddings()
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

def retrieve_drug_chunks(drug_names: list, index) -> dict:
    results = {}
    for drug in drug_names:
        query = f"drug interactions and warnings for {drug}"
        docs = index.similarity_search(query, k=3)
        match = None
        for doc in docs:
            if drug.lower() in doc.metadata.get("drug", "").lower():
                match = (doc.page_content, doc.metadata)
                break
        results[drug] = match
    return results

def check_retrieval_completeness(retrieval_results: dict) -> list:
    return [drug for drug, result in retrieval_results.items() if result is None]