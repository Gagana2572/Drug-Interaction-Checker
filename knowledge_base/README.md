# Knowledge Base

This project uses the FDA OpenFDA API as its knowledge base for RAG.

## Data Source
- **Source**: FDA DailyMed via OpenFDA API
- **URL**: https://api.fda.gov/drug/label.json
- **Type**: Official FDA drug labels (publicly available, federally maintained)
- **Sections used**: `drug_interactions` and `warnings` fields only

## How it works
The knowledge base is built at runtime:
1. Drug names are submitted as queries to the FDA OpenFDA API
2. The `drug_interactions` and `warnings` sections are extracted
3. Text is converted to embeddings using `all-MiniLM-L6-v2`
4. Embeddings are stored in a local FAISS vector index in the `index/` folder

## Why not a static knowledge base?
Using the live FDA API ensures the most current label data is always retrieved.
A static pre-built index would risk using outdated label information.

## Sample FDA Label Data
Below is a sample of the raw data structure returned by the FDA API for ibuprofen:

```json
{
  "drug_interactions": [
    "Warfarin: Ibuprofen can increase the anticoagulant effect of warfarin...",
    "Aspirin: Concomitant use of ibuprofen and aspirin is not recommended..."
  ],
  "warnings": [
    "Cardiovascular Risk: NSAIDs may cause an increased risk of serious cardiovascular..."
  ],
  "set_id": "abc123",
  "effective_time": "20230101"
}
```