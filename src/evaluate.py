# evaluate.py

import pandas as pd
from src.ingest import build_index
from src.retriever import retrieve_drug_chunks, check_retrieval_completeness
from src.generator import generate_interaction_summary

BENCHMARK = [
    # High severity — add 20 total
    {"drugs": ["warfarin", "ibuprofen"],         "expected_severity": "major",           "category": "high"},
    {"drugs": ["simvastatin", "clarithromycin"],  "expected_severity": "contraindicated", "category": "high"},
    {"drugs": ["methotrexate", "ibuprofen"],      "expected_severity": "major",           "category": "high"},

    # Moderate — add 20 total
    {"drugs": ["lisinopril", "ibuprofen"],        "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["metformin", "alcohol"],           "expected_severity": "moderate",        "category": "moderate"},

    # Controls (no interaction) — add 10 total
    {"drugs": ["metformin", "lisinopril"],        "expected_severity": "none",            "category": "control"},

    # Keep adding until you reach 50 total...
]

def run_evaluation():
    results = []
    silent_failure_count = 0

    for i, case in enumerate(BENCHMARK):
        print(f"Running case {i+1}/{len(BENCHMARK)}: {case['drugs']}")
        
        index = build_index(case["drugs"])
        retrieval = retrieve_drug_chunks(case["drugs"], index)
        missing = check_retrieval_completeness(retrieval)

        if missing:
            silent_failure_count += 1
            results.append({
                "drugs": str(case["drugs"]),
                "category": case["category"],
                "expected_severity": case["expected_severity"],
                "retrieval_complete": False,
                "missing_drugs": str(missing),
                "output": "BLOCKED — missing labels for: " + str(missing)
            })
        else:
            summary = generate_interaction_summary(case["drugs"], retrieval)
            results.append({
                "drugs": str(case["drugs"]),
                "category": case["category"],
                "expected_severity": case["expected_severity"],
                "retrieval_complete": True,
                "missing_drugs": "none",
                "output": summary
            })

    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    
    total = len(BENCHMARK)
    complete = df["retrieval_complete"].sum()
    
    print(f"\n{'='*40}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*40}")
    print(f"Total queries:        {total}")
    print(f"Retrieval complete:   {complete}/{total}")
    print(f"Silent failure rate:  {silent_failure_count/total:.1%}  (target: 0%)")
    print(f"Results saved to:     evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()