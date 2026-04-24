# evaluate.py

import pandas as pd
from src.ingest import build_index
from src.retriever import retrieve_drug_chunks, check_retrieval_completeness
from src.generator import generate_interaction_summary

BENCHMARK = [
    # HIGH SEVERITY (20 pairs)
    {"drugs": ["warfarin", "ibuprofen"],          "expected_severity": "major",           "category": "high"},
    {"drugs": ["simvastatin", "clarithromycin"],   "expected_severity": "contraindicated", "category": "high"},
    {"drugs": ["methotrexate", "ibuprofen"],       "expected_severity": "major",           "category": "high"},
    {"drugs": ["warfarin", "aspirin"],             "expected_severity": "major",           "category": "high"},
    {"drugs": ["lithium", "ibuprofen"],            "expected_severity": "major",           "category": "high"},
    {"drugs": ["clopidogrel", "omeprazole"],       "expected_severity": "major",           "category": "high"},
    {"drugs": ["warfarin", "fluconazole"],         "expected_severity": "major",           "category": "high"},
    {"drugs": ["simvastatin", "amiodarone"],       "expected_severity": "contraindicated", "category": "high"},
    {"drugs": ["digoxin", "amiodarone"],           "expected_severity": "major",           "category": "high"},
    {"drugs": ["warfarin", "metronidazole"],       "expected_severity": "major",           "category": "high"},
    {"drugs": ["theophylline", "ciprofloxacin"],   "expected_severity": "major",           "category": "high"},
    {"drugs": ["warfarin", "ciprofloxacin"],       "expected_severity": "major",           "category": "high"},
    {"drugs": ["phenytoin", "fluconazole"],        "expected_severity": "major",           "category": "high"},
    {"drugs": ["simvastatin", "gemfibrozil"],      "expected_severity": "contraindicated", "category": "high"},
    {"drugs": ["warfarin", "amiodarone"],          "expected_severity": "major",           "category": "high"},
    {"drugs": ["lithium", "lisinopril"],           "expected_severity": "major",           "category": "high"},
    {"drugs": ["warfarin", "naproxen"],            "expected_severity": "major",           "category": "high"},
    {"drugs": ["digoxin", "clarithromycin"],       "expected_severity": "major",           "category": "high"},
    {"drugs": ["warfarin", "fluoxetine"],          "expected_severity": "major",           "category": "high"},
    {"drugs": ["carbamazepine", "clarithromycin"], "expected_severity": "major",           "category": "high"},

    # MODERATE (20 pairs)
    {"drugs": ["lisinopril", "ibuprofen"],         "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["atorvastatin", "clarithromycin"],  "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["amlodipine", "simvastatin"],       "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["sertraline", "ibuprofen"],         "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["lisinopril", "potassium"],         "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["metoprolol", "diphenhydramine"],   "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["gabapentin", "hydrocodone"],       "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["fluoxetine", "alprazolam"],        "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["atorvastatin", "amlodipine"],      "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["lisinopril", "spironolactone"],    "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["metoprolol", "verapamil"],         "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["warfarin", "acetaminophen"],       "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["sertraline", "tramadol"],          "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["gabapentin", "naproxen"],          "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["metformin", "lisinopril"],         "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["atorvastatin", "digoxin"],         "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["fluoxetine", "tramadol"],          "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["amlodipine", "metoprolol"],        "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["omeprazole", "clopidogrel"],       "expected_severity": "moderate",        "category": "moderate"},
    {"drugs": ["sertraline", "alprazolam"],        "expected_severity": "moderate",        "category": "moderate"},

    # CONTROL — no interaction (10 pairs)
    {"drugs": ["metformin", "atorvastatin"],       "expected_severity": "none",            "category": "control"},
    {"drugs": ["lisinopril", "atorvastatin"],      "expected_severity": "none",            "category": "control"},
    {"drugs": ["metformin", "amlodipine"],         "expected_severity": "none",            "category": "control"},
    {"drugs": ["omeprazole", "metformin"],         "expected_severity": "none",            "category": "control"},
    {"drugs": ["acetaminophen", "lisinopril"],     "expected_severity": "none",            "category": "control"},
    {"drugs": ["atorvastatin", "metoprolol"],      "expected_severity": "none",            "category": "control"},
    {"drugs": ["amlodipine", "omeprazole"],        "expected_severity": "none",            "category": "control"},
    {"drugs": ["metformin", "omeprazole"],         "expected_severity": "none",            "category": "control"},
    {"drugs": ["lisinopril", "metoprolol"],        "expected_severity": "none",            "category": "control"},
    {"drugs": ["acetaminophen", "metformin"],      "expected_severity": "none",            "category": "control"},
]


def score_output(output: str, expected_severity: str) -> dict:
    """
    Scores each output on 4 criteria worth 1 point each.
    Target: 3/4 or higher is a passing output.
    """
    output_lower = output.lower()

    # Criterion 1: Was an interaction identified?
    interaction_identified = any(word in output_lower for word in [
        "interaction", "risk", "warning", "caution", "avoid", "monitor", "contraindicated"
    ])

    # Criterion 2: Was severity mentioned?
    severity_mentioned = any(word in output_lower for word in [
        "major", "moderate", "minor", "contraindicated", "severe", "significant", "none"
    ])

    # Criterion 3: Was a recommended action given?
    action_given = any(word in output_lower for word in [
        "consult", "pharmacist", "physician", "doctor", "monitor", "avoid", "do not"
    ])

    # Criterion 4: Was a source citation present?
    citation_present = any(word in output_lower for word in [
        "fda", "dailymed", "label", "source", "set_id"
    ])

    score = sum([interaction_identified, severity_mentioned, action_given, citation_present])

    return {
        "interaction_identified": interaction_identified,
        "severity_mentioned": severity_mentioned,
        "action_given": action_given,
        "citation_present": citation_present,
        "score": score,
        "passed": score >= 3
    }


def run_evaluation():
    print("Starting evaluation of 50 drug pair benchmark...")
    print("=" * 50)

    results = []
    silent_failure_count = 0
    passed_count = 0
    total_scored = 0

    for i, case in enumerate(BENCHMARK):
        print(f"Running case {i+1}/{len(BENCHMARK)}: {case['drugs']}")

        try:
            index = build_index(case["drugs"])
            retrieval = retrieve_drug_chunks(case["drugs"], index)
            missing = check_retrieval_completeness(retrieval)

            if missing:
                silent_failure_count += 1
                results.append({
                    "case_number": i + 1,
                    "drugs": str(case["drugs"]),
                    "category": case["category"],
                    "expected_severity": case["expected_severity"],
                    "retrieval_complete": False,
                    "missing_drugs": str(missing),
                    "output": "BLOCKED — missing labels for: " + str(missing),
                    "score": 0,
                    "passed": False,
                    "interaction_identified": False,
                    "severity_mentioned": False,
                    "action_given": False,
                    "citation_present": False
                })

            else:
                summary = generate_interaction_summary(case["drugs"], retrieval)
                scores = score_output(summary, case["expected_severity"])
                total_scored += 1

                if scores["passed"]:
                    passed_count += 1

                results.append({
                    "case_number": i + 1,
                    "drugs": str(case["drugs"]),
                    "category": case["category"],
                    "expected_severity": case["expected_severity"],
                    "retrieval_complete": True,
                    "missing_drugs": "none",
                    "output": summary,
                    "score": scores["score"],
                    "passed": scores["passed"],
                    "interaction_identified": scores["interaction_identified"],
                    "severity_mentioned": scores["severity_mentioned"],
                    "action_given": scores["action_given"],
                    "citation_present": scores["citation_present"]
                })

        except Exception as e:
            print(f"  ERROR on case {i+1}: {e}")
            results.append({
                "case_number": i + 1,
                "drugs": str(case["drugs"]),
                "category": case["category"],
                "expected_severity": case["expected_severity"],
                "retrieval_complete": False,
                "missing_drugs": "error",
                "output": f"ERROR: {str(e)}",
                "score": 0,
                "passed": False,
                "interaction_identified": False,
                "severity_mentioned": False,
                "action_given": False,
                "citation_present": False
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)

    # Print summary
    total = len(BENCHMARK)
    complete = df["retrieval_complete"].sum()
    pass_rate = (passed_count / total_scored * 100) if total_scored > 0 else 0
    silent_failure_rate = (silent_failure_count / total * 100)

    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total queries run:         {total}")
    print(f"Retrieval complete:        {complete}/{total}")
    print(f"Silent failure rate:       {silent_failure_rate:.1f}%  (target: 0%)")
    print(f"Queries scored:            {total_scored}")
    print(f"Passed (score >= 3/4):     {passed_count}/{total_scored}")
    print(f"Overall pass rate:         {pass_rate:.1f}%  (target: >= 80%)")
    print(f"{'='*50}")
    print(f"Results saved to:          evaluation_results.csv")
    print(f"Open in Excel to review individual outputs.")


if __name__ == "__main__":
    run_evaluation()