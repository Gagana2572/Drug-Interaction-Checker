# Drug Interaction Checker

An AI-powered drug interaction checker for non-expert caregivers, grounded in official FDA label data using Retrieval-Augmented Generation (RAG).

> Built for INFO 7375 Generative AI · Northeastern University
> By **Gagana M** & **Madhumitha Nandhikatti**

---

## 🔗 Links

- 🚀 **Live Demo**: [Click here](https://drug-interaction-checker-ndqfsmpfggqqybc32vjlve.streamlit.app/)

---

## 🧠 What It Does

Most caregivers managing elderly patients with multiple medications turn to general-purpose AI chatbots for drug interaction information — the most dangerous option available to them. General LLMs generate answers from training memory with no source traceability, no staleness detection, and no retrieval completeness guarantee.

This system is architecturally different. It:

1. Accepts drug names as input (brand or generic)
2. Normalizes brand names to generic via the NIH RxNorm API
3. Retrieves the official FDA drug label for each drug in real time
4. Verifies that a label was found for every drug before generating output
5. Synthesizes a plain-English interaction summary grounded in the retrieved label text
6. Cites the exact FDA label document for every claim

The model is not remembering. It is reading.

---

## 🏗️ Architecture

User Input → RxNorm Normalizer → FDA OpenFDA API → FAISS Vector Store → Completeness Check → LLaMA 3.3 70B → Summary + Citations

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.1 8B via Groq API |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers, local) |
| Vector Store | FAISS (local) |
| RAG Framework | LangChain |
| Drug Label Data | FDA OpenFDA API |
| Brand→Generic | NIH RxNorm API |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 🚀 Setup Instructions

**1. Clone the repo**

    git clone https://github.com/Gagana2572/Drug-Interaction-Checker.git
    cd Drug-Interaction-Checker

**2. Create and activate virtual environment**

    python -m venv venv
    source venv/bin/activate
    # Windows: venv\Scripts\activate

**3. Install dependencies**

    pip install -r requirements.txt

**4. Add your Groq API key**

Copy `.env.example` to `.env` and replace `your_groq_key_here` with your actual key from https://console.groq.com

**5. Run the app**

    streamlit run app.py

---

## 🧪 How to Use

1. Type one or more drug names separated by commas
2. Brand names work too — "Advil" is automatically recognized as "ibuprofen"
3. Click **Check Interactions**
4. Read the plain-English summary with severity level and recommended action
5. Check the cited FDA label sources at the bottom

**Example inputs to try:**

- `warfarin, ibuprofen` — major interaction
- `Advil, warfarin` — tests brand name normalization
- `simvastatin, clarithromycin` — contraindicated interaction
- `metformin, lisinopril` — minor/no interaction
- `warfarin, fakeDrug999` — tests completeness warning

---

## 🛡️ Key Safety Feature: Completeness Check

The most dangerous output this system could produce is a confidently incomplete answer — where one drug label fails to retrieve silently and the model generates a partial interaction report that looks complete.

To prevent this, before any output is generated the system verifies that an FDA label was successfully retrieved for every drug in the query. If any label is missing, a visible warning is shown and the user is directed to consult a pharmacist directly.

---

## 📊 Evaluation

Run the benchmark evaluation:

    python evaluate.py

This runs 50 drug pair queries across three categories:
- 20 high severity (contraindicated/major) pairs
- 20 moderate interaction pairs
- 10 control cases (no documented interaction)

Results are saved to `evaluation_results.csv`.

Target metrics:
- Silent failure rate: **0%**
- Overall pass rate: **≥ 80%**

---

## ⚖️ Ethical Considerations

- **Data source**: FDA DailyMed only — federally maintained, publicly available, authoritative by legal definition. No scraped web content.
- **Privacy**: No personal health data collected. Users enter drug names only. System is stateless — nothing stored between sessions.
- **Scope**: Explicitly limited to FDA-approved US drug labels. Non-US drug names may not be found.
- **Disclaimer**: Every output carries a persistent disclaimer — this tool is for educational purposes only and is not a substitute for a licensed pharmacist or physician.
- **Transparency**: Every claim is cited back to a specific FDA label document with ID and last updated date.

---

## 📁 Project Structure

    drug-interaction-checker/
    ├── src/
    │   ├── __init__.py
    │   ├── normalizer.py      # Brand → generic via RxNorm
    │   ├── ingest.py          # FDA label fetch + FAISS index builder
    │   ├── retriever.py       # Vector search + completeness check
    │   └── generator.py       # Prompt engineering + Groq LLM call
    ├── docs/
    │   └── index.html         # GitHub Pages web page
    ├── app.py                 # Streamlit web app
    ├── evaluate.py            # Evaluation pipeline
    ├── requirements.txt
    ├── .env.example
    └── README.md

---

## ⚠️ Disclaimer

This tool summarizes FDA label information for educational purposes only. It is not a substitute for consultation with a licensed pharmacist or physician. Do not make medication decisions based solely on this output.
