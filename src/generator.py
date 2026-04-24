# src/generator.py

import groq
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a pharmacist assistant helping non-expert caregivers understand 
drug interactions. You summarize FDA drug label information in plain English.
You MUST follow these rules strictly:
1. Only state facts that are present in the FDA label text given to you.
2. If the label does not mention an interaction, say "No interaction documented in FDA label."
3. Always state severity: contraindicated / major / moderate / minor / none documented.
4. Always state a recommended action: avoid combination / consult pharmacist / monitor symptoms.
5. Use simple words. No medical jargon. Write for someone with no medical training.
6. Do NOT fill gaps from general knowledge. If it is not in the label, do not say it.
7. Always end your response with this exact disclaimer, word for word:
Warning: This tool summarizes FDA label information for educational purposes only. It is not a 
substitute for consultation with a licensed pharmacist or physician. Do not make medication 
decisions based solely on this output."""

def get_api_key():
    """Always fetch fresh API key — never cached."""
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")

def generate_interaction_summary(drug_names: list, retrieval_results: dict) -> str:
    
    # Create fresh client every call using latest key
    client = groq.Groq(api_key=get_api_key())
    
    context_parts = []
    citations = []

    for drug, result in retrieval_results.items():
        if result:
            chunk_text, metadata = result
            context_parts.append(f"=== FDA LABEL: {drug.upper()} ===\n{chunk_text}")
            citations.append(metadata)

    context = "\n\n".join(context_parts)
    context = context[:3000]
    drugs_str = " and ".join(drug_names)

    user_message = f"""Using ONLY the FDA label excerpts below, explain the interaction 
risks between {drugs_str} to a non-expert caregiver.
For each drug pair, clearly state:
1. Does an interaction exist? (yes / no / unknown based on label)
2. Severity level
3. What symptoms or risks to watch for
4. What the caregiver should do

FDA LABEL DATA:
{context}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=512
        )
    except Exception as e:
        if "rate_limit" in str(e).lower():
            return f" Rate limit hit. Full error: {str(e)}"
        if "invalid_api_key" in str(e).lower() or "authentication" in str(e).lower():
            return " API key error. Please check your Groq API key in Streamlit secrets."
        return f" An error occurred: {str(e)}"

    summary = response.choices[0].message.content

    citation_block = "\n\n---\nSources (FDA DailyMed):\n"
    for c in citations:
        citation_block += (
            f"- **{c.get('drug_name', 'unknown').title()}**: "
            f"Label ID: `{c.get('set_id', 'unknown')}` | "
            f"Last updated: `{c.get('last_updated', 'unknown')}`\n"
            f"  🔗 {c.get('source', '')}\n"
        )

    return summary + citation_block