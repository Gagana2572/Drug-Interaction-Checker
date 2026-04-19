# app.py
import streamlit as st
from src.normalizer import brand_to_generic
from src.ingest import build_index
from src.retriever import retrieve_drug_chunks, check_retrieval_completeness
from src.generator import generate_interaction_summary

st.title("Drug Interaction Checker")
st.write("Enter two or more drugs to check for interactions.")

drug_input = st.text_input("Enter drug names separated by commas (e.g. Advil, warfarin)")

if st.button("Check Interactions"):
    if drug_input:
        with st.spinner("Checking interactions..."):
            drugs = [d.strip() for d in drug_input.split(",")]
            generic_drugs = [brand_to_generic(d) for d in drugs]
            st.write(f"Checking: {', '.join(generic_drugs)}")

            index = build_index(generic_drugs)
            results = retrieve_drug_chunks(generic_drugs, index)
            missing = check_retrieval_completeness(results)

            if missing:
                st.warning(f"No FDA data found for: {', '.join(missing)}")

            if any(v is not None for v in results.values()):
                summary = generate_interaction_summary(generic_drugs, results)
                st.markdown(summary)
            else:
                st.error("Could not find FDA label data for any of the drugs entered.")
    else:
        st.warning("Please enter at least two drug names.")
