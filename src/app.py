# app.py

import streamlit as st
from src.normalizer import brand_to_generic
from src.ingest import build_index
from src.retriever import retrieve_drug_chunks, check_retrieval_completeness
from src.generator import generate_interaction_summary

st.set_page_config(
    page_title="Drug Interaction Checker",
    page_icon="💊",
    layout="centered"
)

st.title("💊 Drug Interaction Checker")
st.caption("For non-expert caregivers — grounded in FDA label data, not AI memory")
st.markdown("---")

drug_input = st.text_input(
    "Enter drug names, separated by commas:",
    placeholder="e.g. Advil, warfarin, lisinopril"
)

if st.button("Check Interactions", type="primary") and drug_input:

    raw_drugs = [d.strip() for d in drug_input.split(",") if d.strip()]

    # Stage 1: Normalize brand → generic
    with st.spinner("Looking up drug names..."):
        generic_drugs = []
        for drug in raw_drugs:
            generic = brand_to_generic(drug)
            if generic != drug.lower():
                st.success(f"✅ Brand name **'{drug}'** → recognized as generic: **'{generic}'**")
            generic_drugs.append(generic)

    # Show which drugs are being checked
    st.info(f"🔍 Checking interactions for: **{', '.join(generic_drugs)}**")

    # Stage 2: Fetch FDA labels + build index
    with st.spinner("Retrieving FDA label data..."):
        try:
            index = build_index(generic_drugs)
            retrieval_results = retrieve_drug_chunks(generic_drugs, index)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()

    # Stage 3: Completeness check — safety gate
    missing = check_retrieval_completeness(retrieval_results)

    if missing:
        st.warning(
            f"⚠️ **Incomplete data warning:** FDA label data could not be retrieved for: "
            f"**{', '.join(missing)}**.\n\n"
            f"The summary below covers only the drugs that were found. "
            f"**Please consult a pharmacist directly before making any decisions.**"
        )

    # Stage 4: Generate summary
    found = {k: v for k, v in retrieval_results.items() if v is not None}

    if found:
        with st.spinner("Generating interaction summary from FDA labels..."):
            try:
                summary = generate_interaction_summary(list(found.keys()), found)
                st.markdown("## 📋 Interaction Summary")
                with st.container(border=True):
                    st.markdown(summary)
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    st.warning(
                        "⚠️ The AI service is temporarily busy. "
                        "Please wait 30 seconds and try again."
                    )
                else:
                    st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.error(
            "❌ No FDA label data could be retrieved for any of the entered drugs. "
            "Please check spelling and try again, or consult a pharmacist directly."
        )