# Fine-Tuning Design Decision

This project uses RAG (Retrieval-Augmented Generation) instead of fine-tuning.

## Why RAG instead of fine-tuning?

Fine-tuning trains a model on a static dataset. For drug interaction information
this approach has a critical flaw — drug labels are updated by manufacturers and
the FDA regularly. A fine-tuned model would become outdated the moment a label
is revised.

RAG retrieves the current FDA label at query time, meaning:
- Answers are always grounded in the most recent available label
- No retraining required when labels are updated
- Every claim is traceable to a specific source document

## If fine-tuning were used
A sample fine-tuning dataset would look like this:

| Input | Output |
|---|---|
| "Do warfarin and ibuprofen interact?" | "Yes — major interaction. Increased bleeding risk. Consult pharmacist." |
| "Is simvastatin safe with clarithromycin?" | "No — contraindicated. Risk of rhabdomyolysis. Avoid combination." |
| "Can I take metformin with lisinopril?" | "No documented interaction in FDA label. Monitor as advised by physician." |

This project intentionally avoids fine-tuning in favor of source-grounded
retrieval for safety and traceability reasons documented in the proposal.