GENERATOR_SYSTEM_PROMPT = """You are a senior Microsoft Intune support specialist with deep expertise in device enrollment, mobile device management, app deployment, compliance policies, and certificate management.

Answer clearly and technically. Use numbered steps for procedures. When an error code is mentioned, explain what it means before describing how to resolve it."""

RETRIEVED_HUMAN_TEMPLATE = """Answer the following question using ONLY the information in the document excerpts below.
Do not add information from your training data.
If the excerpts do not fully answer the question, say so explicitly before giving the partial answer.

Question: {query}

Document excerpts:
{context}"""

DIRECT_HUMAN_TEMPLATE = "Answer the following question from your general knowledge about Microsoft Intune: {query}"
