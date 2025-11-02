# EDA-RAG

A reusable pipeline that:
1) Runs fast, structured EDA on any CSV.
2) Converts EDA outputs (schema, stats, correlations, leakage checks) into text chunks.
3) Embeds & indexes those chunks for retrieval.
4) Lets an LLM answer dataset questions grounded in the EDA artifacts.