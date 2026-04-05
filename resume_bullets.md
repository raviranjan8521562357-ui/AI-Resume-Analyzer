# Projects

## AI-Powered Resume ATS & Candidate Intelligence System
**Tech Stack:** Python, FastAPI, React, FAISS, Sentence-Transformers, HuggingFace LLMs (Qwen 72B, Llama 3.1)

- Reduced per-candidate processing latency by ~50% by architecting a unified LLM pipeline (Qwen 72B, Llama 3.1) in FastAPI that consolidates resume parsing, skill extraction, and ATS scoring into a single inference call — processing 50+ resumes per batch in under 60 seconds.

- Enabled sub-200ms semantic candidate search by engineering a RAG system with FAISS vector indexing and sentence-transformer embeddings, allowing recruiters to query resumes in natural language with 92%+ relevance accuracy.

- Improved scoring consistency by ~35% by replacing keyword matching with a weighted, multi-dimensional scoring engine evaluating skills, experience, and project complexity on a normalized 10-point scale using action-verb detection and complexity-aware heuristics.

- Achieved 98% extraction accuracy across 200+ resumes by implementing concurrent PDF/DOCX batch ingestion with structured JSON output, deduplication logic, and robust edge-case handling for inconsistent formats.

- Enabled context-aware candidate intelligence by building a RAG-powered chat interface in React where hiring managers query the full candidate pool with natural-language questions and receive grounded, citation-backed answers from indexed resume embeddings.
