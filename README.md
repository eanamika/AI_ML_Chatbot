# AI_ML_Chatbot
# A Retrieval-Augmented AI Chatbot built using SBERT, pgvector, FlagReranker, and Llama 3.1 70B.
This project presents a Large Language Model–powered AI/ML chatbot designed to answer domain-specific questions with high accuracy.
It integrates a semantic search IR pipeline, LLM-based answer generation, and a Flask web portal with user feedback logging.

# System Architecture 
User Query → SBERT Embedding → Vector Search (pgvector) → Top-15 Results
→ Reranking (FlagReranker) → Top-5 Context
→ Llama 3.1 70B (Groq API) → Final Answer → Web Portal
→ User Feedback → PostgreSQL

# Methodology
1. Data Collection
2. Data Cleaning
3. Data Storage

# IR PipeLine
Query Processing -> Similarity Search -> Reranking -> Answer Generation

# LLM Integration (RAG)
Model Used: Llama 3.1 70B via Groq API
Why this approach: 1. Combines strengths of retrieval (precision) and generation (fluency)
                   2. Improves factual accuracy and domain specificity.
Pipeline: 1. Retrieve highly relevant text
          2. Pass top-5 results as context
          3. LLM generates final summarized answer

# Web Portal 
Frontend: HTML templates, CSS styling, JavaScript for dynamic updates
Backend (Flask): Query processing, Database retrieval, LLM integration, Response rendering
User Feedback System: 1. Database stores:user query, generated answer, thumbs-up/down feedback
                      2. Feedback used to evaluate future improvements

# Accuracy Analysis 
Total Questions Tested: 137
Correct Answers: 114
Incorrect Answers: 23
Accuracy: 83%

# Notes:
1. Greeting queries excluded.
2. System struggles with major typos, formulas, and code snippet generation.

# Tech Stack:
1. Backend & ML: Python, SBERT (Sentence-BERT), pgvector, FlagReranker, Llama 3.1 70B (Groq API), pandas, regex, PyMuPDF, pdfplumber
2. Frontend: HTML, CSS, JavaScript
3. Database: PostgreSQL
4. Frameworks: Flask

