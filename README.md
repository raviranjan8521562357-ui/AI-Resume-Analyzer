# ⚡ AI-Powered Resume ATS & Candidate Intelligence System

> Production-ready Resume ATS platform leveraging open-source LLMs via HuggingFace Router API to automate and optimize the end-to-end recruitment workflow.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-LLM_Router-FFD21E?logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-FF6F00)
![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?logo=vite&logoColor=white)

---

## 🎯 Overview

A production-ready Resume ATS platform leveraging open-source Large Language Models (Llama 3.1, Qwen 72B, Qwen Coder 32B) via the **HuggingFace Router API** to automate and optimize the end-to-end recruitment workflow. The system performs **intelligent resume parsing**, **semantic skill matching**, and **contextual candidate evaluation** against job descriptions using advanced NLP and embedding techniques.

Implements **multi-resume batch processing with auto-ranking**, enabling HR teams to efficiently shortlist top candidates from large applicant pools. Features a **unified LLM pipeline** that extracts candidate profiles and scores ATS compatibility in a **single API call** for optimized performance. Integrates a **vector database (FAISS)** for scalable similarity search and **Retrieval-Augmented Generation (RAG)**, allowing dynamic querying of candidate profiles.

Includes an **explainable 4-Dimension Profile Fit Breakdown** that evaluates candidates across four dimensions — **Technical Skills**, **Project Quality**, **Experience Quality**, and **Soft Skills** — with AI-generated scores, project-level classification (Beginner → Production), and plain-English explanations. Features a **Smart Chat system** that intelligently routes counting/duration queries to structured backend logic for accuracy, and uses LLM-based RAG only for descriptive questions.

Built with a modular **FastAPI** backend and an interactive **React (Vite)** frontend delivering a premium SaaS-style dashboard experience.

> This solution significantly reduces manual screening effort, improves hiring accuracy, and enhances decision-making in talent acquisition workflows.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 **Multi-Resume Upload** | Batch upload PDF/DOCX resumes with drag & drop and content-hash deduplication |
| 🤖 **AI-Powered Analysis** | Open-source LLMs (Llama 3.1 / Qwen 72B) evaluate resumes against job descriptions |
| 📊 **ATS Scoring (0-100)** | Automated scoring with Accept/Reject decisions |
| 🔍 **Semantic Matching** | Sentence-transformer embeddings + cosine similarity |
| 🧠 **Unified Extraction** | Single LLM call extracts skills, projects, work experience & scores in one pass |
| 📈 **4D Profile Fit** | Technical Skills, Project Quality, Experience Quality, and Soft Skills with explanations |
| 🏗️ **Project Classification** | Auto-classifies projects as Beginner / Intermediate / Advanced / Production |
| 💼 **Experience Typing** | Distinguishes Full-time, Internship, Freelance, and Contract roles |
| ⭐ **Shortlisting** | Star and filter top candidates |
| 💬 **Smart RAG Chat** | Structured-query routing for counting/duration questions + LLM-RAG for descriptive Q&A |
| 🏆 **Candidate Ranking** | Auto-ranked by ATS score with filter tabs |
| 🎨 **Modern Dashboard UI** | Clean light SaaS-style interface with interactive charts & donut visualizations |
| 🔄 **Model Fallback Chain** | Automatic failover across 3 LLMs with retry & exponential backoff |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────┐
│  React Frontend  │────▶│          FastAPI Backend              │
│  (Vite + JSX)   │◀────│                                      │
└─────────────────┘     │  ┌────────────┐  ┌───────────────┐   │
                        │  │  pypdf     │  │ sentence-     │   │
                        │  │  python-   │  │ transformers  │   │
                        │  │  docx      │  │ + FAISS       │   │
                        │  └─────┬──────┘  └───────┬───────┘   │
                        │        │                 │           │
                        │  ┌─────▼─────────────────▼───────┐   │
                        │  │   HuggingFace Router API      │   │
                        │  │  (Llama 3.1 / Qwen 72B /      │   │
                        │  │   Qwen Coder 32B fallback)    │   │
                        │  └───────────────────────────────┘   │
                        └──────────────────────────────────────┘
```

### Request Flow

```
Resume Upload → Text Extraction (pypdf/docx) → Text Cleaning & Chunking
     ↓
Job Description → Embedding (all-MiniLM-L6-v2) → Cosine Similarity
     ↓
Unified LLM Call → ATS Score + Candidate Extraction (1 API call)
     ↓
Profile Fit Computation (4 Dimensions):
  ├─ Technical Skills  → skill match ratio + breadth bonus
  ├─ Project Quality   → level classification + relevance bonus
  ├─ Experience Quality → role weighting + type classification
  └─ Soft Skills       → education + articulation + diversity
     ↓
Ranked Results → Dashboard Charts + Detail Panels + Smart Chat
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11, FastAPI 0.115, Uvicorn |
| **LLMs** | Llama-3.1-8B-Instruct (primary), Qwen2.5-72B-Instruct, Qwen2.5-Coder-32B-Instruct (via HuggingFace Router) |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Vector DB** | FAISS (faiss-cpu) |
| **Text Extraction** | pypdf 4.3, python-docx 1.1 |
| **Frontend** | React 18.3, Vite 5.4 |
| **Styling** | Vanilla CSS (light theme, glassmorphism, SVG donut charts) |

---

## 📁 Project Structure

```
resume_ats/
├── backend/
│   ├── main.py                 # FastAPI app — routes, CORS, 4D profile fit scoring,
│   │                           #   project classification, smart chat routing
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # HUGGINGFACE_API_KEY
│   ├── .env.example            # Example env file
│   ├── services/
│   │   ├── analyzer.py         # HuggingFace LLM — unified ATS scoring & extraction,
│   │   │                       #   fallback analysis, response normalization
│   │   ├── embedder.py         # Sentence-transformers + FAISS index (singleton model)
│   │   └── extractor.py        # PDF/DOCX text extraction
│   └── utils/
│       └── text_utils.py       # Text cleaning & chunking for RAG
├── frontend/
│   ├── index.html              # Entry point with Inter font
│   ├── package.json            # React + Vite deps
│   ├── vite.config.js          # Dev server config (proxy to backend :8000)
│   └── src/
│       ├── main.jsx            # React mount
│       ├── App.jsx             # Dashboard layout & state management
│       ├── App.css             # Premium light theme CSS
│       └── components/
│           ├── UploadSection.jsx    # Multi-file upload with drag & drop
│           ├── CandidateTable.jsx   # Ranked candidate table with filters
│           ├── CandidateDetail.jsx  # Slide-in detail panel + RAG chat
│           └── DashboardCharts.jsx  # Skill breakdown bars + 4D Profile Fit donut chart
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- HuggingFace API Key ([Get one free](https://huggingface.co/settings/tokens))

### Installation

```bash
# Clone the project
cd resume_ats

# Backend setup
cd backend
pip install -r requirements.txt

# Add your HuggingFace API key
# Create/edit .env file:
# HUGGINGFACE_API_KEY=hf_your_key_here

# Frontend setup
cd ../frontend
npm install
```

### Running

```bash
# Terminal 1 — Backend
cd backend
python -m uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Health check — returns API version |
| `POST /upload-multiple` | POST | Upload multiple resumes (PDF/DOCX) with content-hash deduplication |
| `POST /analyze-batch` | POST | Analyze all resumes vs job description (unified LLM pipeline + 4D scoring) |
| `GET /candidates` | GET | List all candidates (ranked by ATS score) |
| `GET /candidates/{id}` | GET | Full candidate details + profile fit + resume text |
| `POST /shortlist` | POST | Toggle shortlist status for a candidate |
| `POST /chat` | POST | Smart chat — structured query routing + RAG-based Q&A |
| `DELETE /candidates` | DELETE | Clear all candidates |

---

## 📸 Screenshots

### Dashboard — Upload & Analyze
- Multi-file upload with drag & drop
- Job description input
- One-click batch analysis

### Candidate Ranking Table
- Ranked by ATS score
- Match percentage with progress bars
- Accept/Reject badges
- Experience level
- Shortlist toggle (★)

### Dashboard Charts (Right Panel)
- **Skill Match Breakdown** — Bar chart showing matched (✓) and missing (✕) skills with proficiency percentages
- **4D Profile Fit Donut** — SVG donut chart visualizing Technical Skills (blue), Project Quality (purple), Experience (amber), and Soft Skills (teal) with real computed scores and explanations

### Candidate Detail Panel
- Full ATS analysis with explanation
- Skills extraction (matched ✓ / missing ✕)
- Projects with tech stack and descriptions
- Work experience history with bullet points, technologies, and experience type badges
- RAG-powered smart chat interface

---

## 🧪 How It Works

1. **Upload** — PDF/DOCX resumes are parsed using `pypdf` and `python-docx`, with MD5 content-hash deduplication
2. **Clean & Chunk** — Raw text is cleaned and split into overlapping chunks (500 chars, 50 overlap) for RAG indexing
3. **Embed** — Resume and job description are converted to vectors using `all-MiniLM-L6-v2` (singleton model, loaded once)
4. **Similarity** — Cosine similarity is computed between embeddings
5. **Unified LLM Analysis** — A single HuggingFace LLM call performs both:
   - **ATS Scoring** — Score (0-100), match %, Accept/Reject, matched & missing skills
   - **Candidate Extraction** — Name, email, phone, skills, projects (with tech), work experience (with bullets, technologies, and experience type), education, experience level
6. **Response Normalization** — Robust JSON parsing with markdown code-block stripping, alternate key mapping (`work_experience` → `internships`, `title` → `name`, etc.), and auto-detection of experience type from role titles
7. **4-Dimension Profile Fit Scoring** — Backend computes quality-based scores:
   - **Technical Skills** (0-100) — Skill match ratio × 75 + breadth bonus + base
   - **Project Quality** (0-100) — Auto-classification (Beginner/Intermediate/Advanced/Production) + count bonus + JD-relevance bonus
   - **Experience Quality** (0-100) — Full-time (35pts each) + Internship (18pts each) + role relevance bonus
   - **Soft Skills** (0-100) — Education level + experience tier + articulation signal + project diversity
8. **Ranking** — Candidates are sorted by ATS score (highest first)
9. **Smart Chat** — Query classification routes counting/duration questions to backend structured logic for accuracy; descriptive questions go through FAISS-powered RAG + LLM pipeline

### Project Level Classification

Projects are automatically classified based on keywords in their description and tech stack:

| Level | Trigger Keywords (examples) | Points |
|---|---|---|
| **Production** | deployed, kubernetes, docker, aws, million, monitoring, ci/cd | 40 |
| **Advanced** | machine learning, LLM, RAG, FAISS, transformers, langchain | 30 |
| **Intermediate** | REST API, React, Django, authentication, database | 20 |
| **Beginner** | Default / simple projects | 10 |

### Smart Chat Query Routing

The chat system classifies questions before processing:

| Query Type | Example | Handling |
|---|---|---|
| `count_experience` | "How many roles does this candidate have?" | Structured backend logic |
| `count_internship` | "How many internships?" | Structured backend logic |
| `count_fulltime` | "How many full-time jobs?" | Structured backend logic |
| `count_projects` | "How many projects?" | Structured backend logic |
| `total_duration` | "Total months of experience?" | Duration calculation with date parsing |
| `descriptive` | "Explain their ML expertise" | FAISS RAG → LLM |

### LLM Fallback Strategy

The system uses a model fallback chain for resilience:

```
Llama-3.1-8B-Instruct (primary — fast, good JSON compliance)
  └─▶ Qwen2.5-72B-Instruct (accurate fallback — excellent JSON)
       └─▶ Qwen2.5-Coder-32B-Instruct (code-aware fallback)
```

Each model gets 3 retry attempts with exponential backoff (5s, 10s, 15s) before falling through to the next. The system also includes a **keyword-based fallback analyzer** when all LLM calls fail, using similarity score + simple skill matching.

---

## 🔮 Future Scope

- [ ] Database persistence (PostgreSQL/MongoDB)
- [ ] Resume improvement suggestions
- [ ] Email notifications for shortlisted candidates
- [ ] PDF report generation
- [ ] Multi-JD comparison
- [ ] Deployment (Docker + Cloud)
- [ ] Side-by-side candidate comparison view
- [ ] Export ranked results to CSV/Excel
- [ ] Authentication & role-based access

---

## 📄 License

This project is for educational and portfolio purposes.

---

**Built with ❤️ using HuggingFace LLMs, FastAPI, React, FAISS & Sentence-Transformers**
