"""
Resume ATS HR Dashboard — FastAPI Backend
Multi-resume analysis, candidate ranking, and RAG chat.
"""
import uuid
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.text_utils import clean_text, chunk_text
from services.extractor import extract_text
from services.embedder import compute_similarity, build_index, search_index
from services.analyzer import analyze_resume, chat_with_resume


def _classify_project_level(project: dict) -> str:
    """
    Classify a project as Beginner / Intermediate / Advanced / Production
    based on keywords in description and tech stack.
    """
    text = " ".join([
        (project.get("name") or ""),
        (project.get("description") or ""),
        " ".join(project.get("tech", [])),
    ]).lower()

    # Production-level keywords (deployed, real-time, scalable systems)
    production_kw = [
        "deployed", "production", "real-time", "scalable", "pipeline", "ci/cd",
        "monitoring", "kubernetes", "docker", "aws", "gcp", "azure", "cloud",
        "load balancing", "million", "1000+", "500+", "100+", "staging",
        "microservice", "enterprise", "sla", "uptime", "webhook", "cron",
    ]
    # Advanced keywords (ML, distributed, complex system design)
    advanced_kw = [
        "machine learning", "deep learning", "nlp", "transformer", "bert",
        "llm", "rag", "faiss", "vector", "embeddings", "neural", "fine-tun",
        "distributed", "full-stack", "full stack", "langchain", "openai",
        "classification", "recommendation", "chatbot", "voice",
    ]
    # Intermediate keywords (standard frameworks, databases)
    intermediate_kw = [
        "database", "rest", "api", "authentication", "testing", "react",
        "node", "django", "flask", "fastapi", "sql", "mongodb", "firebase",
        "express", "crud", "jwt", "oauth", "graphql",
    ]

    prod_hits = sum(1 for kw in production_kw if kw in text)
    adv_hits = sum(1 for kw in advanced_kw if kw in text)
    inter_hits = sum(1 for kw in intermediate_kw if kw in text)

    if prod_hits >= 2:
        return "Production"
    if adv_hits >= 2 or (adv_hits >= 1 and prod_hits >= 1):
        return "Advanced"
    if inter_hits >= 2 or adv_hits >= 1:
        return "Intermediate"
    return "Beginner"


# Points per project level
PROJECT_LEVEL_SCORES = {
    "Production": 40,
    "Advanced": 30,
    "Intermediate": 20,
    "Beginner": 10,
}


# Impact keywords that signal real-world production experience
_IMPACT_KEYWORDS = [
    "production", "deployed", "client", "real-world", "users", "uptime",
    "scalable", "revenue", "traffic", "sla", "monitoring", "million",
    "enterprise", "customer", "launch", "live", "release",
]

# Dimension weights for weighted composite (must sum to 1.0)
_WEIGHTS = {
    "technical": 0.30,
    "project_quality": 0.20,
    "experience": 0.40,
    "soft_skills": 0.10,
}


def compute_profile_fit(analysis: dict, candidate_info: dict, job_description: str) -> dict:
    """
    Compute a quality-based profile fit breakdown.
    4 dimensions: Technical Skills (30%), Project Quality (20%),
    Experience Quality (40%), Soft Skills (10%).
    Heavily penalizes candidates with no real-world experience.
    """
    matched = analysis.get("matched_skills", [])
    missing = analysis.get("missing_skills", [])
    all_skills = candidate_info.get("skills", [])
    total_required = len(matched) + len(missing)
    internships = candidate_info.get("internships", [])  # all work experiences
    projects = candidate_info.get("projects", [])
    exp_years = candidate_info.get("total_experience_years", 0)
    exp_level = candidate_info.get("experience_level", "Entry Level")
    education = (candidate_info.get("education") or "").lower()
    jd_lower = job_description.lower()

    # Collect all resume text for impact keyword scanning
    _all_text_parts = []
    for p in projects:
        _all_text_parts.append((p.get("description") or "").lower())
        _all_text_parts.append((p.get("name") or "").lower())
    for exp in internships:
        for wd in exp.get("work_done", []):
            _all_text_parts.append(wd.lower())
        _all_text_parts.append((exp.get("role") or "").lower())
    _all_text = " ".join(_all_text_parts)

    # ─── 1. Technical Skills Score (0-100) — Weight: 30% ───
    if total_required > 0:
        skill_match_ratio = len(matched) / total_required
    else:
        skill_match_ratio = 0.5

    skill_breadth_bonus = min(10, len(all_skills) * 1.0)
    tech_score = min(100, round(skill_match_ratio * 75 + skill_breadth_bonus + 5))

    tech_explain = f"{len(matched)}/{total_required} required skills matched"
    if len(all_skills) > 5:
        tech_explain += f", {len(all_skills)} total skills"

    # ─── 2. Project Quality Score (0-100) — Weight: 20% ───
    project_levels = []
    for p in projects:
        level = _classify_project_level(p)
        p["_level"] = level  # Attach for frontend display
        project_levels.append(level)

    if project_levels:
        level_scores = [PROJECT_LEVEL_SCORES.get(l, 10) for l in project_levels]
        # Weighted: best project counts most (40%), average of rest (60%)
        best = max(level_scores)
        avg_rest = sum(level_scores) / len(level_scores)
        raw_project = best * 0.4 + avg_rest * 0.6
        # Bonus for having multiple projects
        count_bonus = min(15, (len(projects) - 1) * 5)
        # Bonus for JD-relevant tech in projects
        relevance_bonus = 0
        for p in projects:
            for t in p.get("tech", []):
                if t.lower() in jd_lower:
                    relevance_bonus += 3
        relevance_bonus = min(15, relevance_bonus)
        project_score = min(100, round(raw_project + count_bonus + relevance_bonus))
    else:
        project_score = 0

    # Build explanation
    level_counts = {}
    for l in project_levels:
        level_counts[l] = level_counts.get(l, 0) + 1
    proj_parts = [f"{v} {k}" for k, v in level_counts.items()]
    proj_explain = f"{len(projects)} project{'s' if len(projects) != 1 else ''}"
    if proj_parts:
        proj_explain += f" ({', '.join(proj_parts)})"

    # ─── 3. Experience Quality Score (0-100) — Weight: 40% (DOMINANT) ───
    fulltime_roles = [e for e in internships if e.get("experience_type") == "Full-time"]
    intern_roles = [e for e in internships if e.get("experience_type") == "Internship"]
    other_roles = [e for e in internships if e.get("experience_type") not in ("Full-time", "Internship")]

    has_any_experience = len(internships) > 0

    # Full-time = 35 pts each (max 70), Internship = 15 pts each (max 30)
    ft_score = min(70, len(fulltime_roles) * 35)
    intern_score_val = min(30, len(intern_roles) * 15)
    other_score = min(10, len(other_roles) * 5)

    # Role relevance bonus (only if they have experience)
    role_relevance = 0
    for exp in internships:
        role = (exp.get("role") or "").lower()
        if any(kw in jd_lower for kw in role.split() if len(kw) > 3):
            role_relevance += 5
        if any(kw in role for kw in ["engineer", "developer", "analyst", "scientist", "ml", "ai", "data"]):
            role_relevance += 3
    role_relevance = min(20, role_relevance)

    # Impact-based bonus: keywords indicating real-world production work
    impact_hits = sum(1 for kw in _IMPACT_KEYWORDS if kw in _all_text)
    if impact_hits >= 5:
        impact_bonus = 10
    elif impact_hits >= 3:
        impact_bonus = 7
    elif impact_hits >= 1:
        impact_bonus = 5
    else:
        impact_bonus = 0

    if has_any_experience:
        exp_score = min(100, ft_score + intern_score_val + other_score + role_relevance + impact_bonus)
    else:
        # No professional experience at all → hard cap at 10
        exp_score = min(10, impact_bonus)

    # Build explanation
    exp_parts = []
    if fulltime_roles:
        exp_parts.append(f"{len(fulltime_roles)} full-time role{'s' if len(fulltime_roles) != 1 else ''}")
    if intern_roles:
        exp_parts.append(f"{len(intern_roles)} internship{'s' if len(intern_roles) != 1 else ''}")
    if other_roles:
        exp_parts.append(f"{len(other_roles)} other role{'s' if len(other_roles) != 1 else ''}")
    if exp_years > 0:
        exp_parts.append(f"{exp_years} yr{'s' if exp_years != 1 else ''} total")
    if impact_bonus > 0:
        exp_parts.append(f"impact bonus +{impact_bonus}")
    if role_relevance > 0:
        exp_parts.append("relevant roles")
    if not exp_parts:
        exp_parts.append("no professional experience")
    exp_explain = ", ".join(exp_parts)

    # ─── 4. Soft Skills Score (0-100) — Weight: 10% ───
    edu_score = 15  # default
    if any(kw in education for kw in ["master", "m.tech", "m.s.", "mba", "phd", "doctorate"]):
        edu_score = 30
    elif any(kw in education for kw in ["bachelor", "b.tech", "b.e.", "b.sc", "b.s."]):
        edu_score = 22

    level_bonus = 10
    if "mid" in exp_level.lower():
        level_bonus = 22
    elif "senior" in exp_level.lower():
        level_bonus = 35

    # Communication signal: multiple work_done bullets suggests good articulation
    articulation = 0
    for exp in internships:
        wd = exp.get("work_done", [])
        if len(wd) >= 3:
            articulation += 8
        elif len(wd) >= 1:
            articulation += 4
    articulation = min(20, articulation)

    # Diversity bonus: breadth of project types
    diversity = min(15, len(set(project_levels)) * 5) if project_levels else 0

    soft_score = min(100, edu_score + level_bonus + articulation + diversity)

    soft_parts = []
    if edu_score >= 22:
        soft_parts.append(education.split(",")[0].strip() if education else "degree")
    soft_parts.append(exp_level.lower())
    if articulation > 0:
        soft_parts.append("strong articulation")
    soft_explain = ", ".join(soft_parts)

    # ─── Weighted Composite Score (0-100) ───
    composite = round(
        tech_score * _WEIGHTS["technical"]
        + project_score * _WEIGHTS["project_quality"]
        + exp_score * _WEIGHTS["experience"]
        + soft_score * _WEIGHTS["soft_skills"]
    )

    return {
        "technical": {"score": tech_score, "explanation": tech_explain},
        "project_quality": {"score": project_score, "explanation": proj_explain},
        "experience": {"score": exp_score, "explanation": exp_explain},
        "soft_skills": {"score": soft_score, "explanation": soft_explain},
        "composite_score": composite,
        "weights": _WEIGHTS,
    }

# --- App Setup ---
app = FastAPI(
    title="Resume ATS HR Dashboard",
    description="AI-powered multi-resume screening & ranking system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory candidate storage ---
# { id: { id, filename, text, chunks, analysis, candidate_info, shortlisted } }
candidates_db: dict[str, dict] = {}


# --- Request Models ---
class BatchAnalyzeRequest(BaseModel):
    job_description: str


class ChatRequest(BaseModel):
    question: str
    candidate_id: str


class ShortlistRequest(BaseModel):
    candidate_id: str
    shortlisted: bool


# --- Routes ---

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Resume ATS HR Dashboard API v2.0"}


@app.post("/upload-multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    """
    Upload multiple resumes (PDF/DOCX).
    Extracts text and stores each candidate.
    """
    results = []
    
    for file in files:
        if not file.filename:
            continue

        allowed = ('.pdf', '.docx')
        if not file.filename.lower().endswith(allowed):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Unsupported file type"
            })
            continue

        try:
            file_bytes = await file.read()
            raw_text = extract_text(file.filename, file_bytes)
            cleaned = clean_text(raw_text)

            if not cleaned:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Could not extract text"
                })
                continue

            # Deduplication: use content hash to detect duplicate uploads
            content_hash = hashlib.md5(file_bytes).hexdigest()[:12]
            
            # Check if this exact file was already uploaded (by content hash OR filename)
            existing_id = None
            for eid, entry in candidates_db.items():
                if entry.get("content_hash") == content_hash or entry["filename"] == file.filename:
                    existing_id = eid
                    break
            
            cid = existing_id or str(uuid.uuid4())[:8]
            chunks = chunk_text(cleaned)
            
            candidates_db[cid] = {
                "id": cid,
                "filename": file.filename,
                "text": cleaned,
                "chunks": chunks,
                "content_hash": content_hash,
                "analysis": None,
                "candidate_info": None,
                "shortlisted": False,
            }

            results.append({
                "id": cid,
                "filename": file.filename,
                "success": True,
                "char_count": len(cleaned)
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {"uploaded": len([r for r in results if r.get("success")]), "results": results}


@app.post("/analyze-batch")
async def analyze_batch(request: BatchAnalyzeRequest):
    """
    Analyze ALL uploaded resumes against a job description.
    Returns ranked candidates with scores and extracted info.
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is empty")

    if not candidates_db:
        raise HTTPException(status_code=400, detail="No resumes uploaded yet")

    results = []

    for cid, candidate in candidates_db.items():
        try:
            # Step 1: Compute similarity
            similarity = compute_similarity(candidate["text"], request.job_description)

            # Step 2: Combined LLM analysis & extraction (1 call to save time!)
            combined_result = analyze_resume(
                resume_text=candidate["text"],
                job_description=request.job_description,
                similarity_score=similarity
            )
            
            # Map back to structured fields for the frontend
            analysis = {
                "ats_score": combined_result.get("ats_score", 0),
                "match_percentage": combined_result.get("match_percentage", 0),
                "decision": combined_result.get("decision", "Reject"),
                "explanation": combined_result.get("explanation", ""),
                "matched_skills": combined_result.get("matched_skills", []),
                "missing_skills": combined_result.get("missing_skills", []),
                "similarity_score": round(similarity, 4)
            }
            
            info = {
                "candidate_name": combined_result.get("candidate_name", "Unknown"),
                "email": combined_result.get("email"),
                "phone": combined_result.get("phone"),
                "skills": combined_result.get("skills", []),
                "projects": combined_result.get("projects", []),
                "internships": combined_result.get("internships", []),
                "experience_level": combined_result.get("experience_level", "Entry Level"),
                "education": combined_result.get("education"),
                "total_experience_years": combined_result.get("total_experience_years", 0)
            }

            # Step 3: Compute real profile fit scores
            profile_fit = compute_profile_fit(analysis, info, request.job_description)

            # Step 4: Blend LLM score with weighted composite for better ranking
            # 50% LLM ATS score + 50% experience-weighted composite
            llm_score = analysis["ats_score"]
            composite = profile_fit.get("composite_score", llm_score)
            blended = round(llm_score * 0.50 + composite * 0.50)
            analysis["ats_score"] = blended
            # Re-evaluate decision based on blended score
            analysis["decision"] = "Accept" if blended >= 60 else "Reject"

            # Store in DB
            candidate["analysis"] = analysis
            candidate["candidate_info"] = info
            candidate["profile_fit"] = profile_fit

            results.append({
                "id": cid,
                "filename": candidate["filename"],
                "shortlisted": candidate["shortlisted"],
                "analysis": analysis,
                "candidate_info": info,
                "profile_fit": profile_fit,
            })

        except Exception as e:
            results.append({
                "id": cid,
                "filename": candidate["filename"],
                "shortlisted": candidate["shortlisted"],
                "analysis": {
                    "ats_score": 0,
                    "match_percentage": 0,
                    "decision": "Error",
                    "explanation": str(e),
                    "matched_skills": [],
                    "missing_skills": [],
                    "similarity_score": 0
                },
                "candidate_info": None,
                "error": str(e)
            })

    # Sort by ATS score (highest first)
    results.sort(key=lambda x: x["analysis"].get("ats_score", 0), reverse=True)

    return {"candidates": results, "total": len(results)}


@app.get("/candidates")
def get_candidates():
    """Get all stored candidates with their analysis results."""
    results = []
    for cid, c in candidates_db.items():
        results.append({
            "id": cid,
            "filename": c["filename"],
            "shortlisted": c["shortlisted"],
            "analysis": c["analysis"],
            "candidate_info": c["candidate_info"],
            "profile_fit": c.get("profile_fit"),
        })
    # Sort by ATS score
    results.sort(key=lambda x: (x["analysis"] or {}).get("ats_score", 0), reverse=True)
    return {"candidates": results}


@app.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: str):
    """Get full details for a single candidate."""
    candidate = candidates_db.get(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    return {
        "id": candidate["id"],
        "filename": candidate["filename"],
        "shortlisted": candidate["shortlisted"],
        "analysis": candidate["analysis"],
        "candidate_info": candidate["candidate_info"],
        "profile_fit": candidate.get("profile_fit"),
        "text": candidate["text"],
    }


@app.post("/shortlist")
def toggle_shortlist(request: ShortlistRequest):
    """Toggle shortlist status for a candidate."""
    candidate = candidates_db.get(request.candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    candidate["shortlisted"] = request.shortlisted
    return {"id": request.candidate_id, "shortlisted": request.shortlisted}

# ─── Structured Query Handling for Experience Questions ───

import re as _re
from datetime import datetime

MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _parse_date(date_str: str) -> datetime | None:
    """Parse a date string like 'Oct 2025', 'August 2024', 'Present' into a datetime."""
    date_str = date_str.strip().lower()
    if date_str in ("present", "current", "now", "ongoing"):
        return datetime.now()
    # Try patterns: "Oct 2025", "October 2025", "2025"
    parts = date_str.replace(",", "").split()
    if len(parts) == 2:
        month_str, year_str = parts
        month = MONTH_MAP.get(month_str[:3])
        try:
            year = int(year_str)
        except ValueError:
            return None
        if month and year:
            return datetime(year, month, 1)
    elif len(parts) == 1:
        try:
            year = int(parts[0])
            return datetime(year, 1, 1)
        except ValueError:
            return None
    return None


def _calc_duration_months(duration_str: str) -> int:
    """Calculate duration in months from a string like 'Oct 2025 - Present'."""
    if not duration_str:
        return 0
    parts = _re.split(r'\s*[-–—to]+\s*', duration_str, maxsplit=1)
    if len(parts) != 2:
        return 0
    start = _parse_date(parts[0])
    end = _parse_date(parts[1])
    if not start or not end:
        return 0
    diff = (end.year - start.year) * 12 + (end.month - start.month)
    return max(1, diff)  # At least 1 month


def _classify_query(question: str) -> str:
    """
    Classify a chat question into a query type.
    Returns: 'count_experience', 'count_internship', 'count_fulltime',
             'total_duration', 'count_projects', or 'descriptive'
    """
    q = question.lower().strip()

    # Counting queries
    if any(kw in q for kw in ["how many internship", "number of internship", "count internship",
                               "how many intern ", "total internship"]):
        return "count_internship"
    if any(kw in q for kw in ["how many full-time", "how many fulltime", "number of full-time",
                               "full time role", "full-time role", "how many jobs"]):
        return "count_fulltime"
    if any(kw in q for kw in ["how many experience", "how many work experience", "number of experience",
                               "how many role", "how many position", "total roles", "number of roles"]):
        return "count_experience"
    if any(kw in q for kw in ["how many project", "number of project", "count project", "total project"]):
        return "count_projects"

    # Duration queries
    if any(kw in q for kw in ["total month", "total experience", "months of experience",
                               "years of experience", "how long", "experience duration",
                               "total duration", "work duration", "how much experience"]):
        return "total_duration"

    return "descriptive"


def _handle_structured_query(query_type: str, candidate_info: dict) -> str:
    """Handle counting/duration queries using structured data instead of LLM."""
    experiences = candidate_info.get("internships", [])
    projects = candidate_info.get("projects", [])

    fulltime = [e for e in experiences if e.get("experience_type") == "Full-time"]
    internships = [e for e in experiences if e.get("experience_type") == "Internship"]

    if query_type == "count_experience":
        parts = []
        if fulltime:
            parts.append(f"{len(fulltime)} full-time role{'s' if len(fulltime) != 1 else ''}")
        if internships:
            parts.append(f"{len(internships)} internship{'s' if len(internships) != 1 else ''}")
        other = len(experiences) - len(fulltime) - len(internships)
        if other > 0:
            parts.append(f"{other} other role{'s' if other != 1 else ''}")
        total = len(experiences)
        summary = f"This candidate has **{total} work experience{'s' if total != 1 else ''}**"
        if parts:
            summary += f" ({', '.join(parts)})"
        summary += ":\n"
        for i, exp in enumerate(experiences, 1):
            exp_type = exp.get("experience_type", "Unknown")
            summary += f"\n{i}. **{exp.get('role', 'N/A')}** at {exp.get('company', 'Unknown')} ({exp.get('duration', 'N/A')}) — _{exp_type}_"
        return summary

    if query_type == "count_internship":
        count = len(internships)
        if count == 0:
            return "This candidate has **no internships** listed on their resume."
        summary = f"This candidate has **{count} internship{'s' if count != 1 else ''}**:\n"
        for i, exp in enumerate(internships, 1):
            summary += f"\n{i}. **{exp.get('role', 'N/A')}** at {exp.get('company', 'Unknown')} ({exp.get('duration', 'N/A')})"
        return summary

    if query_type == "count_fulltime":
        count = len(fulltime)
        if count == 0:
            return "This candidate has **no full-time roles** listed on their resume."
        summary = f"This candidate has **{count} full-time role{'s' if count != 1 else ''}**:\n"
        for i, exp in enumerate(fulltime, 1):
            summary += f"\n{i}. **{exp.get('role', 'N/A')}** at {exp.get('company', 'Unknown')} ({exp.get('duration', 'N/A')})"
        return summary

    if query_type == "count_projects":
        count = len(projects)
        if count == 0:
            return "This candidate has **no projects** listed on their resume."
        summary = f"This candidate has **{count} project{'s' if count != 1 else ''}**:\n"
        for i, p in enumerate(projects, 1):
            summary += f"\n{i}. **{p.get('name', 'Untitled')}**"
            if p.get("description"):
                summary += f" — {p['description']}"
        return summary

    if query_type == "total_duration":
        total_months = 0
        breakdown = []
        for exp in experiences:
            duration_str = exp.get("duration", "")
            months = _calc_duration_months(duration_str)
            total_months += months
            exp_type = exp.get("experience_type", "Unknown")
            breakdown.append(f"- **{exp.get('role', 'N/A')}** at {exp.get('company', 'Unknown')}: {months} month{'s' if months != 1 else ''} ({duration_str}) — _{exp_type}_")

        years = total_months // 12
        remaining_months = total_months % 12

        if total_months == 0:
            return "Could not calculate experience duration. No parseable date ranges found in the work experience entries."

        duration_text = ""
        if years > 0:
            duration_text += f"{years} year{'s' if years != 1 else ''}"
        if remaining_months > 0:
            if duration_text:
                duration_text += f" and "
            duration_text += f"{remaining_months} month{'s' if remaining_months != 1 else ''}"

        summary = f"**Total experience: {total_months} months ({duration_text})**\n\nBreakdown:\n"
        summary += "\n".join(breakdown)
        return summary

    return None


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Smart chat with a candidate's resume.
    Routes counting/duration queries to backend logic for accuracy.
    Only uses LLM for descriptive questions.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")

    candidate = candidates_db.get(request.candidate_id)
    if not candidate:
        raise HTTPException(status_code=400, detail="Candidate not found")

    try:
        # Step 1: Classify the query
        query_type = _classify_query(request.question)

        # Step 2: If it's a structured query and we have candidate_info, use backend logic
        if query_type != "descriptive" and candidate.get("candidate_info"):
            answer = _handle_structured_query(query_type, candidate["candidate_info"])
            if answer:
                return {"question": request.question, "answer": answer}

        # Step 3: Fall back to LLM-based RAG for descriptive questions
        chunks = candidate["chunks"]
        index, indexed_chunks = build_index(chunks)
        relevant = search_index(index, indexed_chunks, request.question, top_k=3)
        answer = chat_with_resume(request.question, relevant)

        return {"question": request.question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.delete("/candidates")
def clear_candidates():
    """Clear all stored candidates."""
    candidates_db.clear()
    return {"message": "All candidates cleared"}

