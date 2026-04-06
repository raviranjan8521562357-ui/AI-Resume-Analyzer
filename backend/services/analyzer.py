"""
LLM analysis using Hugging Face Inference API (router.huggingface.co).
Handles ATS scoring, candidate extraction, and RAG-based chat.
"""
import json
import os
import re
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# HuggingFace Router API (OpenAI-compatible)
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

# Working models on HF router (verified)
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",     # Fast primary — good JSON, low latency
    "Qwen/Qwen2.5-72B-Instruct",           # Accurate fallback — excellent JSON compliance
    "Qwen/Qwen2.5-Coder-32B-Instruct",      # Code-aware fallback
]


def _get_headers() -> dict:
    """Get authorization headers for HF API."""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY not set. Add it to your .env file.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _call_llm(messages: list[dict], max_tokens: int = 2048) -> str:
    """
    Call HuggingFace router API with model fallback + retry.
    Uses OpenAI-compatible chat completions format.
    """
    headers = _get_headers()
    last_error = None

    for model in MODELS:
        for attempt in range(2):
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.9,
                }
                response = requests.post(
                    HF_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=25,
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"].strip()

                error_text = response.text
                status = response.status_code

                # Model not supported — skip to next model
                if status == 400:
                    print(f"[{model}] Not supported on router, trying next...")
                    break

                # Rate limit or overloaded — retry with backoff
                if status in (429, 503, 500):
                    wait = 3 * (attempt + 1)
                    print(f"[{model}] HTTP {status} (attempt {attempt+1}/2). Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                # Other error
                print(f"[{model}] HTTP {status}: {error_text[:100]}")
                break

            except requests.exceptions.Timeout:
                wait = 3 * (attempt + 1)
                print(f"[{model}] Timeout (attempt {attempt+1}/2). Waiting {wait}s...")
                time.sleep(wait)
                continue
            except Exception as e:
                last_error = e
                print(f"[{model}] Error: {str(e)[:100]}")
                break

    raise ValueError(f"All HuggingFace models failed. Last error: {last_error}")


def _parse_json_response(response_text: str) -> dict:
    """
    Robustly parse JSON from LLM response.
    Handles markdown code blocks, extra text before/after JSON, etc.
    """
    text = response_text.strip()

    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
        text = text.strip()
    elif "```" in text:
        text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text (between first { and last })
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not extract JSON from response", text, 0)


def analyze_resume(resume_text: str, job_description: str, similarity_score: float) -> dict:
    """
    Analyze a resume against a job description AND extract candidate info in 1 call.
    Returns a combined dict with ATS score, match %, decision, explanation, skills, and candidate profile.
    """
    # Use generous limits so projects/internships at the end aren't cut off
    resume_truncated = resume_text[:5000]
    jd_truncated = job_description[:1500]

    messages = [
        {
            "role": "system",
            "content": "You are an expert ATS (Applicant Tracking System) analyzer and resume parser. You always respond with ONLY valid JSON, no explanations before or after the JSON object."
        },
        {
            "role": "user",
            "content": f"""Analyze the following resume against the job description. The embedding similarity score between them is {similarity_score:.2f} (0-1 scale).

RESUME:
{resume_truncated}

JOB DESCRIPTION:
{jd_truncated}

IMPORTANT EXTRACTION RULES:
- Extract ALL projects found in the resume. Do NOT limit to one.
- Extract ALL work experiences — full-time jobs, internships, part-time, freelance, contract. Check sections: Work Experience, Experience, Professional Experience, Employment History, Internship.
- If the resume has 2 work experiences (e.g. 1 full-time + 1 internship), the work_experience array MUST have 2 objects. NEVER merge or skip.
- Count the roles in the resume's Work Experience section. Your work_experience array MUST have the same count.

Respond with ONLY valid JSON in this exact format (no other text):
{{
    "ats_score": <number 0-100>,
    "match_percentage": <number 0-100>,
    "decision": "<Accept or Reject>",
    "explanation": "<2-3 sentence explanation>",
    "matched_skills": ["skill1", "skill2", "skill3"],
    "missing_skills": ["skill1", "skill2"],
    "candidate_name": "<full name or Unknown>",
    "email": "<email or null>",
    "phone": "<phone or null>",
    "skills": ["skill1", "skill2", "skill3"],
    "projects": [
        {{
            "name": "<project name>",
            "description": "<1-2 line description>",
            "tech": ["tech1", "tech2"]
        }}
    ],
    "work_experience": [
        {{
            "company": "<company 1>",
            "role": "<role title>",
            "experience_type": "Full-time",
            "duration": "Oct 2025 - Present",
            "work_done": ["<bullet 1>", "<bullet 2>", "<bullet 3>"],
            "technologies_used": ["tech1", "tech2"]
        }},
        {{
            "company": "<company 2>",
            "role": "<role title>",
            "experience_type": "Internship",
            "duration": "Aug 2025 - Sep 2025",
            "work_done": ["<bullet 1>", "<bullet 2>"],
            "technologies_used": ["tech1", "tech2"]
        }}
    ],
    "experience_level": "<Entry Level / Mid Level / Senior Level>",
    "education": "<highest education>",
    "total_experience_years": <number or 0>
}}

CRITICAL RULES:
1. work_experience MUST be a JSON array with one object per role. The example shows 2 — include ALL roles found, even if more or fewer than 2.
2. experience_type: "Full-time" for engineer/developer/analyst roles, "Internship" for intern roles.
3. Include ALL bullet points from each role in work_done (do not summarize).
4. projects MUST also include every project found.

Scoring guidelines:
- ats_score: Overall ATS compatibility (keywords, formatting, relevance). Factor in the similarity score.
- match_percentage: How well the candidate matches the job requirements
- decision: Accept if ats_score >= 60, else Reject"""
        }
    ]

    response_text = _call_llm(messages, max_tokens=2500)

    # Debug: log raw response to help diagnose extraction issues
    print(f"[DEBUG] Raw LLM response for resume (first 500 chars): {response_text[:500]}")

    try:
        result = _parse_json_response(response_text)
        # Debug: log how many projects/work experiences were extracted
        proj_count = len(result.get('projects', []))
        we_count = len(result.get('work_experience', result.get('internships', [])))
        print(f"[DEBUG] Extracted {proj_count} projects, {we_count} work experiences")
    except (json.JSONDecodeError, Exception) as e:
        print(f"[HF] JSON parse failed: {e}. Using fallback scoring.")
        result = _build_fallback_analysis(resume_text, job_description, similarity_score)

    # Ensure all required keys exist with defaults
    return _ensure_complete_result(result, similarity_score)


def _build_fallback_analysis(resume_text: str, job_description: str, similarity_score: float) -> dict:
    """
    Build a reasonable analysis using similarity score + keyword matching
    when LLM fails to return valid JSON.
    """
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    # Extract skills by simple keyword matching
    common_skills = [
        "python", "java", "javascript", "react", "node.js", "sql", "aws", "docker",
        "kubernetes", "git", "html", "css", "typescript", "mongodb", "postgresql",
        "machine learning", "deep learning", "tensorflow", "pytorch", "flask",
        "django", "fastapi", "spring", "angular", "vue", "c++", "c#", "rust", "go",
        "linux", "azure", "gcp", "redis", "kafka", "graphql", "rest api",
    ]

    jd_skills = [s for s in common_skills if s in jd_lower]
    matched = [s for s in jd_skills if s in resume_lower]
    missing = [s for s in jd_skills if s not in resume_lower]

    # Extract name (first line heuristic)
    lines = resume_text.strip().split("\n")
    candidate_name = lines[0].strip() if lines else "Unknown"
    if len(candidate_name) > 40 or len(candidate_name) < 2:
        candidate_name = "Unknown"

    # Extract email
    email_match = re.search(r'[\w.-]+@[\w.-]+\.\w+', resume_text)
    email = email_match.group() if email_match else None

    # Extract phone
    phone_match = re.search(r'[\+]?[\d\s\-\(\)]{10,15}', resume_text)
    phone = phone_match.group().strip() if phone_match else None

    score = int(similarity_score * 100)

    return {
        "ats_score": score,
        "match_percentage": score,
        "decision": "Accept" if score >= 60 else "Reject",
        "explanation": f"Fallback analysis: {len(matched)}/{len(jd_skills)} required skills found. Similarity score: {similarity_score:.2f}.",
        "matched_skills": matched,
        "missing_skills": missing,
        "candidate_name": candidate_name,
        "email": email,
        "phone": phone,
        "skills": matched,
        "projects": [],
        "internships": [],
        "experience_level": "Entry Level",
        "education": None,
        "total_experience_years": 0,
    }


def _ensure_complete_result(result: dict, similarity_score: float) -> dict:
    """Ensure all required keys exist and lists are properly formatted."""
    defaults = {
        "ats_score": int(similarity_score * 100),
        "match_percentage": int(similarity_score * 100),
        "decision": "Reject",
        "explanation": "",
        "matched_skills": [],
        "missing_skills": [],
        "candidate_name": "Unknown",
        "email": None,
        "phone": None,
        "skills": [],
        "projects": [],
        "internships": [],
        "experience_level": "Entry Level",
        "education": None,
        "total_experience_years": 0,
    }
    for key, default_val in defaults.items():
        if key not in result or result[key] is None:
            result[key] = default_val

    # Ensure projects is always a list of dicts (LLM may return a single dict)
    if isinstance(result["projects"], dict):
        result["projects"] = [result["projects"]]
    elif not isinstance(result["projects"], list):
        result["projects"] = []

    # Map work_experience -> internships for frontend compatibility
    if "work_experience" in result:
        we = result.pop("work_experience")
        if isinstance(we, list):
            # Merge with any existing internships (avoid duplicates)
            existing = result.get("internships", [])
            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                existing = []
            result["internships"] = existing + we
        elif isinstance(we, dict):
            existing = result.get("internships", [])
            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                existing = []
            result["internships"] = existing + [we]

    # Ensure internships is always a list of dicts
    if isinstance(result["internships"], dict):
        result["internships"] = [result["internships"]]
    elif not isinstance(result["internships"], list):
        result["internships"] = []

    # Normalize project entries — ensure each has required keys
    for p in result["projects"]:
        if isinstance(p, dict):
            p.setdefault("name", "Untitled Project")
            p.setdefault("description", "")
            p.setdefault("tech", [])
            # Handle alternate key names LLMs sometimes use
            if "title" in p and not p.get("name"):
                p["name"] = p.pop("title")
            if "tech_stack" in p and not p.get("tech"):
                p["tech"] = p.pop("tech_stack")
                if isinstance(p["tech"], str):
                    p["tech"] = [t.strip() for t in p["tech"].split(",")]

    # Normalize internship/experience entries
    for intern in result["internships"]:
        if isinstance(intern, dict):
            intern.setdefault("company", "Unknown")
            intern.setdefault("role", "N/A")
            intern.setdefault("duration", "")
            intern.setdefault("work_done", [])
            intern.setdefault("technologies_used", [])
            # Handle alternate key names LLMs sometimes use
            if "organization" in intern and intern.get("company") == "Unknown":
                intern["company"] = intern.pop("organization")
            if "position" in intern and intern.get("role") == "N/A":
                intern["role"] = intern.pop("position")
            if "title" in intern and intern.get("role") == "N/A":
                intern["role"] = intern.pop("title")
            # Auto-detect experience_type if not provided
            if "experience_type" not in intern or not intern["experience_type"]:
                role_lower = (intern.get("role") or "").lower()
                if "intern" in role_lower:
                    intern["experience_type"] = "Internship"
                elif any(kw in role_lower for kw in ["freelance", "freelancer"]):
                    intern["experience_type"] = "Freelance"
                elif any(kw in role_lower for kw in ["contract", "contractor"]):
                    intern["experience_type"] = "Contract"
                else:
                    intern["experience_type"] = "Full-time"
            # Alternate keys for work_done
            for alt in ["responsibilities", "description", "tasks", "bullets", "achievements"]:
                if alt in intern and not intern.get("work_done"):
                    val = intern.pop(alt)
                    if isinstance(val, str):
                        intern["work_done"] = [val]
                    elif isinstance(val, list):
                        intern["work_done"] = val
            # Alternate keys for technologies_used
            for alt in ["tech", "tech_stack", "tools", "technologies", "stack"]:
                if alt in intern and not intern.get("technologies_used"):
                    val = intern.pop(alt)
                    if isinstance(val, str):
                        intern["technologies_used"] = [t.strip() for t in val.split(",")]
                    elif isinstance(val, list):
                        intern["technologies_used"] = val
            # Ensure work_done items are strings
            intern["work_done"] = [str(w) for w in intern["work_done"] if w]

    return result


def chat_with_resume(question: str, context_chunks: list[str]) -> str:
    """Answer a question about a resume using RAG context chunks."""
    context = "\n---\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant analyzing a resume. Answer based ONLY on the provided context. If the answer is not in the context, say 'I couldn't find that information in the resume.'"
        },
        {
            "role": "user",
            "content": f"""RESUME CONTEXT:
{context[:4000]}

QUESTION: {question}

Answer concisely and directly:"""
        }
    ]

    return _call_llm(messages, max_tokens=512)
