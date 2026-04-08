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


def _call_llm(messages: list[dict], max_tokens: int = 4000) -> str:
    """
    Call HuggingFace router API with model fallback + retry.
    Uses OpenAI-compatible chat completions format.
    """
    headers = _get_headers()
    last_error = None

    for model in MODELS:
        for attempt in range(3):
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "top_p": 0.9,
                }
                response = requests.post(
                    HF_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=90,
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    if not content:
                        print(f"[{model}] Empty response (attempt {attempt+1}/3)")
                        time.sleep(2)
                        continue
                    return content

                error_text = response.text
                status = response.status_code

                # Model not supported — skip to next model
                if status == 400:
                    print(f"[{model}] Not supported on router, trying next...")
                    break

                # Rate limit or overloaded — retry with backoff
                if status in (429, 503, 500, 502):
                    wait = 5 * (attempt + 1)
                    print(f"[{model}] HTTP {status} (attempt {attempt+1}/3). Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                # Other error
                print(f"[{model}] HTTP {status}: {error_text[:200]}")
                last_error = ValueError(f"HTTP {status}: {error_text[:200]}")
                break

            except requests.exceptions.Timeout:
                wait = 5 * (attempt + 1)
                print(f"[{model}] Timeout (attempt {attempt+1}/3). Waiting {wait}s...")
                last_error = TimeoutError(f"{model} timed out")
                time.sleep(wait)
                continue
            except Exception as e:
                last_error = e
                print(f"[{model}] Error: {str(e)[:200]}")
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
    resume_truncated = resume_text[:6000]
    jd_truncated = job_description[:1500]

    messages = [
        {
            "role": "system",
            "content": "You are an expert ATS (Applicant Tracking System) analyzer and resume parser. You always respond with ONLY valid JSON, no explanations before or after the JSON object. Never add any text outside the JSON."
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

Respond with ONLY this JSON (no other text before or after):
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
            "company": "<company name>",
            "role": "<role title>",
            "experience_type": "Full-time or Internship",
            "duration": "Month Year - Month Year",
            "work_done": ["<bullet 1>", "<bullet 2>"],
            "technologies_used": ["tech1", "tech2"]
        }}
    ],
    "experience_level": "<Entry Level / Mid Level / Senior Level>",
    "education": "<highest education>",
    "total_experience_years": <number or 0>
}}

CRITICAL:
1. Output ONLY JSON. No markdown, no explanation, no code fences.
2. work_experience MUST list every role found.
3. projects MUST list every project found.
4. experience_type: "Full-time" for jobs, "Internship" for internships.
5. ats_score >= 60 means Accept, else Reject."""
        }
    ]

    # Attempt 1: Full analysis
    result = None
    try:
        response_text = _call_llm(messages, max_tokens=4000)
        print(f"[DEBUG] Raw LLM response (first 500 chars): {response_text[:500]}")
        result = _parse_json_response(response_text)
        proj_count = len(result.get('projects', []))
        we_count = len(result.get('work_experience', result.get('internships', [])))
        print(f"[DEBUG] Extracted {proj_count} projects, {we_count} work experiences")
    except json.JSONDecodeError as e:
        print(f"[HF] JSON parse failed on attempt 1: {e}")
    except Exception as e:
        print(f"[HF] Analysis attempt 1 failed: {type(e).__name__}: {e}")

    # Attempt 2: Retry with simpler prompt if first attempt failed
    if result is None:
        try:
            print("[HF] Retrying with simplified prompt...")
            simple_messages = [
                {
                    "role": "system",
                    "content": "Output ONLY valid JSON. No other text."
                },
                {
                    "role": "user",
                    "content": f"""Parse this resume and output JSON with these keys:
ats_score (0-100), match_percentage (0-100), decision (Accept/Reject), explanation (string),
matched_skills (array), missing_skills (array), candidate_name (string), email (string/null),
phone (string/null), skills (array), projects (array of {{name, description, tech}}),
work_experience (array of {{company, role, experience_type, duration, work_done, technologies_used}}),
experience_level (string), education (string), total_experience_years (number).

Resume: {resume_truncated[:4000]}
Job: {jd_truncated[:800]}
Similarity: {similarity_score:.2f}"""
                }
            ]
            response_text = _call_llm(simple_messages, max_tokens=4000)
            print(f"[DEBUG] Retry response (first 500 chars): {response_text[:500]}")
            result = _parse_json_response(response_text)
            print(f"[DEBUG] Retry succeeded! Projects: {len(result.get('projects', []))}, Experience: {len(result.get('work_experience', []))}")
        except Exception as e2:
            print(f"[HF] Retry also failed: {type(e2).__name__}: {e2}")

    # Attempt 3: Use regex-enhanced fallback
    if result is None:
        print("[HF] All LLM attempts failed. Using enhanced fallback extraction.")
        result = _build_fallback_analysis(resume_text, job_description, similarity_score)

    # Ensure all required keys exist with defaults
    return _ensure_complete_result(result, similarity_score)


def _build_fallback_analysis(resume_text: str, job_description: str, similarity_score: float) -> dict:
    """
    Build a comprehensive analysis using regex-based extraction
    when LLM fails to return valid JSON.
    Extracts name, email, phone, skills, projects, experience, and education.
    """
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    # Extract skills by keyword matching
    common_skills = [
        "python", "java", "javascript", "react", "node.js", "sql", "aws", "docker",
        "kubernetes", "git", "html", "css", "typescript", "mongodb", "postgresql",
        "machine learning", "deep learning", "tensorflow", "pytorch", "flask",
        "django", "fastapi", "spring", "angular", "vue", "c++", "c#", "rust", "go",
        "linux", "azure", "gcp", "redis", "kafka", "graphql", "rest api",
        "express", "next.js", "tailwind", "sass", "webpack", "vite",
        "pandas", "numpy", "scikit-learn", "selenium", "jest",
        "mysql", "sqlite", "firebase", "supabase", "prisma",
        "langchain", "openai", "hugging face", "nlp", "opencv",
        "figma", "photoshop", "jira", "postman", "swagger",
    ]
    all_resume_skills = [s for s in common_skills if s in resume_lower]
    jd_skills = [s for s in common_skills if s in jd_lower]
    matched = [s for s in jd_skills if s in resume_lower]
    missing = [s for s in jd_skills if s not in resume_lower]

    # Extract name (first non-empty line that looks like a name)
    lines = [l.strip() for l in resume_text.strip().split("\n") if l.strip()]
    candidate_name = "Unknown"
    for line in lines[:5]:
        # Skip lines with emails, phones, URLs, or too long
        if '@' in line or 'http' in line or len(line) > 50 or len(line) < 2:
            continue
        # Skip lines that look like headers
        if any(kw in line.lower() for kw in ['resume', 'curriculum', 'objective', 'summary', 'profile']):
            continue
        # A name is typically 2-4 words, mostly alphabetic
        words = line.split()
        if 1 <= len(words) <= 5 and all(w.replace('.', '').replace('-', '').isalpha() for w in words):
            candidate_name = line
            break

    # Extract email
    email_match = re.search(r'[\w.+-]+@[\w.-]+\.\w{2,}', resume_text)
    email = email_match.group() if email_match else None

    # Extract phone
    phone_match = re.search(r'(?:\+\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}', resume_text)
    phone = phone_match.group().strip() if phone_match else None

    # Extract education
    education = None
    edu_patterns = [
        r'(?:B\.?(?:Tech|E|Sc|S|A)|Bachelor|M\.?(?:Tech|S|Sc|E|A|BA)|Master|Ph\.?D|MBA)[^\n]{5,80}',
    ]
    for pat in edu_patterns:
        edu_match = re.search(pat, resume_text, re.IGNORECASE)
        if edu_match:
            education = edu_match.group().strip()
            break

    # Extract projects using section detection
    projects = []
    project_section = re.search(
        r'(?:projects?|personal projects?|academic projects?)\s*[:\n](.+?)(?=\n(?:experience|work|education|skills|certification|achievement|award|reference|$))',
        resume_text, re.IGNORECASE | re.DOTALL
    )
    if project_section:
        proj_text = project_section.group(1)
        # Find project names (lines that look like titles)
        proj_lines = [l.strip() for l in proj_text.split('\n') if l.strip()]
        current_proj = None
        for line in proj_lines:
            # Project title heuristic: short line, not starting with bullet
            if not line.startswith(('•', '-', '–', '*', '◦')) and len(line) < 100 and len(line) > 3:
                if current_proj:
                    projects.append(current_proj)
                current_proj = {"name": line.split('|')[0].split('–')[0].strip(), "description": "", "tech": []}
                # Try to extract tech from the line
                tech_match = re.search(r'(?:tech|stack|tools?|using)\s*[:\-]\s*(.+)', line, re.IGNORECASE)
                if tech_match:
                    current_proj["tech"] = [t.strip() for t in tech_match.group(1).split(',')]
            elif current_proj and line.startswith(('•', '-', '–', '*', '◦')):
                desc_text = line.lstrip('•-–*◦ ').strip()
                if current_proj["description"]:
                    current_proj["description"] += "; " + desc_text
                else:
                    current_proj["description"] = desc_text
        if current_proj:
            projects.append(current_proj)

    # Extract work experience using section detection
    internships = []
    exp_section = re.search(
        r'(?:work experience|experience|professional experience|employment|internship)s?\s*[:\n](.+?)(?=\n(?:project|education|skills|certification|achievement|award|reference|$))',
        resume_text, re.IGNORECASE | re.DOTALL
    )
    if exp_section:
        exp_text = exp_section.group(1)
        exp_lines = [l.strip() for l in exp_text.split('\n') if l.strip()]
        current_exp = None
        for line in exp_lines:
            # Check if line has a date range (likely a role header)
            has_date = bool(re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{4})', line, re.IGNORECASE))
            is_bullet = line.startswith(('•', '-', '–', '*', '◦'))
            if not is_bullet and (has_date or ('|' in line and len(line) < 120)):
                if current_exp:
                    internships.append(current_exp)
                # Parse role/company from the line
                parts = re.split(r'[|–—]', line)
                role_company = parts[0].strip() if parts else line
                duration = parts[-1].strip() if len(parts) > 1 else ""
                # Try to split role and company
                rc_parts = re.split(r'\s+at\s+|\s*,\s*|\s+@\s+', role_company, maxsplit=1)
                role = rc_parts[0].strip()
                company = rc_parts[1].strip() if len(rc_parts) > 1 else "Unknown"
                exp_type = "Internship" if "intern" in role.lower() else "Full-time"
                current_exp = {
                    "company": company, "role": role, "experience_type": exp_type,
                    "duration": duration, "work_done": [], "technologies_used": []
                }
            elif current_exp and is_bullet:
                current_exp["work_done"].append(line.lstrip('•-–*◦ ').strip())
        if current_exp:
            internships.append(current_exp)

    score = max(25, int(similarity_score * 100))
    # Boost score if many skills match
    if len(matched) > 3:
        score = min(100, score + 10)

    explanation = f"Regex-based fallback analysis: {len(matched)}/{len(jd_skills)} required skills found."
    if projects:
        explanation += f" {len(projects)} project(s) detected."
    if internships:
        explanation += f" {len(internships)} work experience(s) detected."
    explanation += f" Similarity: {similarity_score:.2f}."

    return {
        "ats_score": score,
        "match_percentage": score,
        "decision": "Accept" if score >= 60 else "Reject",
        "explanation": explanation,
        "matched_skills": matched,
        "missing_skills": missing,
        "candidate_name": candidate_name,
        "email": email,
        "phone": phone,
        "skills": all_resume_skills if all_resume_skills else matched,
        "projects": projects,
        "internships": internships,
        "experience_level": "Mid Level" if len(internships) > 1 else "Entry Level",
        "education": education,
        "total_experience_years": len(internships),  # rough estimate
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
