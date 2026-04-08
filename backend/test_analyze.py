"""Quick test to reproduce the 'NoneType not callable' error."""
import sys
sys.path.insert(0, '.')

from services.analyzer import analyze_resume

resume_text = """John Doe
Software Engineer
Email: john@example.com
Phone: +1-555-1234

Skills: Python, JavaScript, React, Node.js, Docker, AWS

Work Experience:
- Software Engineer at TechCo (Jan 2024 - Present)
  Built REST APIs with FastAPI
  Deployed microservices on AWS

Projects:
- AI Resume Analyzer: Built with Python, FastAPI, React
- Chat Bot: Built with LLM, RAG, FAISS
"""

jd = "Looking for a skilled Python developer with React and AWS experience."

try:
    result = analyze_resume(resume_text, jd, 0.7)
    print(f"SUCCESS! Result keys: {list(result.keys())}")
    print(f"ATS Score: {result.get('ats_score')}")
    print(f"Skills: {result.get('skills')}")
    print(f"Name: {result.get('candidate_name')}")
    print(f"Projects count: {len(result.get('projects', []))}")
    print(f"Internships count: {len(result.get('internships', []))}")
except Exception as e:
    import traceback
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
