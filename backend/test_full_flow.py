"""Test the full analysis flow as done in main.py."""
import sys
sys.path.insert(0, '.')

# Simulate what main.py does
from services.embedder import compute_similarity, build_index, search_index
from services.analyzer import analyze_resume, chat_with_resume
from utils.text_utils import clean_text, chunk_text

resume_text = """John Doe
Software Engineer
Email: john@example.com
Phone: +1-555-1234

Skills: Python, JavaScript, React, Node.js, Docker, AWS

Work Experience:
Software Engineer at TechCo (Jan 2024 - Present)
  Built REST APIs with FastAPI
  Deployed microservices on AWS

Projects:
AI Resume Analyzer: Built with Python, FastAPI, React
Chat Bot: Built with LLM, RAG, FAISS
"""

jd = "Looking for a skilled Python developer with React and AWS experience."

print("Step 1: clean_text and chunk_text...")
cleaned = clean_text(resume_text)
chunks = chunk_text(cleaned)
print(f"  Cleaned: {len(cleaned)} chars, {len(chunks)} chunks")

print("\nStep 2: compute_similarity...")
try:
    similarity = compute_similarity(cleaned, jd)
    print(f"  Similarity: {similarity:.4f}")
except Exception as e:
    import traceback
    print(f"  ERROR in compute_similarity: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: analyze_resume...")
try:
    combined_result = analyze_resume(
        resume_text=cleaned,
        job_description=jd,
        similarity_score=similarity
    )
    print(f"  ATS Score: {combined_result.get('ats_score')}")
    print(f"  Skills: {combined_result.get('skills', [])}")
    print(f"  Name: {combined_result.get('candidate_name')}")
    print(f"  Projects: {len(combined_result.get('projects', []))}")
    print(f"  Internships: {len(combined_result.get('internships', []))}")
except Exception as e:
    import traceback
    print(f"  ERROR in analyze_resume: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: build_index and search_index...")
try:
    index, indexed_chunks = build_index(chunks)
    relevant = search_index(index, indexed_chunks, "What skills does this candidate have?", top_k=3)
    print(f"  Found {len(relevant)} relevant chunks")
except Exception as e:
    import traceback
    print(f"  ERROR in build_index/search_index: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\nAll steps completed successfully!")
