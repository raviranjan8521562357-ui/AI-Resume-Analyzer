"""Test multiple HF embedding approaches in detail."""
import os
import requests
import numpy as np
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("HUGGINGFACE_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

texts = ["test resume with python skills", "looking for python developer"]

# Approach 1: Router v1/embeddings (OpenAI compat)
print("=== Test 1: v1/embeddings ===")
url1 = "https://router.huggingface.co/v1/embeddings"
payload1 = {"model": "sentence-transformers/all-MiniLM-L6-v2", "input": texts}
try:
    r = requests.post(url1, headers=headers, json=payload1, timeout=30)
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Approach 2: Novita provider
print("\n=== Test 2: novita provider ===")
url2 = "https://router.huggingface.co/novita/v3/openai/embeddings"
payload2 = {"model": "sentence-transformers/all-MiniLM-L6-v2", "input": texts}
try:
    r = requests.post(url2, headers=headers, json=payload2, timeout=30)
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# Approach 3: Use the LLM to compute similarity (most reliable)
print("\n=== Test 3: LLM similarity ===")
url3 = "https://router.huggingface.co/v1/chat/completions"
payload3 = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": f"Rate similarity 0.0-1.0. Only reply with a number.\nA: {texts[0]}\nB: {texts[1]}"}],
    "max_tokens": 10,
    "temperature": 0.1,
}
try:
    r = requests.post(url3, headers=headers, json=payload3, timeout=30)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        answer = r.json()["choices"][0]["message"]["content"].strip()
        print(f"Similarity: {answer}")
except Exception as e:
    print(f"Error: {e}")

# Approach 4: TEI via hf-inference
print("\n=== Test 4: hf-inference feature-extraction ===")
url4 = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
payload4 = {"inputs": texts[0]}
try:
    r = requests.post(url4, headers=headers, json=payload4, timeout=30)
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text[:300]}")
except Exception as e:
    print(f"Error: {e}")
