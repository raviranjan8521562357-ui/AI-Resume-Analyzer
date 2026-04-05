# AI Resume Analyzer - Backend

This is the FastAPI-powered backend for the AI Resume Analyzer. It handles document parsing, vector embeddings (FAISS), and communication with the Hugging Face Router API to evaluate resumes and respond to chat queries.

## Environment Variables

To run this project, you need to configure the following environment variables. In a local development setup, you can create a `.env` file in the `backend` directory. When deploying to a service like Render or Railway, add these exactly as shown in their environment variables dashboard.

### `HUGGINGFACE_API_KEY` (Required)
*   **Description:** The authentication token used to access the open-source LLMs via the HuggingFace router.
*   **How to get it:** Create a free account at Hugging Face and generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
*   **Example format:** `hf_xxxxxxxxxxxxxxxxxxx`

## Running Locally

1. Ensure you have Python 3.11+ installed.
2. Navigate into the backend directory: `cd backend`
3. Install dependencies: `pip install -r requirements.txt`
4. Add your `.env` file with the `HUGGINGFACE_API_KEY`.
5. Start the development server:
   ```bash
   python -m uvicorn main:app --reload --port 8000
   ```
6. The backend API will be available at `http://localhost:8000`. You can view the automatic API documentation at `http://localhost:8000/docs`.
