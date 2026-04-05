# AI Resume Analyzer - Frontend

This is the interactive frontend dashboard for the AI Resume Analyzer built using React and Vite. It provides a multi-resume drag-and-drop interface, candidate ranking tables, 4D-profile visualizations, and a RAG-powered chat panel.

## Environment Variables

To run or deploy this project, you need to configure the following environment variables. 

For local development, create a `.env` file in the `frontend` root directory. When deploying to a service like Vercel or Netlify, add this exact variable name in their Environment Variables configuration panel.

### `VITE_API_BASE_URL`
*   **Description:** The base URL of your FastAPI backend service. The frontend uses this to know where to send uploaded resumes and API requests.
*   **Local Development Example:** `http://localhost:8000`
*   **Production Deployment Example:** `https://your-backend-app.onrender.com` (Note: Do not include a trailing slash `/` at the end).

## Running Locally

1. Ensure you have Node.js (version 18+) installed.
2. Navigate into the frontend directory: `cd frontend`
3. Install dependencies: `npm install`
4. Make sure your local `.env` has `VITE_API_BASE_URL=http://localhost:8000`.
5. Start the Vite development server:
   ```bash
   npm run dev
   ```
6. The application will be available in your browser at `http://localhost:5173`.
