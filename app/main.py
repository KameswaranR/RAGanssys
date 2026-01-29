from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
import faiss
import numpy as np
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .utils import extract_text, chunk_text, get_embedding, generate_answer

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global State (In-memory for this assessment)
index = faiss.IndexFlatL2(384)
doc_map = {} 

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DocuMind AI</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --bg: #f8fafc;
                --card: #ffffff;
                --text: #1e293b;
            }
            body { 
                font-family: 'Inter', sans-serif; 
                background-color: var(--bg); 
                color: var(--text);
                margin: 0; padding: 20px;
                display: flex; flex-direction: column; align-items: center;
            }
            .container { max-width: 800px; width: 100%; }
            .header { text-align: center; margin-bottom: 30px; }
            
            /* Section Styling */
            .section { 
                background: var(--card); 
                padding: 25px; 
                border-radius: 16px; 
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                border: 1px solid #e2e8f0;
            }

            /* Upload Button & Input */
            input[type="file"] { margin-bottom: 10px; }
            button {
                background: var(--primary);
                color: white; border: none;
                padding: 10px 20px; border-radius: 8px;
                cursor: pointer; font-weight: 600;
                transition: opacity 0.2s;
            }
            button:hover { opacity: 0.9; }
            button:disabled { background: #94a3b8; }

            /* Chat Area */
            #chat-history { 
                background: #f1f5f9; 
                padding: 20px; 
                border-radius: 12px; 
                height: 400px; 
                overflow-y: auto; 
                display: flex; flex-direction: column; gap: 12px;
                border: 1px inset #e2e8f0;
            }
            .msg { padding: 12px 16px; border-radius: 12px; max-width: 80%; line-height: 1.5; }
            .user-msg { background: var(--primary); color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
            .ai-msg { background: white; color: var(--text); align-self: flex-start; border-bottom-left-radius: 2px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }

            .input-group { display: flex; gap: 10px; margin-top: 15px; }
            input[type="text"] {
                flex-grow: 1; padding: 12px;
                border: 1px solid #cbd5e1; border-radius: 8px; outline: none;
            }
            input[type="text"]:focus { border-color: var(--primary); ring: 2px var(--primary); }

            /* Loading Spinner */
            .spinner {
                display: inline-block; width: 12px; height: 12px;
                border: 2px solid rgba(255,255,255,.3); border-radius: 50%;
                border-top-color: #fff; animation: spin 1s ease-in-out infinite;
                margin-right: 8px; display: none;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📄 DocuMind AI</h1>
                <p>Upload a PDF and ask questions in plain English.</p>
            </div>
            
            <div class="section">
                <h3>1. Knowledge Base</h3>
                <input type="file" id="fileInput">
                <button id="uploadBtn" onclick="uploadFile()">
                    <span id="uploadSpinner" class="spinner"></span>Upload Document
                </button>
                <p id="uploadStatus" style="font-size: 0.9rem; margin-top: 8px; color: #64748b;"></p>
            </div>

            <div class="section">
                <h3>2. Chat</h3>
                <div id="chat-history">
                    <div class="msg ai-msg">Hello! Upload a document above to get started.</div>
                </div>
                <div class="input-group">
                    <input type="text" id="questionInput" placeholder="How does X work?">
                    <button onclick="askQuestion()">Send</button>
                </div>
            </div>
        </div>

        <script>
            async function uploadFile() {
                const fileBtn = document.getElementById('uploadBtn');
                const spinner = document.getElementById('uploadSpinner');
                const file = document.getElementById('fileInput').files[0];
                if(!file) return alert("Select a file first!");

                const formData = new FormData();
                formData.append('file', file);
                
                fileBtn.disabled = true;
                spinner.style.display = "inline-block";
                document.getElementById('uploadStatus').innerText = "Analyzing document...";

                try {
                    await fetch('/upload', { method: 'POST', body: formData });
                    document.getElementById('uploadStatus').innerText = "✅ Knowledge base updated!";
                } catch (e) {
                    document.getElementById('uploadStatus').innerText = "❌ Upload failed.";
                } finally {
                    fileBtn.disabled = false;
                    spinner.style.display = "none";
                }
            }

            async function askQuestion() {
                const input = document.getElementById('questionInput');
                const q = input.value;
                if(!q) return;

                const history = document.getElementById('chat-history');
                history.innerHTML += `<div class="msg user-msg">${q}</div>`;
                input.value = "";
                history.scrollTop = history.scrollHeight;

                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ question: q })
                });
                
                const data = await response.json();
                history.innerHTML += `<div class="msg ai-msg">${data.answer}</div>`;
                history.scrollTop = history.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

class QueryRequest(BaseModel):
    question: str

@app.post("/upload", status_code=202)
@limiter.limit("5/minute")
async def upload(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(400, "Unsupported file type.")
    
    content = await file.read()
    background_tasks.add_task(ingest_worker, content, file.content_type)
    return {"status": "Processing"}

def ingest_worker(content: bytes, file_type: str):
    text = extract_text(content, file_type)
    chunks = chunk_text(text)
    for chunk in chunks:
        vector = np.array([get_embedding(chunk)]).astype("float32")
        index.add(vector)
        doc_map[index.ntotal - 1] = chunk

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, req: QueryRequest):
    if index.ntotal == 0:
        raise HTTPException(400, "No data indexed.")
    
    # Retrieval
    q_vec = np.array([get_embedding(req.question)]).astype("float32")
    distances, indices = index.search(q_vec, k=3)
    
    context = "\n".join([doc_map[i] for i in indices[0] if i in doc_map])
    answer = generate_answer(req.question, context)
    
    return {"answer": answer, "sources": context[:200] + "..."}