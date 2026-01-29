import os
import numpy as np
from PyPDF2 import PdfReader
from io import BytesIO
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# 1. LOCAL FREE EMBEDDINGS (No API Key needed)
# This downloads a ~90MB model the first time you run it
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. FREE LLM (Using Hugging Face Hub or any free provider)
# You can get a free token from huggingface.co/settings/tokens
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# 1. LOAD the .env file (This is the step often missed)
load_dotenv()

# 2. RETRIEVE the token
HF_TOKEN = os.getenv("HF_TOKEN")

# 3. VERIFY it exists before starting the client
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found! Check your .env file.")

# 4. PASS the token explicitly to the client
hf_client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2", 
    token=HF_TOKEN
)

def extract_text(file_content: bytes, file_type: str) -> str:
    if file_type == "application/pdf":
        reader = PdfReader(BytesIO(file_content))
        return "".join([page.extract_text() for page in reader.pages])
    return file_content.decode("utf-8")

def chunk_text(text: str, size: int = 700, overlap: int = 100):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i : i + size])
    return chunks

def get_embedding(text: str):
    # This runs on your CPU, no cost!
    return embed_model.encode(text)

def generate_answer(question: str, context: str):
    # Construct the message list for the Chat API
    messages = [
        {
            "role": "system", 
            "content": "Answer the question using only the provided context."
        },
        {
            "role": "user", 
            "content": f"Context: {context}\n\nQuestion: {question}"
        }
    ]
    
    # Use chat_completion instead of text_generation
    response = hf_client.chat_completion(
        messages=messages,
        max_tokens=500
    )
    
    return response.choices[0].message.content