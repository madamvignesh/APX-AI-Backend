from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os, uuid, logging
from datetime import datetime, timezone
from pathlib import Path


from dotenv import load_dotenv
load_dotenv()

# genai.configure(api_key="AIzaSyDTJoCV-7sD8rEs43CUjbMz6J0pu3CAxxo")

# for m in genai.list_models():
#     print(m.name, m.supported_generation_methods)
# -----------------------------
# Helper: read env or Render secret file
# -----------------------------

def get_secret(key: str) -> str | None:
    """Return environment variable value or /etc/secrets/<key> if present."""
    val = os.environ.get(key)
    if val:
        return val
    secret_path = Path(f"/etc/secrets/{key}")
    if secret_path.exists():
        try:
            return secret_path.read_text(encoding="utf-8").strip()
        except Exception:
            logging.exception(f"Failed to read secret file for {key}")
    return None


# -----------------------------
# Configuration / Secrets
# -----------------------------

# Note: On Render, do NOT rely on a local .env file. Use Render environment variables or secret files.
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY not set — Gemini calls will fail until you provide the key.")

# Configure Gemini SDK
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        logging.exception("Failed to configure Google Generative AI client")

# -----------------------------
# FastAPI app + router
# -----------------------------
app = FastAPI()
api_router = APIRouter(prefix="/api")

# -----------------------------
# Pydantic models
# -----------------------------
class ChatMessage(BaseModel):
    message: str
    conversation_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

class ContactForm(BaseModel):
    name: str
    email: str
    message: str

# -----------------------------
# Chat endpoint
# -----------------------------
@api_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(msg: ChatMessage):
    system_prompt = (
        "You are an emergency assistant. Give clear, step-by-step help. The user is in an emergency situation. Do not respond with anything else. Respond in 4 lines"
    )

    try:
        model = genai.GenerativeModel("gemini-flash-latest")

        response = model.generate_content([
            system_prompt,
            f"User: {msg.message}"
        ])

        ai_text = response.text.strip() if hasattr(response, "text") else "Help is on the way."

        return {
            "response": ai_text,
            "conversation_id": msg.conversation_id or str(uuid.uuid4())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# -----------------------------
# Contact endpoint
# -----------------------------
@api_router.post("/contact")
async def contact(form: ContactForm):
    try:
        return {"success": True, "message": "Thank you! We'll get back to you soon."}
    except Exception as e:
        logging.exception(f"Contact form error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit contact form.")

# -----------------------------
# Health / readiness endpoints
# -----------------------------
@app.get("/api")
async def root_index():
    return {"message": "APX AI Backend is running", "docs": "/docs"}

@app.get("/")
async def root():
    # lightweight Gemini check
    if not GEMINI_API_KEY:
        return {"message": "Gemini Check Failed: GEMINI_API_KEY not set.", "status": "error", "gemini_check": "failed"}

    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content("Reply exactly with: 'APX AI System is fully operational.'")
        text = getattr(response, "text", None) or (response.candidates[0].get("content") if getattr(response, "candidates", None) else None)
        return {"message": (text or "No text returned").strip(), "status": "active", "gemini_check": "success"}
    except Exception as e:
        logging.exception("Gemini healthcheck failed")
        return {"message": f"Gemini Check Failed: {str(e)}", "status": "error", "gemini_check": "failed"}

# -----------------------------
# CORS middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.split(",") if CORS_ORIGINS else ["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
