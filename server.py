from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
import google.generativeai as genai
import os, uuid, logging
from datetime import datetime, timezone
from pathlib import Path

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
MONGO_URL = get_secret("MONGO_URL") or ""
DB_NAME = get_secret("DB_NAME") or os.environ.get("DB_NAME") or "apx_db"
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

if not MONGO_URL:
    logging.warning("MONGO_URL not set — defaulting to localhost (won't work in production).")

if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY not set — Gemini calls will fail until you provide the key.")

# Configure Gemini SDK
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        logging.exception("Failed to configure Google Generative AI client")

# -----------------------------
# MongoDB client
# -----------------------------
try:
    client = AsyncIOMotorClient(MONGO_URL) if MONGO_URL else AsyncIOMotorClient()
    db = client[DB_NAME]
except Exception:
    logging.exception("Failed to initialize MongoDB client; continuing without DB access")
    client = None
    db = None

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
    conversation_id = msg.conversation_id or str(uuid.uuid4())

    system_prompt = (
        "You are APX AI, a Pre-Ambulance Assistant trained to provide immediate,\n"
        "life-saving first aid guidance during medical emergencies.\n"
        "Your role: Provide CLEAR, STEP-BY-STEP emergency instructions, stay CALM, \n"
        "ask critical questions, guide through procedures, and remind to call emergency services."
    )

    try:
        # pull last 5 messages (if DB available)
        past_messages = []
        if db is not None:
            history_cursor = db.conversations.find({"conversation_id": conversation_id}).sort("timestamp", -1).limit(5)
            past_messages = await history_cursor.to_list(length=5)
            past_messages.reverse()

        # Build conversation history text
        conversation_context = [system_prompt]
        for h in past_messages:
            conversation_context.append(f"User: {h.get('user_message','')}")
            conversation_context.append(f"APX AI: {h.get('ai_response','')}")

        conversation_context.append(f"User: {msg.message}")

        # Use a supported model name
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        # generate_content accepts text or list/context depending on SDK; keep as provided list
        resp = model.generate_content(conversation_context)

        # The SDK may return different shapes; try common access patterns
        ai_text = None
        if hasattr(resp, "text") and resp.text:
            ai_text = resp.text.strip()
        else:
            # try content in candidate responses
            try:
                # Some SDKs return .candidates[0].content or .candidates[0].output
                candidates = getattr(resp, "candidates", None)
                if candidates and len(candidates) > 0:
                    first = candidates[0]
                    ai_text = first.get("content") or first.get("output") or str(first)
            except Exception:
                ai_text = None

        if not ai_text:
            ai_text = "Please describe the emergency."

        # Store conversation (if DB available)
        if db is not None:
            await db.conversations.insert_one({
                "conversation_id": conversation_id,
                "user_message": msg.message,
                "ai_response": ai_text,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        return ChatResponse(response=ai_text, conversation_id=conversation_id)

    except Exception as e:
        logging.exception(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="AI service unavailable.")

# -----------------------------
# Contact endpoint
# -----------------------------
@api_router.post("/contact")
async def contact(form: ContactForm):
    try:
        if db is None:
            logging.warning("Contact form submitted but DB is not available; request will not be persisted.")
            return {"success": True, "message": "Thank you! We'll get back to you soon (note: DB unavailable)."}

        await db.contact_queries.insert_one({
            "id": str(uuid.uuid4()),
            "name": form.name,
            "email": form.email,
            "message": form.message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
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
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
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

# Startup / Shutdown
@app.on_event("startup")
async def startup_event():
    masked = MONGO_URL.split("@")[-1] if MONGO_URL and "@" in MONGO_URL else ("localhost/local" if MONGO_URL else "not-set")
    logging.info(f"Connecting to MongoDB at: ...@{masked}")

@app.on_event("shutdown")
async def shutdown_db():
    try:
        if client:
            client.close()
    except Exception:
        logging.exception("Error closing MongoDB client")
