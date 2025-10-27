from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import google.generativeai as genai
import httpx, os, uuid, logging
from datetime import datetime, timezone
from pathlib import Path

# Load env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# MongoDB
client = AsyncIOMotorClient(os.environ["MONGO_URL"])
db = client[os.environ["DB_NAME"]]

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# n8n fallback
N8N_WEBHOOK_URL = "https://team-apx.app.n8n.cloud/webhook/33ufnHMgyOogu0dJFw9E3jiOfu1_6oeV1asyVaGkY8pXu2Y55"

# FastAPI app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Models
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

# Chat endpoint
@api_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(msg: ChatMessage):
    conversation_id = msg.conversation_id or str(uuid.uuid4())

    system_prompt = """
    You are APX AI, a Pre-Ambulance Assistant trained to provide immediate,
    life-saving first aid guidance during medical emergencies.
    Your role: 
    - Provide CLEAR, STEP-BY-STEP emergency instructions
    - Stay CALM and REASSURING
    - Ask critical questions to assess the situation
    - Guide the caller through first aid procedures
    - Keep responses CONCISE and ACTIONABLE
    - Prioritize life-saving actions
    - Remind caller to call emergency services (ambulance - 108 or 100) if not already done.
    Always: 1. Ask "What is the emergency?" 2. Assess severity 3. Provide clear, numbered steps 4. Stay with caller.
    Keep responses under 100 words when possible.
    """

    try:
        # ✅ Get past 5 messages for context
        history_cursor = db.conversations.find(
            {"conversation_id": conversation_id}
        ).sort("timestamp", -1).limit(5)
        past_messages = await history_cursor.to_list(length=5)
        past_messages.reverse()  # chronological order

        # Build conversation history
        conversation_context = [system_prompt]
        for h in past_messages:
            conversation_context.append(f"User: {h['user_message']}")
            conversation_context.append(f"APX AI: {h['ai_response']}")

        # Append current message
        conversation_context.append(f"User: {msg.message}")

        # Generate response
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(conversation_context)
        ai_text = resp.text.strip() if resp.text else "Please describe the emergency."

        # Store new message
        await db.conversations.insert_one({
            "conversation_id": conversation_id,
            "user_message": msg.message,
            "ai_response": ai_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        return ChatResponse(response=ai_text, conversation_id=conversation_id)

    except Exception as e:
        logging.error(f"Chat error: {e}")

        # Fallback: send to n8n webhook
        try:
            async with httpx.AsyncClient() as client_http:
                res = await client_http.post(
                    N8N_WEBHOOK_URL, json={"message": msg.message}, timeout=10.0
                )
                ai_text = res.json().get("response", "Please call emergency services immediately.")
            return ChatResponse(response=ai_text, conversation_id=conversation_id)
        except Exception as e2:
            logging.error(f"n8n fallback failed: {e2}")
            raise HTTPException(status_code=500, detail="AI service unavailable.")

# Contact endpoint
@api_router.post("/contact")
async def contact(form: ContactForm):
    try:
        await db.contact_queries.insert_one({
            "id": str(uuid.uuid4()),
            "name": form.name,
            "email": form.email,
            "message": form.message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return {"success": True, "message": "Thank you! We'll get back to you soon."}
    except Exception as e:
        logging.error(f"Contact form error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit contact form.")

# Root
@api_router.get("/api")
async def root():
    return {"message": "APX AI - Emergency Assistant API", "status": "active"}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(api_router)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Shutdown Mongo
@app.on_event("shutdown")
async def shutdown_db():
    client.close()
