"""
main.py — FastAPI backend for VoiceGo (đặt xe bằng giọng nói cho người khiếm thị).

Server-side only what needs to be: FPT.AI STT/TTS proxies (CORS + keep keys off
the client) and the conversational agent loop (Groq function-calling + tools).
"""
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from data import NODES
from voice import speech_to_text, text_to_speech, whisper_stt
from geocode import resolve_destination
from agent import run_agent

app = FastAPI(title="VoiceGo API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TtsRequest(BaseModel):
    text: str
    voice: str = "banmai"
    speed: str = ""


class GeocodeRequest(BaseModel):
    text: str
    lat: float | None = None
    lng: float | None = None


class ChatRequest(BaseModel):
    messages: list[dict]


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "voicego", "places": len(NODES)}


@app.post("/api/voice/stt")
async def voice_stt(file: UploadFile = File(...)):
    """Transcribe Vietnamese audio via Groq Whisper (accurate); FPT ASR fallback."""
    audio = await file.read()
    result = whisper_stt(audio, file.filename or "speech.wav")
    if not result.get("text"):
        result = speech_to_text(audio)  # fallback if Whisper unavailable
    return result


@app.post("/api/voice/tts")
def voice_tts(req: TtsRequest):
    """Synthesize Vietnamese speech via FPT TTS, return mp3 bytes."""
    audio = text_to_speech(req.text, req.voice, req.speed)
    if audio is None:
        return JSONResponse({"error": "tts_failed"}, status_code=502)
    return Response(content=audio, media_type="audio/mpeg")


@app.post("/api/agent/chat")
def agent_chat(req: ChatRequest):
    """One turn of the conversational booking agent (Groq + tools)."""
    return run_agent(req.messages)


@app.post("/api/voice/geocode")
def voice_geocode(req: GeocodeRequest):
    """Debug: resolve a place name to a real address + coordinates."""
    return resolve_destination(req.text, req.lat, req.lng)
