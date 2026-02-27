"""
EngageAI Conference Backend — FastAPI
Handles room management, participant WebSocket streams (emotion + audio),
and an admin WebSocket that receives live updates for all participants.
"""

import asyncio
import base64
import json
import logging
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from room_manager import RoomManager
from emotion_detector import EmotionDetector
from audio_analyzer import AudioAnalyzer
from engagement_scorer import EngagementScorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
logger = logging.getLogger("engageai")

app = FastAPI(title="EngageAI Conference", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ─────────────────────────────────────────────────────────────────
room_manager     = RoomManager()
emotion_detector = EmotionDetector()
audio_analyzer   = AudioAnalyzer()

# admin WebSocket connections keyed by room_id
admin_connections: dict[str, WebSocket] = {}


# ── REST endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "service": "EngageAI Conference"}


@app.post("/api/rooms")
async def create_room():
    room = room_manager.create_room()
    logger.info(f"Room created: {room.id}")
    return {"room_id": room.id, "admin_token": room.admin_token}


@app.get("/api/rooms/{room_id}")
async def get_room(room_id: str):
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return {
        "room_id": room.id,
        "participants": [
            {"id": p.id, "name": p.name, "is_online": p.is_online}
            for p in room.participants.values()
        ],
    }


class JoinRequest(BaseModel):
    name: str


@app.post("/api/rooms/{room_id}/join")
async def join_room(room_id: str, req: JoinRequest):
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    participant = room_manager.add_participant(room_id, req.name)
    if not participant:
        raise HTTPException(status_code=404, detail="Room not found")
    logger.info(f"Participant joined: {participant.name} → room {room_id}")
    return {
        "participant_id": participant.id,
        "room_id": participant.room_id,
        "name": participant.name,
    }


# ── WebSocket: participant ─────────────────────────────────────────────────────

@app.websocket("/ws/participant/{room_id}/{participant_id}")
async def participant_ws(websocket: WebSocket, room_id: str, participant_id: str):
    participant = room_manager.get_participant(room_id, participant_id)
    if not participant:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    participant.is_online = True
    scorer = EngagementScorer()
    logger.info(f"[WS] {participant.name} connected to room {room_id}")

    await _notify_admin(room_id, participant, "joined")

    try:
        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)
            result: dict = {}

            # ── Video frame ───────────────────────────────────────────
            if "frame" in data:
                try:
                    frame_bytes = base64.b64decode(data["frame"])
                    arr   = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        result["emotion"] = emotion_detector.detect(frame)
                except Exception as e:
                    logger.error(f"Frame error ({participant.name}): {e}")
                    result["emotion"] = {"dominant": "unknown", "face_detected": False}

            # ── Audio chunk ───────────────────────────────────────────
            if "audio" in data:
                try:
                    sr = int(data.get("sampleRate", 16000))
                    result["audio"] = audio_analyzer.analyze(data["audio"], sr)
                except Exception as e:
                    logger.error(f"Audio error ({participant.name}): {e}")

            # ── Engagement score ──────────────────────────────────────
            result["engagement"] = scorer.score(result)

            # Update participant record
            if "emotion"    in result: participant.emotion    = result["emotion"]
            if "audio"      in result: participant.audio      = result["audio"]
            if "engagement" in result: participant.engagement = result["engagement"]
            participant.last_seen = time.time()

            # Reply to participant
            await websocket.send_text(json.dumps(result))

            # Push update to admin
            await _notify_admin(room_id, participant, "update")

    except WebSocketDisconnect:
        participant.is_online = False
        logger.info(f"[WS] {participant.name} disconnected from room {room_id}")
        await _notify_admin(room_id, participant, "left")


# ── WebSocket: admin ───────────────────────────────────────────────────────────

@app.websocket("/ws/admin/{room_id}")
async def admin_ws(
    websocket: WebSocket,
    room_id: str,
    token: str = Query(None),
):
    room = room_manager.get_room(room_id)
    if not room or room.admin_token != token:
        await websocket.close(code=4003)
        return

    await websocket.accept()
    admin_connections[room_id] = websocket
    logger.info(f"[WS] Admin connected to room {room_id}")

    # Send snapshot of current participants
    for p in room.participants.values():
        await websocket.send_text(json.dumps({
            "type": "update",
            "participant": _participant_payload(p),
        }))

    try:
        while True:
            # Wait for keep-alive pings from the frontend
            await websocket.receive_text()
    except WebSocketDisconnect:
        admin_connections.pop(room_id, None)
        logger.info(f"[WS] Admin disconnected from room {room_id}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _participant_payload(p) -> dict:
    return {
        "id":         p.id,
        "name":       p.name,
        "emotion":    p.emotion,
        "audio":      p.audio,
        "engagement": p.engagement,
        "is_online":  p.is_online,
    }


async def _notify_admin(room_id: str, participant, event: str = "update"):
    ws = admin_connections.get(room_id)
    if not ws:
        return
    try:
        await ws.send_text(json.dumps({
            "type":        event,
            "participant": _participant_payload(participant),
        }))
    except Exception as e:
        logger.error(f"Admin notify failed: {e}")
        admin_connections.pop(room_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
