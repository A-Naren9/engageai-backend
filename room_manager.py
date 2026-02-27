"""
Room and participant lifecycle management for EngageAI Conference.
All state is in-memory; rooms persist for the lifetime of the server process.
"""

import secrets
import random
import string
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Participant:
    id: str
    name: str
    room_id: str
    joined_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    emotion: dict = field(default_factory=dict)
    audio: dict = field(default_factory=dict)
    engagement: dict = field(default_factory=dict)
    is_online: bool = False


@dataclass
class Room:
    id: str
    admin_token: str
    created_at: float = field(default_factory=time.time)
    participants: Dict[str, Participant] = field(default_factory=dict)


class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, Room] = {}

    def create_room(self) -> Room:
        """Create a new room with a human-readable 6-character code."""
        for _ in range(20):
            code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
            if code not in self.rooms:
                break
        admin_token = secrets.token_urlsafe(24)
        room = Room(id=code, admin_token=admin_token)
        self.rooms[code] = room
        return room

    def get_room(self, room_id: str) -> Optional[Room]:
        return self.rooms.get(room_id.upper())

    def add_participant(self, room_id: str, name: str) -> Optional[Participant]:
        room = self.get_room(room_id)
        if not room:
            return None
        pid = secrets.token_urlsafe(10)
        participant = Participant(
            id=pid,
            name=name.strip()[:50],
            room_id=room_id.upper(),
        )
        room.participants[pid] = participant
        return participant

    def get_participant(self, room_id: str, participant_id: str) -> Optional[Participant]:
        room = self.get_room(room_id)
        if room:
            return room.participants.get(participant_id)
        return None
