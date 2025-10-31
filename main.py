#!/usr/bin/env python3
"""
shazomi - In-house audio fingerprinting for Omi
Real-time song identification using abracadabra algorithm + Supabase
"""

import os
import logging
import struct
import asyncio
import numpy as np
from typing import Dict
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx

# Import our fingerprinting modules
from fingerprinting import fingerprint_audio, best_match
from supabase_storage import get_matches, get_info_for_song_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
OMI_APP_ID = os.getenv("OMI_APP_ID")
OMI_APP_SECRET = os.getenv("OMI_APP_SECRET")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# API endpoints
OMI_NOTIFICATION_URL = f"https://api.omi.me/v2/integrations/{OMI_APP_ID}/notification"

# Audio processing settings
SAMPLE_RATE = 16000  # Hz - Omi DevKit 2 sample rate
BYTES_PER_SAMPLE = 2  # 16-bit audio
MIN_AUDIO_DURATION = 5  # seconds - minimum audio needed for fingerprinting
MAX_AUDIO_DURATION = 10  # seconds - maximum to accumulate
COOLDOWN_SECONDS = 30  # seconds between identifications per user
MIN_MATCH_SCORE = 10  # Minimum score to consider a match valid

app = FastAPI(title="shazomi")

# In-memory storage for audio streams and last identification times
audio_buffers: Dict[str, bytearray] = defaultdict(bytearray)
last_identification: Dict[str, datetime] = {}
identified_songs: Dict[str, set] = defaultdict(set)  # Track identified songs per user


async def send_notification(uid: str, message: str) -> bool:
    """
    Send notification to Omi user using v2 API

    Args:
        uid: User's Omi ID
        message: Notification message text

    Returns:
        True if notification sent successfully, False otherwise
    """
    if not OMI_APP_ID or not OMI_APP_SECRET:
        logger.warning("OMI credentials not set, skipping notification")
        return False

    # v2 API uses query parameters for uid and message
    url = f"{OMI_NOTIFICATION_URL}?uid={uid}&message={message}"
    headers = {
        "Authorization": f"Bearer {OMI_APP_SECRET}",
        "Content-Type": "application/json",
        "Content-Length": "0"
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers)
            response.raise_for_status()
            logger.info(f"Notification sent to user {uid}")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"Notification API error: {e.response.status_code} - {e.response.text}")
        return False
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return False


async def process_audio_buffer(uid: str, sample_rate: int):
    """
    Process accumulated audio buffer for a user using fingerprinting

    Args:
        uid: User ID
        sample_rate: Audio sample rate
    """
    buffer = audio_buffers.get(uid)
    if not buffer:
        return

    # Check if we have enough audio
    duration = len(buffer) / (sample_rate * BYTES_PER_SAMPLE)
    if duration < MIN_AUDIO_DURATION:
        logger.info(f"User {uid}: Not enough audio yet ({duration:.1f}s)")
        return

    # Check cooldown
    if uid in last_identification:
        time_since_last = datetime.now() - last_identification[uid]
        if time_since_last.total_seconds() < COOLDOWN_SECONDS:
            logger.info(f"User {uid}: In cooldown period")
            audio_buffers[uid].clear()  # Clear buffer but don't process
            return

    logger.info(f"User {uid}: Processing {duration:.1f}s of audio")

    # Convert bytes to numpy array (int16)
    audio_array = np.frombuffer(bytes(buffer), dtype=np.int16)

    # Clear buffer before processing to avoid re-processing
    audio_buffers[uid].clear()
    last_identification[uid] = datetime.now()

    # Generate fingerprint hashes
    logger.info(f"User {uid}: Generating fingerprint...")
    hashes = fingerprint_audio(audio_array)

    if not hashes:
        logger.warning(f"User {uid}: No hashes generated")
        await send_notification(uid, "Could not fingerprint audio")
        return

    # Find matches in database
    logger.info(f"User {uid}: Searching for matches...")
    matches = get_matches(hashes)

    if not matches:
        logger.info(f"User {uid}: No matches found")
        await send_notification(uid, "🎵 Song not in database")
        return

    # Get best match
    song_id, score = best_match(matches)

    if not song_id or score < MIN_MATCH_SCORE:
        logger.info(f"User {uid}: Match score too low ({score})")
        await send_notification(uid, "🎵 No confident match found")
        return

    # Get song info
    info = get_info_for_song_id(song_id)

    if info:
        artist, album, title = info
        song_key = f"{title}:{artist}"

        # Check if we've already identified this song recently
        if song_key in identified_songs[uid]:
            logger.info(f"User {uid}: Already identified {title}")
            return

        # Add to identified songs
        identified_songs[uid].add(song_key)

        # Format and send notification
        now = datetime.now()
        message = f"🎵 {title} by {artist} (score: {score}) at {now.strftime('%I:%M %p')}"

        await send_notification(uid, message)
        logger.info(f"User {uid}: Identified {title} - {artist} (score: {score})")
    else:
        logger.warning(f"User {uid}: Song ID {song_id} not found in database")


@app.post("/audio-stream")
async def audio_stream(
    request: Request,
    uid: str = Query(..., description="User ID"),
    sample_rate: int = Query(16000, description="Audio sample rate in Hz")
):
    """
    Real-time audio streaming endpoint
    Receives raw PCM audio bytes from Omi device

    Args:
        uid: User ID from query parameter
        sample_rate: Audio sample rate (default 16000 Hz)
    """
    try:
        # Read raw audio bytes
        audio_bytes = await request.body()
        bytes_received = len(audio_bytes)

        if bytes_received == 0:
            return PlainTextResponse("OK", status_code=200)

        logger.info(f"User {uid}: Received {bytes_received} bytes at {sample_rate}Hz")

        # Append to user's audio buffer
        audio_buffers[uid].extend(audio_bytes)

        # Check if we should process the buffer
        current_duration = len(audio_buffers[uid]) / (sample_rate * BYTES_PER_SAMPLE)

        if current_duration >= MAX_AUDIO_DURATION:
            # Process in background to not block the response
            asyncio.create_task(process_audio_buffer(uid, sample_rate))

        return PlainTextResponse("OK", status_code=200)

    except Exception as e:
        logger.error(f"Audio stream error: {e}", exc_info=True)
        return PlainTextResponse("Error", status_code=500)


@app.get("/setup-completed")
async def setup_completed(uid: str = Query(..., description="User ID")):
    """
    Check if setup is completed for a user
    For shazomi, setup is always complete as it requires no user configuration

    Args:
        uid: User ID

    Returns:
        JSON with is_setup_completed flag
    """
    logger.info(f"Setup check for user {uid}")
    return JSONResponse(
        content={"is_setup_completed": True}
    )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "shazomi",
        "version": "3.0",
        "mode": "audio_fingerprinting",
        "active_users": len(audio_buffers),
        "config": {
            "sample_rate": SAMPLE_RATE,
            "min_duration": MIN_AUDIO_DURATION,
            "max_duration": MAX_AUDIO_DURATION,
            "cooldown": COOLDOWN_SECONDS,
            "min_match_score": MIN_MATCH_SCORE
        }
    }


@app.get("/stats")
async def stats(uid: str = Query(None, description="Optional user ID for user-specific stats")):
    """
    Get statistics about audio processing

    Args:
        uid: Optional user ID to get user-specific stats
    """
    if uid:
        buffer_size = len(audio_buffers.get(uid, bytearray()))
        duration = buffer_size / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        return {
            "uid": uid,
            "buffer_size_bytes": buffer_size,
            "buffer_duration_seconds": round(duration, 2),
            "songs_identified": len(identified_songs.get(uid, set())),
            "last_identification": last_identification.get(uid).isoformat() if uid in last_identification else None
        }

    return {
        "total_users": len(audio_buffers),
        "total_songs_identified": sum(len(songs) for songs in identified_songs.values()),
        "users_with_audio": [uid for uid, buf in audio_buffers.items() if len(buf) > 0]
    }


# Cleanup task to prevent memory leaks
async def cleanup_old_data():
    """Periodically clean up old data to prevent memory leaks"""
    while True:
        await asyncio.sleep(300)  # Run every 5 minutes

        now = datetime.now()
        cutoff = now - timedelta(hours=1)

        # Clean up old identifications
        for uid in list(last_identification.keys()):
            if last_identification[uid] < cutoff:
                del last_identification[uid]
                if uid in identified_songs:
                    identified_songs[uid].clear()
                logger.info(f"Cleaned up old data for user {uid}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    # Check environment variables
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL and SUPABASE_KEY must be set!")
    if not OMI_APP_ID or not OMI_APP_SECRET:
        logger.warning("OMI credentials not set - notifications will not work")

    asyncio.create_task(cleanup_old_data())
    logger.info("shazomi started - Real-time audio fingerprinting mode")
    logger.info(f"Audio stream endpoint: /audio-stream")
    logger.info(f"Setup check endpoint: /setup-completed")
    logger.info(f"Algorithm: abracadabra (Shazam-based)")
    logger.info(f"Database: Supabase PostgreSQL")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
