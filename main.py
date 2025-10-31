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
from logging.handlers import RotatingFileHandler

# Import our fingerprinting modules
from fingerprinting import fingerprint_audio, best_match
from supabase_storage import get_matches, get_info_for_song_id, store_listening_history, get_user_listening_history

# Configure persistent logging to file + console
os.makedirs('logs', exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler with rotation (10MB max, keep 5 backups)
file_handler = RotatingFileHandler(
    'logs/shazomi.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
    Following official Omi API spec: https://docs.omi.me/doc/developer/apps/

    Args:
        uid: User's Omi ID
        message: Notification message text

    Returns:
        True if notification sent successfully, False otherwise
    """
    if not OMI_APP_ID or not OMI_APP_SECRET:
        logger.warning("OMI credentials not set, skipping notification")
        return False

    # v2 API spec: POST /v2/integrations/{app_id}/notification
    # Query params: uid and message
    # Headers: Authorization, Content-Type, Content-Length
    from urllib.parse import quote

    url = f"{OMI_NOTIFICATION_URL}?uid={quote(uid)}&message={quote(message)}"
    headers = {
        "Authorization": f"Bearer {OMI_APP_SECRET}",
        "Content-Type": "application/json",
        "Content-Length": "0"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers)
            response.raise_for_status()
            logger.info(f"✅ Notification sent to user {uid}: {message[:50]}...")
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Notification API error {e.response.status_code}: {e.response.text}")
        return False
    except httpx.TimeoutException:
        logger.error(f"⏱️  Notification timeout for user {uid}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to send notification: {e}")
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
        await send_notification(uid, "🎵 Hmm, I don't recognize this song. Make sure it's in your database!")
        return

    # Get best match
    song_id, score = best_match(matches)

    if not song_id or score < MIN_MATCH_SCORE:
        logger.info(f"User {uid}: Match score too low ({score})")
        await send_notification(uid, "🎵 I heard something, but couldn't match it confidently. Try again with clearer audio!")
        return

    # Get song info
    info = get_info_for_song_id(song_id)

    if info:
        artist, album, title = info
        song_key = f"{title}:{artist}"

        # Check if we've already identified this song recently
        if song_key in identified_songs[uid]:
            logger.info(f"User {uid}: Already identified {title} recently, skipping notification")
            return

        # Add to identified songs
        identified_songs[uid].add(song_key)

        # Store listening history in database
        logger.info(f"📝 Storing listening history for {uid}: {title} by {artist}")
        store_success = store_listening_history(uid, song_id, artist, album, title, score)
        if not store_success:
            logger.warning(f"⚠️  Failed to store listening history for {uid}")

        # Format and send conversational notification
        confidence = "🔥" if score >= 50 else "✨" if score >= 30 else "👍"
        message = f"🎵 That's \"{title}\" by {artist}! {confidence} (Match: {score})"

        await send_notification(uid, message)
        logger.info(f"✅ User {uid}: Identified {title} - {artist} (score: {score})")
    else:
        logger.warning(f"❌ User {uid}: Song ID {song_id} not found in database")
        await send_notification(uid, "🎵 Found a match but couldn't retrieve song info. Database might need updating!")


@app.post("/webhook/audio-chunk")
async def audio_chunk_webhook(request: Request):
    """
    Webhook endpoint for Omi audio bites
    Receives discrete 5s audio chunks from Omi via Railway webhook

    Expected payload:
    - uid: User ID
    - audio_data: Base64 encoded audio bytes
    - sample_rate: Audio sample rate (optional, defaults to 16000)

    Or raw audio bytes with uid in query params
    """
    try:
        # Try to get uid from query params first
        uid = request.query_params.get("uid")
        sample_rate = int(request.query_params.get("sample_rate", 16000))

        # Try to parse JSON payload
        try:
            body = await request.json()
            if not uid:
                uid = body.get("uid")
            sample_rate = body.get("sample_rate", sample_rate)

            # Check if audio is base64 encoded
            import base64
            if "audio_data" in body:
                audio_bytes = base64.b64decode(body["audio_data"])
            else:
                # Fall back to raw body
                audio_bytes = await request.body()
        except:
            # If JSON parsing fails, treat as raw audio bytes
            audio_bytes = await request.body()

        if not uid:
            logger.error("❌ Webhook received without uid")
            return JSONResponse(
                content={"error": "uid required"},
                status_code=400
            )

        bytes_received = len(audio_bytes)

        if bytes_received == 0:
            logger.warning(f"⚠️  User {uid}: Empty audio chunk received")
            return JSONResponse(content={"status": "ok", "message": "Empty audio"})

        duration = bytes_received / (sample_rate * BYTES_PER_SAMPLE)
        logger.info(f"🎤 User {uid}: Webhook received {bytes_received} bytes ({duration:.1f}s) at {sample_rate}Hz")

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # Check if we have enough audio (at least 5 seconds)
        if duration < MIN_AUDIO_DURATION:
            logger.info(f"⚠️  User {uid}: Audio too short ({duration:.1f}s < {MIN_AUDIO_DURATION}s)")
            return JSONResponse(
                content={"status": "ok", "message": f"Audio too short: {duration:.1f}s"}
            )

        # Check cooldown
        if uid in last_identification:
            time_since_last = datetime.now() - last_identification[uid]
            if time_since_last.total_seconds() < COOLDOWN_SECONDS:
                logger.info(f"⏸️  User {uid}: In cooldown period ({time_since_last.total_seconds():.0f}s)")
                return JSONResponse(
                    content={"status": "ok", "message": "Cooldown active"}
                )

        # Update last identification time
        last_identification[uid] = datetime.now()

        # Generate fingerprint hashes
        logger.info(f"🔍 User {uid}: Generating fingerprint...")
        hashes = fingerprint_audio(audio_array)

        if not hashes:
            logger.warning(f"⚠️  User {uid}: No hashes generated")
            await send_notification(uid, "🎵 Couldn't fingerprint that audio. Try again!")
            return JSONResponse(content={"status": "error", "message": "No fingerprint generated"})

        # Find matches in database
        logger.info(f"🔎 User {uid}: Searching database with {len(hashes)} hashes...")
        matches = get_matches(hashes)

        if not matches:
            logger.info(f"❌ User {uid}: No matches found")
            await send_notification(uid, "🎵 Hmm, I don't recognize this song. Make sure it's in your database!")
            return JSONResponse(content={"status": "ok", "message": "No matches found"})

        # Get best match
        song_id, score = best_match(matches)

        if not song_id or score < MIN_MATCH_SCORE:
            logger.info(f"⚠️  User {uid}: Match score too low ({score})")
            await send_notification(uid, "🎵 I heard something, but couldn't match it confidently. Try again with clearer audio!")
            return JSONResponse(content={"status": "ok", "message": f"Score too low: {score}"})

        # Get song info
        info = get_info_for_song_id(song_id)

        if info:
            artist, album, title = info
            song_key = f"{title}:{artist}"

            # Check if we've already identified this song recently
            if song_key in identified_songs[uid]:
                logger.info(f"🔁 User {uid}: Already identified {title} recently, skipping notification")
                return JSONResponse(content={"status": "ok", "message": "Already identified recently"})

            # Add to identified songs
            identified_songs[uid].add(song_key)

            # Store listening history in database
            logger.info(f"📝 Storing listening history for {uid}: {title} by {artist}")
            store_success = store_listening_history(uid, song_id, artist, album, title, score)
            if not store_success:
                logger.warning(f"⚠️  Failed to store listening history for {uid}")

            # Format and send conversational notification
            confidence = "🔥" if score >= 50 else "✨" if score >= 30 else "👍"
            message = f"🎵 That's \"{title}\" by {artist}! {confidence} (Match: {score})"

            await send_notification(uid, message)
            logger.info(f"✅ User {uid}: Identified {title} - {artist} (score: {score})")

            return JSONResponse(content={
                "status": "success",
                "song": {
                    "title": title,
                    "artist": artist,
                    "album": album,
                    "score": score
                }
            })
        else:
            logger.warning(f"❌ User {uid}: Song ID {song_id} not found in database")
            await send_notification(uid, "🎵 Found a match but couldn't retrieve song info. Database might need updating!")
            return JSONResponse(content={"status": "error", "message": "Song info not found"})

    except Exception as e:
        logger.error(f"❌ Webhook error: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


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
        "version": "4.0",
        "mode": "webhook + streaming",
        "active_users": len(audio_buffers),
        "endpoints": {
            "webhook": "/webhook/audio-chunk",
            "streaming": "/audio-stream",
            "test": "/test-notification",
            "history": "/listening-history"
        },
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


@app.get("/listening-history")
async def listening_history(
    uid: str = Query(..., description="User ID"),
    limit: int = Query(10, description="Number of records to return", ge=1, le=100)
):
    """
    Get user's listening history from database

    Args:
        uid: User ID
        limit: Maximum number of records (1-100, default 10)

    Returns:
        List of listening history records
    """
    try:
        logger.info(f"📊 Fetching listening history for user {uid} (limit: {limit})")
        history = get_user_listening_history(uid, limit)

        return JSONResponse(
            content={
                "uid": uid,
                "count": len(history),
                "history": history
            }
        )
    except Exception as e:
        logger.error(f"❌ Error fetching listening history: {e}")
        return JSONResponse(
            content={
                "error": "Failed to fetch listening history",
                "details": str(e)
            },
            status_code=500
        )


@app.post("/test-notification")
async def test_notification(uid: str = Query(..., description="User ID to send test notification")):
    """
    Test endpoint to manually trigger a notification

    Args:
        uid: User ID to send the test notification to

    Returns:
        Success/failure status
    """
    logger.info(f"Test notification requested for user {uid}")

    test_message = f"🎵 shazomi test notification sent at {datetime.now().strftime('%I:%M %p')}"
    success = await send_notification(uid, test_message)

    if success:
        logger.info(f"Test notification sent successfully to {uid}")
        return JSONResponse(
            content={
                "status": "success",
                "uid": uid,
                "message": test_message,
                "timestamp": datetime.now().isoformat()
            }
        )
    else:
        logger.error(f"Failed to send test notification to {uid}")
        return JSONResponse(
            content={
                "status": "error",
                "uid": uid,
                "error": "Failed to send notification. Check OMI credentials."
            },
            status_code=500
        )


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
    logger.info("🎵 shazomi v4.0 started - Webhook + Streaming mode")
    logger.info("📍 Endpoints:")
    logger.info("  - Webhook (Omi audio bites): POST /webhook/audio-chunk")
    logger.info("  - Real-time streaming: POST /audio-stream")
    logger.info("  - Listening history: GET /listening-history?uid=<uid>")
    logger.info("  - Test notification: POST /test-notification?uid=<uid>")
    logger.info("  - Setup check: GET /setup-completed?uid=<uid>")
    logger.info("  - Health: GET /health")
    logger.info(f"🔍 Algorithm: abracadabra (Shazam-based)")
    logger.info(f"💾 Database: Supabase PostgreSQL")
    logger.info(f"📝 Logs: logs/shazomi.log")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
