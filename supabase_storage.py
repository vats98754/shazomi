"""
Supabase storage adapter for abracadabra
Replaces SQLite storage with Supabase PostgreSQL
"""

import os
import uuid
from collections import defaultdict
from typing import List, Tuple, Optional
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)

# Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """Get or create Supabase client"""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def setup_db():
    """
    Create the database tables in Supabase.

    Run the following SQL in Supabase SQL Editor:

    CREATE TABLE IF NOT EXISTS fingerprint_hash (
        id BIGSERIAL PRIMARY KEY,
        hash_value BIGINT NOT NULL,
        time_offset REAL NOT NULL,
        song_id TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_hash_value ON fingerprint_hash (hash_value);

    CREATE TABLE IF NOT EXISTS song_info (
        song_id TEXT PRIMARY KEY,
        artist TEXT,
        album TEXT,
        title TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    logger.info("Database should be set up manually in Supabase SQL Editor")
    logger.info("See docstring for SQL commands")


def song_in_db(filename: str) -> bool:
    """
    Check whether a path has already been registered.

    Args:
        filename: The path to check

    Returns:
        Whether the path exists in the database yet
    """
    try:
        supabase = get_supabase()
        song_id = str(uuid.uuid5(uuid.NAMESPACE_OID, filename).int)

        result = supabase.table("song_info").select("song_id").eq("song_id", song_id).execute()

        return len(result.data) > 0
    except Exception as e:
        logger.error(f"Error checking if song in DB: {e}")
        return False


def store_song(hashes: List[Tuple[int, float, str]], song_info: Tuple[str, str, str]):
    """
    Register a song in the database.

    Args:
        hashes: A list of tuples of the form (hash, time offset, song_id)
        song_info: A tuple of form (artist, album, title) describing the song
    """
    if len(hashes) < 1:
        logger.warning("No hashes to store")
        return

    try:
        supabase = get_supabase()
        song_id = hashes[0][2]

        # Store song info
        artist, album, title = song_info
        song_data = {
            "song_id": song_id,
            "artist": artist or "Unknown",
            "album": album or "Unknown",
            "title": title or "Unknown"
        }

        supabase.table("song_info").insert(song_data).execute()
        logger.info(f"Stored song info: {title}")

        # Store hashes in batches (Supabase has limits on bulk inserts)
        batch_size = 1000
        hash_data = [{"hash_value": h[0], "time_offset": h[1], "song_id": h[2]} for h in hashes]

        for i in range(0, len(hash_data), batch_size):
            batch = hash_data[i:i+batch_size]
            supabase.table("fingerprint_hash").insert(batch).execute()
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(hash_data)-1)//batch_size + 1}")

    except Exception as e:
        logger.error(f"Error storing song: {e}")
        raise


def get_matches(hashes: List[Tuple[int, float, str]], threshold: int = 5) -> dict:
    """
    Get matching songs for a set of hashes.

    Args:
        hashes: A list of hashes
        threshold: Return songs that have more than threshold matches

    Returns:
        A dictionary mapping song_id to a list of time offset tuples.
        The tuples are of the form (result offset, original hash offset).
    """
    try:
        supabase = get_supabase()

        # Create hash lookup dict
        h_dict = {}
        for h, t, _ in hashes:
            h_dict[h] = t

        # Query in batches to avoid URL length limits
        all_results = []
        batch_size = 100
        hash_values = [h[0] for h in hashes]

        for i in range(0, len(hash_values), batch_size):
            batch = hash_values[i:i+batch_size]

            result = supabase.table("fingerprint_hash").select("hash_value, time_offset, song_id").in_("hash_value", batch).execute()

            if result.data:
                all_results.extend(result.data)

        # Build result dictionary
        result_dict = defaultdict(list)
        for r in all_results:
            result_dict[r["song_id"]].append((r["time_offset"], h_dict[r["hash_value"]]))

        # Filter by threshold
        filtered = {k: v for k, v in result_dict.items() if len(v) >= threshold}

        logger.info(f"Found {len(filtered)} matching songs")
        return filtered

    except Exception as e:
        logger.error(f"Error getting matches: {e}")
        return {}


def get_info_for_song_id(song_id: str) -> Optional[Tuple[str, str, str]]:
    """
    Lookup song information for a given ID.

    Args:
        song_id: The song ID

    Returns:
        Tuple of (artist, album, title) or None if not found
    """
    try:
        supabase = get_supabase()

        result = supabase.table("song_info").select("artist, album, title").eq("song_id", song_id).execute()

        if result.data and len(result.data) > 0:
            data = result.data[0]
            return (data["artist"], data["album"], data["title"])

        return None

    except Exception as e:
        logger.error(f"Error getting song info: {e}")
        return None


def store_listening_history(uid: str, song_id: str, artist: str, album: str, title: str, match_score: int) -> bool:
    """
    Store a listening history record for a user.

    Args:
        uid: User ID
        song_id: Song ID
        artist: Artist name
        album: Album name
        title: Song title
        match_score: Match confidence score

    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase()

        history_data = {
            "uid": uid,
            "song_id": song_id,
            "artist": artist,
            "album": album,
            "title": title,
            "match_score": match_score
        }

        supabase.table("listening_history").insert(history_data).execute()
        logger.info(f"Stored listening history: {uid} listened to {title}")
        return True

    except Exception as e:
        logger.error(f"Error storing listening history: {e}")
        return False


def get_user_listening_history(uid: str, limit: int = 10) -> List[dict]:
    """
    Get recent listening history for a user.

    Args:
        uid: User ID
        limit: Maximum number of records to return

    Returns:
        List of listening history records
    """
    try:
        supabase = get_supabase()

        result = supabase.table("listening_history")\
            .select("*")\
            .eq("uid", uid)\
            .order("detected_at", desc=True)\
            .limit(limit)\
            .execute()

        return result.data if result.data else []

    except Exception as e:
        logger.error(f"Error getting listening history: {e}")
        return []
