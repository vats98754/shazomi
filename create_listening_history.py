#!/usr/bin/env python3
"""
Create the listening_history table in Supabase
"""

from dotenv import load_dotenv
from supabase_storage import get_supabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def create_table():
    """Create listening_history table"""

    sql = """
    CREATE TABLE IF NOT EXISTS listening_history (
        id BIGSERIAL PRIMARY KEY,
        uid TEXT NOT NULL,
        song_id TEXT NOT NULL,
        artist TEXT NOT NULL,
        album TEXT,
        title TEXT NOT NULL,
        match_score INTEGER NOT NULL,
        detected_at TIMESTAMP DEFAULT NOW(),
        CONSTRAINT fk_song FOREIGN KEY (song_id) REFERENCES song_info(song_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_listening_history_uid ON listening_history (uid);
    CREATE INDEX IF NOT EXISTS idx_listening_history_song_id ON listening_history (song_id);
    CREATE INDEX IF NOT EXISTS idx_listening_history_detected_at ON listening_history (detected_at DESC);

    ALTER TABLE listening_history ENABLE ROW LEVEL SECURITY;

    CREATE POLICY "Allow all operations on listening_history"
        ON listening_history
        FOR ALL
        USING (true)
        WITH CHECK (true);
    """

    try:
        supabase = get_supabase()

        # Try to query the table - if it fails, we need to create it
        try:
            result = supabase.table("listening_history").select("count").limit(1).execute()
            logger.info("✅ listening_history table already exists")
            return True
        except:
            logger.info("Creating listening_history table...")
            logger.info("Please run the following SQL in Supabase SQL Editor:")
            logger.info("\n" + "=" * 70)
            logger.info(sql)
            logger.info("=" * 70)
            logger.info("\nOr copy from: add_listening_history.sql")
            return False

    except Exception as e:
        logger.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    create_table()
