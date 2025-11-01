#!/usr/bin/env python3
"""
Test Supabase connection and verify credentials
"""

import os
from dotenv import load_dotenv
from supabase import create_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def test_connection():
    """Test Supabase connection"""

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("❌ SUPABASE_URL or SUPABASE_KEY not set in .env")
        return False

    logger.info(f"Testing connection to: {SUPABASE_URL}")

    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

        # Test connection by querying song_info
        result = supabase.table("song_info").select("count").limit(1).execute()

        logger.info("✅ Connection successful!")
        logger.info(f"Supabase client created successfully")

        # Check tables exist
        tables = ["song_info", "fingerprint_hash", "listening_history"]
        for table in tables:
            try:
                result = supabase.table(table).select("count").limit(1).execute()
                logger.info(f"✅ Table '{table}' exists and is accessible")
            except Exception as e:
                logger.error(f"❌ Table '{table}' error: {e}")

        return True

    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    if success:
        logger.info("\n🎉 Ready to populate database!")
    else:
        logger.error("\n⚠️  Fix connection issues before proceeding")
