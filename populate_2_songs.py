#!/usr/bin/env python3
"""
Simple script to populate just the 2 working songs directly from local files
"""

import os
import sys

# Set environment variables BEFORE importing any modules
os.environ['SUPABASE_URL'] = 'https://cekkrqnleagiizmrwvgn.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNla2tycW5sZWFnaWl6bXJ3dmduIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTg3OTA0NCwiZXhwIjoyMDc3NDU1MDQ0fQ.6z2o72SjD9hz1q-ezhe__rPPAmaOmpko5GjkHIU8Cn8'

import logging
import time
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from fingerprinting import fingerprint_audio
from supabase_storage import store_song
from scipy.io import wavfile
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Song:
    title: str
    artist: str
    album: str
    filepath: Path

# Only the 2 working songs
SONGS = [
    Song(
        title="Cartoon On & On",
        artist="Cartoon ft. Daniel Levi",
        album="Single",
        filepath=Path("music_library/Cartoon_ft_Daniel_Levi_Cartoon_On__On.wav")
    ),
    Song(
        title="Heroes Tonight",
        artist="Janji ft. Johnning",
        album="Single",
        filepath=Path("music_library/Janji_ft_Johnning_Heroes_Tonight.wav")
    )
]

def load_audio(wav_file: Path) -> Tuple[np.ndarray, float]:
    """Load WAV file and return audio data + duration"""
    sample_rate, audio_data = wavfile.read(wav_file)

    # Convert to int16
    if audio_data.dtype != np.int16:
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)

    duration = len(audio_data) / sample_rate
    return audio_data, duration

def main():
    logger.info("="*70)
    logger.info("🎵 SHAZOMI - DIRECT DATABASE POPULATION")
    logger.info("="*70)
    logger.info(f"Processing {len(SONGS)} songs")
    logger.info("="*70)

    for song in SONGS:
        logger.info(f"\n🔍 Processing: {song.title} by {song.artist}")

        if not song.filepath.exists():
            logger.error(f"❌ File not found: {song.filepath}")
            continue

        try:
            # Load audio
            logger.info(f"📂 Loading: {song.filepath}")
            audio_data, duration = load_audio(song.filepath)
            logger.info(f"⏱️  Duration: {duration:.1f}s")

            # Generate fingerprints with unique identifier
            logger.info(f"🔍 Generating fingerprints...")
            start_time = time.time()
            identifier = str(song.filepath)  # Use file path as unique identifier
            hashes = fingerprint_audio(audio_data, identifier=identifier)
            processing_time = time.time() - start_time

            if not hashes:
                logger.error(f"❌ No fingerprints generated for {song.title}")
                continue

            logger.info(f"✅ Generated {len(hashes):,} fingerprints in {processing_time:.1f}s")

            # Store in database
            logger.info(f"💾 Storing in database...")
            song_info = (song.artist, song.album, song.title)
            store_song(hashes, song_info)
            logger.info(f"✅ Successfully stored {song.title}")

        except Exception as e:
            logger.error(f"❌ Error processing {song.title}: {e}", exc_info=True)

    logger.info("\n" + "="*70)
    logger.info("🎉 Database population complete!")
    logger.info("="*70)

if __name__ == "__main__":
    main()
