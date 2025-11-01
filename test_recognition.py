#!/usr/bin/env python3
"""
Test recognition by extracting a clip from Cartoon On & On and matching it
"""

import os
import sys

# Set environment variables BEFORE importing any modules
os.environ['SUPABASE_URL'] = 'https://cekkrqnleagiizmrwvgn.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNla2tycW5sZWFnaWl6bXJ3dmduIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTg3OTA0NCwiZXhwIjoyMDc3NDU1MDQ0fQ.6z2o72SjD9hz1q-ezhe__rPPAmaOmpko5GjkHIU8Cn8'

import logging
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from fingerprinting import fingerprint_audio, best_match
from supabase_storage import get_matches, get_info_for_song_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def extract_clip(wav_file: Path, start_sec: float = 30, duration_sec: float = 10) -> np.ndarray:
    """Extract a clip from the audio file"""
    sample_rate, audio_data = wavfile.read(wav_file)

    # Convert to int16
    if audio_data.dtype != np.int16:
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)

    start_sample = int(start_sec * sample_rate)
    end_sample = int((start_sec + duration_sec) * sample_rate)

    clip = audio_data[start_sample:end_sample]
    logger.info(f"Extracted {duration_sec}s clip from {start_sec}s mark ({len(clip)} samples)")

    return clip


def main():
    logger.info("="*70)
    logger.info("🎵 SHAZOMI - RECOGNITION TEST")
    logger.info("="*70)

    # Test file
    test_file = Path("music_library/Cartoon_ft_Daniel_Levi_Cartoon_On__On.wav")

    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return

    # Extract a 10-second clip from 30 seconds into the song
    logger.info(f"\n🎧 Extracting test clip from: {test_file.name}")
    clip = extract_clip(test_file, start_sec=30, duration_sec=10)

    # Generate fingerprints for the unknown audio clip
    # Use "recorded" as identifier since we don't know what song this is
    logger.info(f"🔍 Generating fingerprints for unknown audio...")
    hashes = fingerprint_audio(clip, identifier="recorded")
    logger.info(f"✅ Generated {len(hashes):,} fingerprints from clip")

    # Match against database using threshold=1 (any match counts)
    logger.info(f"\n🔍 Searching database for matches...")
    matches = get_matches(hashes, threshold=1)
    logger.info(f"📊 Raw matches found in {len(matches)} song(s)")

    # Debug: Show match counts
    for song_id, match_list in matches.items():
        logger.info(f"  Song {song_id}: {len(match_list)} raw matches")

    if not matches:
        logger.error("❌ No matches found!")
        return

    # Use the robust best_match algorithm from fingerprinting.py
    logger.info(f"\n🎯 Scoring matches using Shazam algorithm...")
    best_song_id, best_score = best_match(matches)

    if not best_song_id:
        logger.error("❌ No strong match found!")
        return

    # Display results
    logger.info(f"\n{'='*70}")
    logger.info(f"🎵 RECOGNITION RESULTS:")
    logger.info(f"{'='*70}")

    info = get_info_for_song_id(best_song_id)
    if info:
        artist, album, title = info
        logger.info(f"\n🎵 Identified Song: {title}")
        logger.info(f"   👤 Artist: {artist}")
        logger.info(f"   💿 Album: {album}")
        logger.info(f"   📊 Match Score: {best_score}")

        # Score interpretation based on typical Shazam values
        if best_score > 100:
            confidence = "🟢 VERY HIGH"
        elif best_score > 50:
            confidence = "🟢 HIGH"
        elif best_score > 20:
            confidence = "🟡 MEDIUM"
        else:
            confidence = "🔴 LOW"
        logger.info(f"   ✓ Confidence: {confidence}")

        # Check if we got the right song
        if title == "Cartoon On & On":
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ SUCCESS! Correctly identified the song!")
            logger.info(f"{'='*70}")
        else:
            logger.warning(f"\n{'='*70}")
            logger.warning(f"⚠️  Unexpected result - expected 'Cartoon On & On'")
            logger.warning(f"{'='*70}")
    else:
        logger.error(f"❌ Could not retrieve info for song_id: {best_song_id}")

if __name__ == "__main__":
    main()
