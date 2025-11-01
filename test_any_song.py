#!/usr/bin/env python3
"""
Test recognition with ANY song - not hardcoded
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
import subprocess
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def load_audio_file(audio_file: Path) -> tuple[np.ndarray, int]:
    """Load any audio file (WAV, MP3, etc.) and return audio data + sample rate"""
    file_ext = audio_file.suffix.lower()

    if file_ext == '.wav':
        # Direct WAV loading
        sample_rate, audio_data = wavfile.read(audio_file)
    else:
        # Use ffmpeg to convert to WAV
        logger.info(f"Converting {file_ext} to WAV using ffmpeg...")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        try:
            # Convert to 16kHz mono WAV
            cmd = [
                'ffmpeg',
                '-i', str(audio_file),
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # Mono
                '-y',            # Overwrite
                tmp_wav_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                raise Exception(f"ffmpeg conversion failed: {result.stderr}")

            # Load the converted WAV
            sample_rate, audio_data = wavfile.read(tmp_wav_path)

        finally:
            # Clean up temp file
            Path(tmp_wav_path).unlink(missing_ok=True)

    # Convert to int16
    if audio_data.dtype != np.int16:
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)

    return audio_data, sample_rate

def extract_clip(audio_file: Path, start_sec: float = 30, duration_sec: float = 10) -> np.ndarray:
    """Extract a clip from any audio file (WAV, MP3, etc.)"""
    audio_data, sample_rate = load_audio_file(audio_file)

    start_sample = int(start_sec * sample_rate)
    end_sample = int((start_sec + duration_sec) * sample_rate)

    clip = audio_data[start_sample:end_sample]
    logger.info(f"Extracted {duration_sec}s clip from {start_sec}s mark ({len(clip)} samples)")

    return clip

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python test_any_song.py <song_file> [start_sec] [duration_sec]")
        logger.info("Supports: WAV, MP3, M4A, FLAC, OGG, and other audio formats")
        logger.info("Example: python test_any_song.py music_library/song.mp3 30 10")
        sys.exit(1)

    test_file = Path(sys.argv[1])
    start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 10

    logger.info("="*70)
    logger.info("🎵 SHAZOMI - RECOGNITION TEST (ANY SONG)")
    logger.info("="*70)
    logger.info(f"Test file: {test_file}")
    logger.info(f"Expected song: {test_file.stem.replace('_', ' ')}")

    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        return

    # Extract clip
    logger.info(f"\n🎧 Extracting {duration_sec}s clip from {start_sec}s mark...")
    clip = extract_clip(test_file, start_sec=start_sec, duration_sec=duration_sec)

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
        info = get_info_for_song_id(song_id)
        song_name = f"{info[2]} by {info[0]}" if info else song_id
        logger.info(f"  {song_name}: {len(match_list)} raw matches")

    if not matches:
        logger.error("❌ No matches found!")
        return

    # Use the robust best_match algorithm from fingerprinting.py
    logger.info(f"\n🎯 Scoring matches using histogram-based algorithm...")
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

        # Check if we got the right song (fuzzy match on filename)
        expected_title = test_file.stem.replace('_', ' ')
        if title.lower() in expected_title.lower() or expected_title.lower() in title.lower():
            logger.info(f"\n{'='*70}")
            logger.info(f"✅ SUCCESS! Correctly identified the song!")
            logger.info(f"{'='*70}")
        else:
            logger.warning(f"\n{'='*70}")
            logger.warning(f"⚠️  Unexpected result - expected '{expected_title}'")
            logger.warning(f"   Got: '{title}'")
            logger.warning(f"{'='*70}")
    else:
        logger.error(f"❌ Could not retrieve info for song_id: {best_song_id}")

if __name__ == "__main__":
    main()
