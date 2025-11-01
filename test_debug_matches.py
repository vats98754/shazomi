#!/usr/bin/env python3
"""
Debug test to show ALL match scores
"""

import os
os.environ['SUPABASE_URL'] = 'https://cekkrqnleagiizmrwvgn.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNla2tycW5sZWFnaWl6bXJ3dmduIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTg3OTA0NCwiZXhwIjoyMDc3NDU1MDQ0fQ.6z2o72SjD9hz1q-ezhe__rPPAmaOmpko5GjkHIU8Cn8'

import logging
import sys
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from fingerprinting import fingerprint_audio, score_match
from supabase_storage import get_matches, get_info_for_song_id, get_supabase
import subprocess
import tempfile

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_audio_file(audio_file: Path) -> tuple:
    """Load any audio file and return audio data + sample rate"""
    file_ext = audio_file.suffix.lower()

    if file_ext == '.wav':
        sample_rate, audio_data = wavfile.read(audio_file)
    else:
        logger.info(f"Converting {file_ext} to WAV using ffmpeg...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        try:
            cmd = ['ffmpeg', '-i', str(audio_file), '-ar', '16000', '-ac', '1', '-y', tmp_wav_path]
            subprocess.run(cmd, capture_output=True, timeout=30)
            sample_rate, audio_data = wavfile.read(tmp_wav_path)
        finally:
            Path(tmp_wav_path).unlink(missing_ok=True)

    # Convert to int16
    if audio_data.dtype != np.int16:
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)

    return audio_data, sample_rate

def extract_clip(audio_file: Path, start_sec: float = 30, duration_sec: float = 10):
    """Extract a clip from audio file"""
    audio_data, sample_rate = load_audio_file(audio_file)

    start_sample = int(start_sec * sample_rate)
    end_sample = int((start_sec + duration_sec) * sample_rate)

    clip = audio_data[start_sample:end_sample]
    logger.info(f"Extracted {duration_sec}s clip from {start_sec}s mark ({len(clip)} samples)")

    return clip

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python test_debug_matches.py <song_file> [start_sec] [duration_sec]")
        logger.info("Example: python test_debug_matches.py music_library/song.mp3 30 10")
        sys.exit(1)

    test_file = Path(sys.argv[1])
    start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 10

    print("="*80)
    print("🔍 SHAZOMI - DEBUG MATCH ANALYSIS")
    print("="*80)
    print(f"Test file: {test_file}")
    print(f"Clip: {start_sec}s - {start_sec + duration_sec}s ({duration_sec}s)")
    print()

    if not test_file.exists():
        logger.error(f"File not found: {test_file}")
        return

    # Extract clip
    clip = extract_clip(test_file, start_sec=start_sec, duration_sec=duration_sec)

    # Generate fingerprints
    print(f"🔍 Generating fingerprints...")
    hashes = fingerprint_audio(clip, identifier="recorded")
    print(f"✅ Generated {len(hashes):,} fingerprints")
    print()

    # Match against database
    print(f"🔍 Searching database...")
    matches = get_matches(hashes, threshold=1)
    print(f"📊 Found matches in {len(matches)} song(s)")
    print()

    if not matches:
        print("❌ No matches found!")
        return

    # Score ALL matches
    print("="*80)
    print("🎯 ALL MATCH SCORES (sorted by score):")
    print("="*80)

    scored_matches = []
    for song_id, offsets in matches.items():
        score = score_match(offsets)
        scored_matches.append((song_id, score, len(offsets)))

    # Sort by score descending
    scored_matches.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Score':<10} {'Raw Matches':<15} {'Song Title'}")
    print("-"*80)

    for i, (song_id, score, raw_count) in enumerate(scored_matches[:20], 1):
        info = get_info_for_song_id(song_id)
        if info:
            artist, album, title = info
            song_name = f"{title} - {artist}"
        else:
            song_name = song_id

        print(f"{i:<6} {score:<10} {raw_count:<15} {song_name}")

    if len(scored_matches) > 20:
        print(f"... and {len(scored_matches) - 20} more matches")

    print()
    print("="*80)
    print("🏆 BEST MATCH:")
    print("="*80)

    best_song_id, best_score, best_raw = scored_matches[0]
    info = get_info_for_song_id(best_song_id)
    if info:
        artist, album, title = info
        print(f"🎵 Song: {title}")
        print(f"👤 Artist: {artist}")
        print(f"💿 Album: {album}")
        print(f"📊 Score: {best_score}")
        print(f"🔢 Raw Matches: {best_raw}")

        # Show histogram analysis
        offsets = matches[best_song_id]
        tks = [x[0] - x[1] for x in offsets]
        print(f"\n📈 Time Delta Analysis:")
        print(f"   Min delta: {min(tks):.2f}s")
        print(f"   Max delta: {max(tks):.2f}s")
        print(f"   Range: {max(tks) - min(tks):.2f}s")

        # Show top 5 time deltas
        from collections import Counter
        delta_counts = Counter([round(tk, 1) for tk in tks])
        print(f"\n📊 Top Time Delta Bins (0.5s bins):")
        for delta, count in delta_counts.most_common(5):
            print(f"   Δ={delta:6.1f}s: {count:4d} matches")

if __name__ == "__main__":
    main()
