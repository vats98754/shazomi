#!/usr/bin/env python3
"""
Test if fingerprinting is consistent
"""

import numpy as np
from pathlib import Path
from scipy.io import wavfile
from fingerprinting import fingerprint_audio
import logging

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
    return clip

def main():
    test_file = Path("music_library/Cartoon_ft_Daniel_Levi_Cartoon_On__On.wav")

    logger.info("Extracting 10-second clip from 30s mark...")
    clip = extract_clip(test_file, start_sec=30, duration_sec=10)

    # Fingerprint the same clip twice
    logger.info("\n=== First fingerprinting ===")
    hashes1 = fingerprint_audio(clip, identifier="test1")
    logger.info(f"Generated {len(hashes1):,} hashes")

    logger.info("\n=== Second fingerprinting ===")
    hashes2 = fingerprint_audio(clip, identifier="test2")
    logger.info(f"Generated {len(hashes2):,} hashes")

    # Extract just the hash values (without song_id)
    hash_values1 = set([h[0] for h in hashes1])
    hash_values2 = set([h[0] for h in hashes2])

    logger.info(f"\n=== Comparison ===")
    logger.info(f"Unique hashes in run 1: {len(hash_values1)}")
    logger.info(f"Unique hashes in run 2: {len(hash_values2)}")
    logger.info(f"Hashes in common: {len(hash_values1 & hash_values2)}")

    if hash_values1 == hash_values2:
        logger.info("✅ Fingerprinting is perfectly consistent!")
    else:
        logger.error("❌ Fingerprinting is NOT consistent!")
        logger.error(f"   Hashes only in run 1: {len(hash_values1 - hash_values2)}")
        logger.error(f"   Hashes only in run 2: {len(hash_values2 - hash_values1)}")

    # Now compare clip fingerprints to full song fingerprints
    logger.info("\n=== Comparing clip to full song ===")
    logger.info("Loading full song...")
    sample_rate, full_audio = wavfile.read(test_file)
    if full_audio.dtype != np.int16:
        if full_audio.dtype == np.float32 or full_audio.dtype == np.float64:
            full_audio = (full_audio * 32767).astype(np.int16)
        else:
            full_audio = full_audio.astype(np.int16)

    logger.info("Fingerprinting full song...")
    full_hashes = fingerprint_audio(full_audio, identifier="full_song")
    full_hash_values = set([h[0] for h in full_hashes])

    logger.info(f"Full song unique hashes: {len(full_hash_values)}")
    logger.info(f"Clip hashes that match full song: {len(hash_values1 & full_hash_values)}")
    logger.info(f"Match rate: {100 * len(hash_values1 & full_hash_values) / len(hash_values1):.1f}%")

    if len(hash_values1 & full_hash_values) / len(hash_values1) < 0.5:
        logger.error("❌ Match rate is too low! Fingerprinting is not robust.")
    else:
        logger.info("✅ Match rate is reasonable.")

if __name__ == "__main__":
    main()
