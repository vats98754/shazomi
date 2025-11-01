#!/usr/bin/env python3
"""
Download and populate 1000 songs from YouTube (Creative Commons & No Copyright Music)
"""

import os
import sys

# Set environment variables BEFORE importing any modules
os.environ['SUPABASE_URL'] = 'https://cekkrqnleagiizmrwvgn.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNla2tycW5sZWFnaWl6bXJ3dmduIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTg3OTA0NCwiZXhwIjoyMDc3NDU1MDQ0fQ.6z2o72SjD9hz1q-ezhe__rPPAmaOmpko5GjkHIU8Cn8'

import logging
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List
from fingerprinting import fingerprint_audio
from supabase_storage import store_song, get_supabase
from scipy.io import wavfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler('populate_1000.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MUSIC_DIR = Path("music_library_1000")
MUSIC_DIR.mkdir(exist_ok=True)

@dataclass
class Song:
    title: str
    artist: str
    youtube_url: str
    album: str = "Single"

# Curated list of 1000+ no-copyright music channels and playlists
MUSIC_SOURCES = [
    # NoCopyrightSounds (NCS) - Electronic/Gaming music
    "https://www.youtube.com/playlist?list=PLRBp0Fe2GpgmsW46rJyudVFlY6IYjFBIK",  # NCS: The Best of 2023
    "https://www.youtube.com/playlist?list=PLRBp0Fe2GpgnIh0AiYKh7o7HnYAej-5ph",  # NCS: House
    "https://www.youtube.com/playlist?list=PLRBp0Fe2GpgkKMp7j2KmRNYh2fXRVeEoZ",  # NCS: Drumstep

    # Audio Library - No Copyright Music
    "https://www.youtube.com/playlist?list=PLzCxunOM5WFLXkbXRGLuxs_DFtA8TcSbK",  # Audio Library Mix
    "https://www.youtube.com/playlist?list=PLzCxunOM5WFKWHzDdH6km7U1f1d8eQ",     # Audio Library Electronic

    # Vlog No Copyright Music
    "https://www.youtube.com/playlist?list=PLzCxunOM5WFKX4RCpFkNmPZaTJWqwEYy-",  # Vlog Music

    # LAKEY INSPIRED
    "https://www.youtube.com/playlist?list=PLrPr4hSwJOeJEA_oR7Ol7hASQqT8OQwmE",  # LAKEY INSPIRED All Songs

    # Chillhop Music
    "https://www.youtube.com/playlist?list=PLOzDu-MXXLliO9fBNZOQTBDddoA3FzZUo",  # Chillhop Essentials
]

def download_playlist_info(playlist_url: str) -> List[dict]:
    """Download playlist metadata without downloading videos"""
    try:
        cmd = [
            'yt-dlp',
            '--flat-playlist',
            '--dump-json',
            '--playlist-end', '200',  # Limit to 200 per playlist
            playlist_url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"Failed to get playlist info: {result.stderr}")
            return []

        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    video = json.loads(line)
                    videos.append({
                        'title': video.get('title', 'Unknown'),
                        'url': f"https://www.youtube.com/watch?v={video['id']}",
                        'uploader': video.get('uploader', 'Unknown'),
                    })
                except json.JSONDecodeError:
                    continue

        return videos
    except Exception as e:
        logger.error(f"Error getting playlist info: {e}")
        return []

def download_song(video_info: dict, output_dir: Path) -> Path:
    """Download a single song from YouTube and convert to WAV"""
    try:
        title = video_info['title']
        url = video_info['url']
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')[:100]  # Limit filename length

        output_file = output_dir / f"{safe_title}.wav"

        # Skip if already exists
        if output_file.exists():
            logger.info(f"✓ Already downloaded: {safe_title}")
            return output_file

        logger.info(f"⬇️  Downloading: {title}")

        # Download and convert to 16kHz mono WAV in one step
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '--postprocessor-args', 'ffmpeg:-ar 16000 -ac 1',  # 16kHz mono
            '-o', str(output_dir / f"{safe_title}.%(ext)s"),
            '--max-filesize', '50M',  # Skip very large files
            '--match-filter', 'duration < 600',  # Skip videos longer than 10 minutes
            url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0 and output_file.exists():
            return output_file
        else:
            logger.error(f"Download failed for {title}: {result.stderr[:200]}")
            return None

    except Exception as e:
        logger.error(f"Error downloading {video_info.get('title', 'unknown')}: {e}")
        return None

def load_audio(wav_file: Path):
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

def process_song(wav_file: Path, artist: str = "Various Artists") -> bool:
    """Process a single song: fingerprint and store"""
    try:
        title = wav_file.stem.replace('_', ' ')

        logger.info(f"🔍 Processing: {title}")

        # Load audio
        audio_data, duration = load_audio(wav_file)

        if duration < 30:  # Skip very short songs
            logger.warning(f"⏭️  Skipping {title} - too short ({duration:.1f}s)")
            return False

        logger.info(f"⏱️  Duration: {duration:.1f}s")

        # Generate fingerprints
        start_time = time.time()
        identifier = str(wav_file)
        hashes = fingerprint_audio(audio_data, identifier=identifier)
        processing_time = time.time() - start_time

        if not hashes:
            logger.error(f"❌ No fingerprints generated for {title}")
            return False

        logger.info(f"✅ Generated {len(hashes):,} fingerprints in {processing_time:.1f}s")

        # Store in database
        song_info = (artist, "Single", title)
        store_song(hashes, song_info)
        logger.info(f"💾 Stored {title}")

        return True

    except Exception as e:
        logger.error(f"❌ Error processing {wav_file.name}: {e}", exc_info=True)
        return False

def main():
    logger.info("="*70)
    logger.info("🎵 SHAZOMI - 1000 SONG DATABASE POPULATION")
    logger.info("="*70)

    # Test Supabase connection
    logger.info("Testing Supabase connection...")
    try:
        client = get_supabase()
        result = client.table('song_info').select('count', count='exact').limit(1).execute()
        logger.info(f"✅ Supabase connected - Current songs: {result.count}")
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return

    # Check yt-dlp
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        logger.info("✅ yt-dlp available")
    except:
        logger.error("❌ yt-dlp not found. Install with: brew install yt-dlp")
        return

    # Step 1: Collect video info from playlists
    logger.info("\n📥 STEP 1: Collecting song information from playlists...")
    all_videos = []

    for playlist_url in MUSIC_SOURCES:
        logger.info(f"📋 Processing playlist: {playlist_url}")
        videos = download_playlist_info(playlist_url)
        all_videos.extend(videos)
        logger.info(f"   Found {len(videos)} songs")

        if len(all_videos) >= 1000:
            break

    # Deduplicate by title
    seen_titles = set()
    unique_videos = []
    for video in all_videos:
        title_lower = video['title'].lower()
        if title_lower not in seen_titles:
            seen_titles.add(title_lower)
            unique_videos.append(video)

    logger.info(f"\n✅ Collected {len(unique_videos)} unique songs")
    target_count = min(1000, len(unique_videos))
    videos_to_download = unique_videos[:target_count]

    # Step 2: Download songs in parallel
    logger.info(f"\n📥 STEP 2: Downloading {target_count} songs (parallel)...")
    downloaded_files = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_video = {
            executor.submit(download_song, video, MUSIC_DIR): video
            for video in videos_to_download
        }

        for i, future in enumerate(as_completed(future_to_video), 1):
            video = future_to_video[future]
            try:
                wav_file = future.result()
                if wav_file:
                    downloaded_files.append(wav_file)
                logger.info(f"Progress: {i}/{target_count} downloaded ({len(downloaded_files)} successful)")
            except Exception as e:
                logger.error(f"Error downloading {video['title']}: {e}")

    logger.info(f"\n✅ Downloaded {len(downloaded_files)}/{target_count} songs")

    # Step 3: Process and store fingerprints
    logger.info(f"\n🔍 STEP 3: Processing fingerprints...")

    successful = 0
    for i, wav_file in enumerate(downloaded_files, 1):
        logger.info(f"\n[{i}/{len(downloaded_files)}]")
        if process_song(wav_file):
            successful += 1

        # Log progress every 10 songs
        if i % 10 == 0:
            logger.info(f"📊 Progress: {successful}/{i} songs processed successfully")

    logger.info("\n" + "="*70)
    logger.info(f"🎉 DATABASE POPULATION COMPLETE!")
    logger.info(f"✅ Successfully processed: {successful}/{len(downloaded_files)} songs")
    logger.info("="*70)

if __name__ == "__main__":
    main()
