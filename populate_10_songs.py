#!/usr/bin/env python3
"""
High-Performance Database Population for 10 Songs
Downloads, processes, and stores songs with maximum parallelization
Based on abracadabra architecture: https://github.com/notexactlyawe/abracadabra
"""

import os
import sys
import time
import logging
import subprocess
import uuid
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv

# Import fingerprinting modules
from fingerprinting import fingerprint_audio, best_match
from supabase_storage import store_song, song_in_db, get_supabase

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Song:
    """Song metadata"""
    url: str
    title: str
    artist: str
    album: str
    duration_estimate: int  # seconds


# 10 Creative Commons licensed songs from various sources
SONG_LIBRARY = [
    Song(
        url="https://www.youtube.com/watch?v=_tV5LEBDs7w",
        title="Arpent",
        artist="Moby",
        album="Long Ambients",
        duration_estimate=180
    ),
    Song(
        url="https://www.youtube.com/watch?v=SbZRLC8d5Uw",
        title="Ambient Piano Amphibian",
        artist="Nomyn",
        album="YouTube Audio Library",
        duration_estimate=200
    ),
    Song(
        url="https://www.youtube.com/watch?v=hKXu_Ozf8B0",
        title="Jazzy Abstract Beat",
        artist="Bensound",
        album="YouTube Audio Library",
        duration_estimate=180
    ),
    Song(
        url="https://www.youtube.com/watch?v=cPCLFtxpadE",
        title="Chill Day",
        artist="LAKEY INSPIRED",
        album="YouTube Audio Library",
        duration_estimate=120
    ),
    Song(
        url="https://www.youtube.com/watch?v=NAkP8OZ6bFM",
        title="Better Days",
        artist="LAKEY INSPIRED",
        album="YouTube Audio Library",
        duration_estimate=140
    ),
    Song(
        url="https://www.youtube.com/watch?v=YySj-IvGZy8",
        title="Aesthetic",
        artist="Tobu",
        album="NCS Release",
        duration_estimate=180
    ),
    Song(
        url="https://www.youtube.com/watch?v=IvyPv-bFklM",
        title="Elektronomia Sky High",
        artist="Elektronomia",
        album="NCS Release",
        duration_estimate=210
    ),
    Song(
        url="https://www.youtube.com/watch?v=K4DyBUG242c",
        title="Cartoon On & On",
        artist="Cartoon ft Daniel Levi",
        album="NCS Release",
        duration_estimate=195
    ),
    Song(
        url="https://www.youtube.com/watch?v=bM7SZ5SBzyY",
        title="Faded",
        artist="Alan Walker",
        album="NCS Release",
        duration_estimate=210
    ),
    Song(
        url="https://www.youtube.com/watch?v=60ItHLz5WEA",
        title="Heroes Tonight",
        artist="Janji ft Johnning",
        album="NCS Release",
        duration_estimate=185
    ),
]


class FastDownloader:
    """Parallel song downloader using yt-dlp"""

    def __init__(self, output_dir: str = "music_library"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def check_ytdlp(self) -> bool:
        """Check if yt-dlp is installed"""
        try:
            subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("yt-dlp not found. Install with: pip install yt-dlp")
            return False

    def download_song(self, song: Song) -> Optional[Path]:
        """Download a single song"""
        if not self.check_ytdlp():
            return None

        try:
            # Create safe filename
            safe_name = f"{song.artist}_{song.title}".replace(" ", "_").replace("/", "_")
            safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
            output_template = str(self.output_dir / f"{safe_name}.%(ext)s")

            # Check if already downloaded
            expected_file = self.output_dir / f"{safe_name}.mp3"
            if expected_file.exists():
                logger.info(f"✓ Already downloaded: {song.title}")
                return expected_file

            logger.info(f"⬇️  Downloading: {song.title} by {song.artist}")

            # Download with yt-dlp
            cmd = [
                'yt-dlp',
                '-x',  # Extract audio
                '--audio-format', 'mp3',
                '--audio-quality', '0',
                '-o', output_template,
                '--no-playlist',
                '--quiet',
                '--no-warnings',
                song.url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error(f"Download failed for {song.title}: {result.stderr}")
                return None

            if expected_file.exists():
                logger.info(f"✅ Downloaded: {song.title}")
                return expected_file

            return None

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout downloading {song.title}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {song.title}: {e}")
            return None

    def download_all_parallel(self, songs: List[Song], max_workers: int = 4) -> List[Tuple[Song, Path]]:
        """Download all songs in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_song = {executor.submit(self.download_song, song): song for song in songs}

            for future in as_completed(future_to_song):
                song = future_to_song[future]
                try:
                    filepath = future.result()
                    if filepath:
                        results.append((song, filepath))
                except Exception as e:
                    logger.error(f"Error in download thread for {song.title}: {e}")

        return results


class FastAudioProcessor:
    """Optimized audio processing"""

    SAMPLE_RATE = 16000

    @staticmethod
    def convert_to_wav(input_file: Path, song: Song) -> Optional[Path]:
        """Convert audio to WAV using ffmpeg"""
        try:
            output_file = input_file.with_suffix('.wav')

            if output_file.exists():
                return output_file

            cmd = [
                'ffmpeg', '-i', str(input_file),
                '-ar', str(FastAudioProcessor.SAMPLE_RATE),
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                '-loglevel', 'error',
                str(output_file)
            ]

            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            return output_file

        except Exception as e:
            logger.error(f"Error converting {input_file}: {e}")
            return None

    @staticmethod
    def load_audio(wav_file: Path) -> Optional[np.ndarray]:
        """Load WAV file"""
        try:
            from scipy.io import wavfile
            sample_rate, audio_data = wavfile.read(wav_file)

            # Convert to int16
            if audio_data.dtype != np.int16:
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)

            return audio_data

        except Exception as e:
            logger.error(f"Error loading {wav_file}: {e}")
            return None


class ParallelFingerprintProcessor:
    """Process fingerprints using process pool for CPU-bound work"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def process_single_song(self, song: Song, filepath: Path) -> Optional[Tuple[List, Song, float]]:
        """Process a single song and return fingerprints"""
        try:
            logger.info(f"🔍 Processing: {song.title}")

            # Check if already in DB
            if song_in_db(str(filepath)):
                logger.info(f"⏭️  Skipping (already in DB): {song.title}")
                return None

            # Convert to WAV
            wav_file = FastAudioProcessor.convert_to_wav(filepath, song)
            if not wav_file:
                return None

            # Load audio
            audio_data = FastAudioProcessor.load_audio(wav_file)
            if audio_data is None:
                return None

            duration = len(audio_data) / FastAudioProcessor.SAMPLE_RATE

            # Generate fingerprints (this is CPU-intensive)
            start_time = time.time()
            hashes = fingerprint_audio(audio_data)
            processing_time = time.time() - start_time

            if not hashes:
                logger.error(f"No fingerprints for {song.title}")
                return None

            logger.info(f"✅ Generated {len(hashes):,} fingerprints in {processing_time:.1f}s - {song.title}")

            return (hashes, song, duration)

        except Exception as e:
            logger.error(f"Error processing {song.title}: {e}", exc_info=True)
            return None

    def process_all_parallel(self, song_files: List[Tuple[Song, Path]]) -> List[Tuple[List, Song, float]]:
        """Process all songs in parallel using threads (simpler than processes)"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_song = {
                executor.submit(self.process_single_song, song, filepath): song
                for song, filepath in song_files
            }

            for future in as_completed(future_to_song):
                song = future_to_song[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in processing thread for {song.title}: {e}")

        return results


class DatabaseInserter:
    """Fast database insertion with batching and retry logic"""

    def __init__(self):
        self.supabase = get_supabase()

    def store_song(self, hashes: List, song: Song, duration: float, max_retries: int = 3) -> bool:
        """Store song and its fingerprints with retry logic"""

        for attempt in range(max_retries):
            try:
                logger.info(f"💾 Storing: {song.title} ({len(hashes):,} fingerprints) [Attempt {attempt + 1}/{max_retries}]")

                # Prepare song info
                song_info = (song.artist, song.album, song.title)

                # Store using existing function (already has batch logic)
                store_song(hashes, song_info)

                # Update duration if possible
                song_id = hashes[0][2]
                try:
                    self.supabase.table("song_info").update({
                        "duration_seconds": duration
                    }).eq("song_id", song_id).execute()
                    logger.info(f"   ✓ Updated duration: {duration:.1f}s")
                except Exception as e:
                    logger.warning(f"   ⚠ Could not update duration: {e}")

                logger.info(f"✅ Stored: {song.title}")
                return True

            except Exception as e:
                logger.error(f"   ❌ Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.info(f"   ⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to store {song.title} after {max_retries} attempts")
                    return False

        return False

    def store_all(self, processed: List[Tuple[List, Song, float]]) -> Tuple[int, List[str]]:
        """Store all processed songs and return success count + failed songs"""
        success_count = 0
        failed_songs = []

        for idx, (hashes, song, duration) in enumerate(processed, 1):
            logger.info(f"\n[{idx}/{len(processed)}] Processing: {song.title}")

            if self.store_song(hashes, song, duration):
                success_count += 1
            else:
                failed_songs.append(song.title)

        return success_count, failed_songs


def main():
    """Main execution"""
    logger.info("=" * 70)
    logger.info("🎵 SHAZOMI - HIGH-PERFORMANCE DATABASE POPULATION")
    logger.info("=" * 70)
    logger.info(f"Target: {len(SONG_LIBRARY)} songs")
    logger.info(f"Using abracadabra algorithm")
    logger.info("=" * 70)

    # Pre-flight checks
    logger.info("\n🔍 Pre-flight checks...")

    # Check Supabase credentials
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("❌ SUPABASE_URL or SUPABASE_KEY not set in .env")
        logger.error("Run: python get_supabase_key.py for instructions")
        return

    if len(SUPABASE_KEY) < 100:
        logger.error(f"❌ SUPABASE_KEY is too short ({len(SUPABASE_KEY)} chars)")
        logger.error("   Expected: ~200-300 characters starting with 'eyJ...'")
        logger.error(f"   Got: {SUPABASE_KEY[:50]}...")
        logger.error("Run: python get_supabase_key.py for instructions")
        return

    # Test connection
    logger.info("Testing Supabase connection...")
    try:
        supabase = get_supabase()
        result = supabase.table("song_info").select("count").limit(1).execute()
        logger.info("✅ Supabase connection successful")
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        logger.error("Run: python get_supabase_key.py for instructions")
        logger.error("Or: python test_supabase_connection.py to diagnose")
        return

    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.info("✅ ffmpeg available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ ffmpeg not found. Install with: brew install ffmpeg")
        return

    logger.info("✅ All pre-flight checks passed\n")

    start_time = time.time()

    # Step 1: Download songs (parallel)
    logger.info("\n📥 STEP 1: Downloading songs (parallel)...")
    downloader = FastDownloader()

    # Check yt-dlp first
    if not downloader.check_ytdlp():
        logger.error("Please install yt-dlp: pip install yt-dlp")
        return

    download_start = time.time()
    downloaded = downloader.download_all_parallel(SONG_LIBRARY, max_workers=3)
    download_time = time.time() - download_start

    logger.info(f"\n✅ Downloaded {len(downloaded)}/{len(SONG_LIBRARY)} songs in {download_time:.1f}s")

    if not downloaded:
        logger.error("No songs downloaded. Exiting.")
        return

    # Step 2: Process fingerprints (parallel)
    logger.info(f"\n🔍 STEP 2: Processing fingerprints (parallel, {os.cpu_count()} cores)...")
    processor = ParallelFingerprintProcessor(max_workers=os.cpu_count())

    process_start = time.time()
    processed = processor.process_all_parallel(downloaded)
    process_time = time.time() - process_start

    total_fingerprints = sum(len(hashes) for hashes, _, _ in processed)
    logger.info(f"\n✅ Processed {len(processed)} songs → {total_fingerprints:,} fingerprints in {process_time:.1f}s")

    if not processed:
        logger.error("No songs processed. Exiting.")
        return

    # Step 3: Store in database
    logger.info(f"\n💾 STEP 3: Storing in Supabase...")
    inserter = DatabaseInserter()

    store_start = time.time()
    success_count, failed_songs = inserter.store_all(processed)
    store_time = time.time() - store_start

    # Summary
    total_time = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info("📊 SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Downloads:      {len(downloaded)}/{len(SONG_LIBRARY)} songs ({download_time:.1f}s)")
    logger.info(f"  Processing:     {len(processed)} songs ({process_time:.1f}s)")
    logger.info(f"  Fingerprints:   {total_fingerprints:,} total")
    logger.info(f"  Stored:         {success_count}/{len(processed)} songs ({store_time:.1f}s)")
    logger.info(f"  Failed:         {len(failed_songs)} songs")
    logger.info(f"  Total time:     {total_time:.1f}s")
    logger.info(f"  Avg per song:   {total_time/max(success_count, 1):.1f}s")
    logger.info("=" * 70)

    if failed_songs:
        logger.warning("\n⚠️  Failed to store these songs:")
        for song in failed_songs:
            logger.warning(f"  - {song}")

    if success_count > 0:
        logger.info("\n🎉 SUCCESS! Database populated with songs")
        logger.info(f"   {success_count} songs with {total_fingerprints:,} fingerprints")
        logger.info("\nTest recognition:")
        logger.info("  1. Convert audio: ffmpeg -i music_library/Moby_Arpent.mp3 -f s16le -ar 16000 -ac 1 test.raw")
        logger.info("  2. Test: curl -X POST 'http://localhost:8000/webhook/audio-chunk?uid=test' --data-binary @test.raw")
        logger.info("\nView in Supabase:")
        logger.info("  SELECT title, COUNT(*) FROM fingerprint_hash")
        logger.info("  JOIN song_info USING(song_id) GROUP BY title;")
    else:
        logger.error("\n⚠️  No songs stored. Check the errors above.")


if __name__ == "__main__":
    main()
