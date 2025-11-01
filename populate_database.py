#!/usr/bin/env python3
"""
Database Population Script for Shazomi
Downloads songs from open datasets and generates fingerprints in parallel
"""

import os
import sys
import uuid
import logging
import time
import requests
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np

# Import our fingerprinting modules
from fingerprinting import fingerprint_audio, SAMPLE_RATE
from supabase_storage import store_song, song_in_db, get_supabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SongMetadata:
    """Metadata for a song"""
    file_path: str
    artist: str
    album: str
    title: str
    duration: float = 0.0


class MusicDownloader:
    """Download and manage music from open datasets"""

    def __init__(self, download_dir: str = "music_library"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

    def scan_local_directory(self, directory: Optional[str] = None) -> List[SongMetadata]:
        """
        Scan local directory for audio files

        Args:
            directory: Directory to scan (defaults to self.download_dir)

        Returns:
            List of SongMetadata for found audio files
        """
        scan_dir = Path(directory) if directory else self.download_dir

        if not scan_dir.exists():
            logger.warning(f"Directory does not exist: {scan_dir}")
            return []

        # Supported audio formats
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus'}

        songs = []
        for filepath in scan_dir.rglob('*'):
            if filepath.suffix.lower() in audio_extensions:
                # Try to parse filename for metadata
                # Format: Artist_Title.ext or just Title.ext
                filename = filepath.stem

                if '_' in filename:
                    parts = filename.split('_', 1)
                    artist = parts[0].strip()
                    title = parts[1].strip()
                else:
                    artist = "Unknown Artist"
                    title = filename.strip()

                metadata = SongMetadata(
                    file_path=str(filepath),
                    artist=artist,
                    album="Unknown Album",
                    title=title
                )
                songs.append(metadata)
                logger.info(f"Found: {title} by {artist}")

        logger.info(f"Found {len(songs)} audio files in {scan_dir}")
        return songs

    def download_file(self, url: str, filename: str) -> Optional[Path]:
        """Download a file from URL"""
        try:
            filepath = self.download_dir / filename
            if filepath.exists():
                logger.info(f"File already exists: {filename}")
                return filepath

            logger.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded {filename}")
            return filepath

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

    def get_free_music_archive_samples(self) -> List[Dict]:
        """
        Get sample songs from Free Music Archive (FMA)
        Using direct links to CC-licensed music
        """
        # Sample songs from Free Music Archive (small subset for testing)
        # These are direct links to CC0/CC-BY licensed music
        samples = [
            {
                "url": "https://freemusicarchive.org/track/07_-_Quantum_Jazz/download",
                "artist": "Kevin MacLeod",
                "title": "Quantum Jazz",
                "album": "Jazz & Blues",
                "filename": "quantum_jazz.mp3"
            },
            {
                "url": "https://freemusicarchive.org/track/Backbay_Lounge/download",
                "artist": "Kevin MacLeod",
                "title": "Backbay Lounge",
                "album": "Lounge",
                "filename": "backbay_lounge.mp3"
            },
        ]

        return samples

    def get_jamendo_samples(self) -> List[Dict]:
        """
        Jamendo provides CC-licensed music
        Note: These are example placeholders - use actual Jamendo API
        """
        # In production, use Jamendo API: https://api.jamendo.com/
        samples = [
            # Add Jamendo tracks here
        ]
        return samples


class AudioProcessor:
    """Process audio files and extract fingerprints"""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def convert_to_wav(self, input_file: Path) -> Optional[Path]:
        """Convert audio file to WAV format using ffmpeg"""
        try:
            output_file = input_file.with_suffix('.wav')

            if output_file.exists():
                logger.info(f"WAV file already exists: {output_file.name}")
                return output_file

            # Use ffmpeg to convert to WAV with correct sample rate
            cmd = [
                'ffmpeg', '-i', str(input_file),
                '-ar', str(self.sample_rate),  # Sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                str(output_file)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                return None

            logger.info(f"Converted to WAV: {output_file.name}")
            return output_file

        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install: brew install ffmpeg")
            return None
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout converting {input_file}")
            return None
        except Exception as e:
            logger.error(f"Error converting {input_file}: {e}")
            return None

    def load_audio(self, wav_file: Path) -> Optional[np.ndarray]:
        """Load WAV file as numpy array"""
        try:
            # Use scipy or numpy to read WAV
            from scipy.io import wavfile

            sample_rate, audio_data = wavfile.read(wav_file)

            # Convert to int16 if necessary
            if audio_data.dtype != np.int16:
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)

            logger.info(f"Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")
            return audio_data

        except Exception as e:
            logger.error(f"Error loading {wav_file}: {e}")
            return None

    def get_audio_duration(self, audio_data: np.ndarray) -> float:
        """Calculate audio duration in seconds"""
        return len(audio_data) / self.sample_rate


class FingerprintGenerator:
    """Generate fingerprints for songs with parallel processing"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.audio_processor = AudioProcessor()

    def process_song(self, metadata: SongMetadata) -> Optional[Tuple[List, SongMetadata]]:
        """
        Process a single song: convert, fingerprint, and prepare for storage

        Returns:
            Tuple of (hashes, metadata) or None if failed
        """
        try:
            logger.info(f"Processing: {metadata.title} by {metadata.artist}")

            # Check if already in database
            if song_in_db(metadata.file_path):
                logger.info(f"Song already in database: {metadata.title}")
                return None

            # Convert to WAV
            wav_file = self.audio_processor.convert_to_wav(Path(metadata.file_path))
            if not wav_file:
                return None

            # Load audio
            audio_data = self.audio_processor.load_audio(wav_file)
            if audio_data is None:
                return None

            # Update duration
            metadata.duration = self.audio_processor.get_audio_duration(audio_data)

            # Generate fingerprints
            logger.info(f"Generating fingerprints for {metadata.title}...")
            hashes = fingerprint_audio(audio_data)

            if not hashes or len(hashes) == 0:
                logger.error(f"No fingerprints generated for {metadata.title}")
                return None

            logger.info(f"Generated {len(hashes)} fingerprints for {metadata.title}")
            return (hashes, metadata)

        except Exception as e:
            logger.error(f"Error processing {metadata.title}: {e}", exc_info=True)
            return None

    def process_songs_parallel(self, songs: List[SongMetadata]) -> List[Tuple[List, SongMetadata]]:
        """Process multiple songs in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.process_song, song): song
                for song in songs
            }

            # Collect results as they complete
            for future in as_completed(futures):
                song = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in thread for {song.title}: {e}")

        return results


class DatabasePopulator:
    """Populate database with fingerprints"""

    def __init__(self):
        self.supabase = get_supabase()

    def store_songs(self, processed_songs: List[Tuple[List, SongMetadata]]) -> int:
        """
        Store processed songs in database

        Returns:
            Number of songs successfully stored
        """
        success_count = 0

        for hashes, metadata in processed_songs:
            try:
                logger.info(f"Storing {metadata.title} ({len(hashes)} fingerprints)...")

                # Prepare song info tuple
                song_info = (metadata.artist, metadata.album, metadata.title)

                # Store in database
                store_song(hashes, song_info)

                # Update song_stats if needed (this is handled by the view in current schema)
                # But we can add additional fields like duration
                song_id = hashes[0][2]
                try:
                    self.supabase.table("song_info").update({
                        "duration_seconds": metadata.duration
                    }).eq("song_id", song_id).execute()
                except Exception as e:
                    logger.warning(f"Could not update duration: {e}")

                success_count += 1
                logger.info(f"✅ Stored {metadata.title} successfully")

            except Exception as e:
                logger.error(f"Error storing {metadata.title}: {e}")

        return success_count


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Populate Shazomi database with song fingerprints')
    parser.add_argument('--scan', type=str, help='Scan local directory for music files')
    parser.add_argument('--download', action='store_true', help='Download sample songs from open datasets')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--music-dir', type=str, default='music_library', help='Music library directory')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Shazomi Database Population Script")
    logger.info("=" * 60)

    # Initialize components
    downloader = MusicDownloader(download_dir=args.music_dir)
    generator = FingerprintGenerator(max_workers=args.workers)
    populator = DatabasePopulator()

    # Determine source of songs
    songs_to_process = []

    if args.scan:
        # Scan local directory
        logger.info(f"\n📁 Scanning directory: {args.scan}")
        songs_to_process = downloader.scan_local_directory(args.scan)

    elif args.download:
        # Download from open datasets
        logger.info("\n📥 Downloading songs from open datasets...")
        sample_songs = downloader.get_free_music_archive_samples()

        for song_info in sample_songs:
            filepath = downloader.download_file(song_info["url"], song_info["filename"])
            if filepath:
                metadata = SongMetadata(
                    file_path=str(filepath),
                    artist=song_info["artist"],
                    album=song_info["album"],
                    title=song_info["title"]
                )
                songs_to_process.append(metadata)

    else:
        # Default: scan music_library directory
        logger.info(f"\n📁 Scanning default music library: {args.music_dir}")
        songs_to_process = downloader.scan_local_directory()

    logger.info(f"Found {len(songs_to_process)} songs to process")

    if not songs_to_process:
        logger.error("No songs to process. Use --scan <directory> or --download")
        logger.error("Example: python populate_database.py --scan ./music_library")
        return

    # Step 2: Process songs (generate fingerprints)
    logger.info(f"\n🔍 Step 2: Processing {len(songs_to_process)} songs in parallel...")
    start_time = time.time()

    processed_songs = generator.process_songs_parallel(songs_to_process)

    processing_time = time.time() - start_time
    logger.info(f"Processed {len(processed_songs)} songs in {processing_time:.2f} seconds")

    if not processed_songs:
        logger.error("No songs processed successfully. Exiting.")
        return

    # Step 3: Store in database
    logger.info(f"\n💾 Step 3: Storing {len(processed_songs)} songs in database...")
    success_count = populator.store_songs(processed_songs)

    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Summary:")
    logger.info(f"  - Songs found: {len(songs_to_process)}")
    logger.info(f"  - Songs processed: {len(processed_songs)}")
    logger.info(f"  - Songs stored: {success_count}")
    logger.info(f"  - Processing time: {processing_time:.2f}s")
    logger.info(f"  - Total time: {total_time:.2f}s")
    logger.info(f"  - Average per song: {total_time/max(len(processed_songs), 1):.2f}s")
    logger.info("=" * 60)
    logger.info("\n✅ Database population complete!")


if __name__ == "__main__":
    main()
