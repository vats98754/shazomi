#!/usr/bin/env python3
"""
Download test songs from open sources for database population
Uses yt-dlp to download Creative Commons licensed music
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSongDownloader:
    """Download CC-licensed music for testing"""

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

    def download_from_youtube(self, url: str, title: str, artist: str, album: str = "YouTube Audio Library") -> Dict:
        """Download audio from YouTube using yt-dlp"""
        if not self.check_ytdlp():
            return None

        try:
            # Sanitize filename
            safe_filename = "".join(c for c in f"{artist}_{title}" if c.isalnum() or c in " _-").strip()
            output_template = str(self.output_dir / f"{safe_filename}.%(ext)s")

            # Download audio only, convert to mp3
            cmd = [
                'yt-dlp',
                '-x',  # Extract audio
                '--audio-format', 'mp3',
                '--audio-quality', '0',  # Best quality
                '-o', output_template,
                url
            ]

            logger.info(f"Downloading: {title} by {artist}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error(f"Download failed: {result.stderr}")
                return None

            # Find the downloaded file
            expected_file = self.output_dir / f"{safe_filename}.mp3"
            if expected_file.exists():
                return {
                    "file_path": str(expected_file),
                    "title": title,
                    "artist": artist,
                    "album": album
                }

            logger.error(f"Downloaded file not found: {expected_file}")
            return None

        except subprocess.TimeoutExpired:
            logger.error(f"Download timeout for {title}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {title}: {e}")
            return None

    def get_youtube_audio_library_samples(self) -> List[Dict]:
        """
        YouTube Audio Library - Creative Commons music
        These are official CC-licensed tracks
        """
        samples = [
            {
                "url": "https://www.youtube.com/watch?v=_tV5LEBDs7w",
                "title": "Arpent",
                "artist": "Moby",
                "album": "YouTube Audio Library"
            },
            {
                "url": "https://www.youtube.com/watch?v=SbZRLC8d5Uw",
                "title": "Ambient Piano",
                "artist": "Nomyn",
                "album": "YouTube Audio Library"
            },
            {
                "url": "https://www.youtube.com/watch?v=hKXu_Ozf8B0",
                "title": "Jazzy Abstract Beat",
                "artist": "Bensound",
                "album": "YouTube Audio Library"
            },
        ]
        return samples


def main():
    """Download test songs"""
    downloader = TestSongDownloader()

    logger.info("Downloading test songs from YouTube Audio Library...")
    samples = downloader.get_youtube_audio_library_samples()

    downloaded = []
    for sample in samples:
        result = downloader.download_from_youtube(
            sample["url"],
            sample["title"],
            sample["artist"],
            sample.get("album", "Unknown")
        )
        if result:
            downloaded.append(result)

    logger.info(f"\n✅ Successfully downloaded {len(downloaded)} songs:")
    for song in downloaded:
        logger.info(f"  - {song['title']} by {song['artist']}")

    logger.info(f"\nSongs saved to: {downloader.output_dir}")
    logger.info("Run populate_database.py next to process these songs!")


if __name__ == "__main__":
    main()
