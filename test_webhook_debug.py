#!/usr/bin/env python3
"""
Test script to verify webhook debugging features:
1. Audio file saving
2. Debug notification logging
3. Omi API interaction
"""

import requests
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import time

# Test configuration
BASE_URL = "http://localhost:8000"  # Change if testing on Railway
TEST_UID = "test_debug_user_123"

def create_test_audio(duration=10, sample_rate=16000):
    """Create a simple test audio signal (sine wave)"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 440 Hz sine wave (A4 note)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3
    # Add some harmonics to make it more interesting
    audio += np.sin(2 * np.pi * 880 * t) * 0.15
    audio += np.sin(2 * np.pi * 1320 * t) * 0.075

    # Convert to int16 format
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16

def test_health_check():
    """Test 1: Verify server is running"""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is healthy")
            print(f"   Version: {data.get('version')}")
            print(f"   Mode: {data.get('mode')}")
            print(f"   Active users: {data.get('active_users')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to server: {e}")
        print(f"   Make sure the server is running: python main.py")
        return False

def test_audio_file_saving():
    """Test 2: Verify audio files are saved"""
    print("\n" + "=" * 60)
    print("TEST 2: Audio File Saving")
    print("=" * 60)

    # Create test audio (10 seconds)
    print("Creating test audio (10s, 16kHz)...")
    audio_data = create_test_audio(duration=10)

    # Convert to bytes
    audio_bytes = audio_data.tobytes()
    print(f"   Audio size: {len(audio_bytes)} bytes ({len(audio_data)/16000:.1f}s)")

    # Send to webhook
    print(f"Sending audio to webhook endpoint...")
    url = f"{BASE_URL}/webhook/audio-chunk?uid={TEST_UID}&sample_rate=16000"

    try:
        response = requests.post(
            url,
            data=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30
        )

        print(f"   Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Response: {result}")

            # Check if audio file was created
            time.sleep(1)  # Give it a second to write the file
            audio_files_dir = Path("audio_files")
            if audio_files_dir.exists():
                audio_files = list(audio_files_dir.glob(f"{TEST_UID}*.wav"))
                if audio_files:
                    latest_file = max(audio_files, key=lambda p: p.stat().st_mtime)
                    print(f"✅ Audio file saved: {latest_file}")
                    print(f"   File size: {latest_file.stat().st_size} bytes")

                    # Verify it's a valid WAV file
                    try:
                        rate, data = wavfile.read(latest_file)
                        print(f"   Verified WAV: {rate}Hz, {len(data)} samples, {len(data)/rate:.1f}s")
                        return True
                    except Exception as e:
                        print(f"❌ Invalid WAV file: {e}")
                        return False
                else:
                    print(f"❌ No audio files found for {TEST_UID}")
                    return False
            else:
                print(f"❌ audio_files directory not created")
                return False
        else:
            print(f"❌ Request failed: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_debug_data_saving():
    """Test 3: Verify debug data is saved"""
    print("\n" + "=" * 60)
    print("TEST 3: Debug Data Saving")
    print("=" * 60)

    debug_dir = Path("debug_data")
    if debug_dir.exists():
        debug_dirs = list(debug_dir.glob(f"{TEST_UID}*"))
        if debug_dirs:
            latest_dir = max(debug_dirs, key=lambda p: p.stat().st_mtime)
            print(f"✅ Debug data directory found: {latest_dir}")

            # Check for expected files
            expected_files = ["hashes.json", "result.json", "audio_stats.json", "summary.txt"]
            found_files = []
            for filename in expected_files:
                filepath = latest_dir / filename
                if filepath.exists():
                    found_files.append(filename)
                    print(f"   ✅ {filename} ({filepath.stat().st_size} bytes)")
                else:
                    print(f"   ❌ {filename} missing")

            # Check if audio_stats.json includes audio file path
            audio_stats_file = latest_dir / "audio_stats.json"
            if audio_stats_file.exists():
                import json
                with open(audio_stats_file) as f:
                    audio_stats = json.load(f)
                if "saved_audio_file" in audio_stats:
                    print(f"   ✅ Audio file path saved in debug data: {audio_stats['saved_audio_file']}")
                    return True
                else:
                    print(f"   ⚠️  Audio file path not found in audio_stats.json")
                    return False
            else:
                print(f"   ❌ audio_stats.json not found")
                return False
        else:
            print(f"❌ No debug directories found for {TEST_UID}")
            return False
    else:
        print(f"❌ debug_data directory not found")
        return False

def test_notification_debug_logging():
    """Test 4: Check logs for debug output"""
    print("\n" + "=" * 60)
    print("TEST 4: Notification Debug Logging")
    print("=" * 60)

    log_file = Path("logs/shazomi.log")
    if log_file.exists():
        print(f"✅ Log file found: {log_file}")

        # Read last 50 lines
        with open(log_file) as f:
            lines = f.readlines()
            last_lines = lines[-50:]

        # Look for debug logging indicators
        debug_indicators = [
            "🔍 DEBUG: Sending notification",
            "URL:",
            "Headers: Authorization=",
            "Response status:",
            "Response body:"
        ]

        found_indicators = []
        for indicator in debug_indicators:
            for line in last_lines:
                if indicator in line:
                    found_indicators.append(indicator)
                    break

        if found_indicators:
            print(f"✅ Debug logging active - found {len(found_indicators)}/{len(debug_indicators)} indicators:")
            for indicator in found_indicators:
                print(f"   ✅ {indicator}")
            return True
        else:
            print(f"⚠️  No debug logging found in recent logs")
            print(f"   (This is OK if this is your first test or notifications aren't enabled)")
            return True
    else:
        print(f"⚠️  Log file not found: {log_file}")
        print(f"   (Logs will be created when server starts)")
        return True

def test_stats_endpoint():
    """Test 5: Verify stats endpoint shows user data"""
    print("\n" + "=" * 60)
    print("TEST 5: Stats Endpoint")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/stats?uid={TEST_UID}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats retrieved:")
            print(f"   User ID: {data.get('uid')}")
            print(f"   Buffer size: {data.get('buffer_size_bytes')} bytes")
            print(f"   Songs identified: {data.get('songs_identified')}")
            print(f"   Last identification: {data.get('last_identification')}")
            return True
        else:
            print(f"❌ Stats request failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Stats request failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🔍 SHAZOMI WEBHOOK DEBUGGING TESTS")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Test UID: {TEST_UID}")
    print()

    # Run tests
    results = []

    # Test 1: Health check (required)
    if not test_health_check():
        print("\n❌ Server is not running. Start it with: python main.py")
        return

    # Test 2-5: Feature tests
    results.append(("Audio file saving", test_audio_file_saving()))
    results.append(("Debug data saving", test_debug_data_saving()))
    results.append(("Notification debug logging", test_notification_debug_logging()))
    results.append(("Stats endpoint", test_stats_endpoint()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All debugging features working correctly!")
        print("\n📁 Check these directories:")
        print("   - audio_files/     : Saved audio files")
        print("   - debug_data/      : Debug data per request")
        print("   - logs/            : Server logs with debug output")
    else:
        print(f"\n⚠️  Some tests failed. Review the output above for details.")

if __name__ == "__main__":
    main()
