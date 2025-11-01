#!/usr/bin/env python3
"""
Check the current database status
"""

import os
os.environ['SUPABASE_URL'] = 'https://cekkrqnleagiizmrwvgn.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNla2tycW5sZWFnaWl6bXJ3dmduIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTg3OTA0NCwiZXhwIjoyMDc3NDU1MDQ0fQ.6z2o72SjD9hz1q-ezhe__rPPAmaOmpko5GjkHIU8Cn8'

from supabase_storage import get_supabase

print("="*70)
print("🎵 SHAZOMI DATABASE STATUS")
print("="*70)

try:
    client = get_supabase()

    # Get song count
    songs = client.table('song_info').select('*').execute()
    fingerprints = client.table('fingerprint_hash').select('song_id', count='exact').execute()

    print(f"\n📊 Statistics:")
    print(f"   Total Songs: {len(songs.data)}")
    print(f"   Total Fingerprints: {fingerprints.count:,}")

    if len(songs.data) > 0:
        avg_fps = fingerprints.count / len(songs.data)
        print(f"   Avg Fingerprints/Song: {avg_fps:,.0f}")

    print(f"\n🎵 Songs in Database:")
    for i, song in enumerate(songs.data[:10], 1):
        print(f"   {i}. {song['title']} - {song['artist']}")

    if len(songs.data) > 10:
        print(f"   ... and {len(songs.data) - 10} more")

    print("\n" + "="*70)

except Exception as e:
    print(f"❌ Error: {e}")
