#!/usr/bin/env python3
"""
Wrapper to run populate_10_songs.py with fresh environment
"""

import os
import sys

# Set environment variables BEFORE importing any modules
os.environ['SUPABASE_URL'] = 'https://cekkrqnleagiizmrwvgn.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNla2tycW5sZWFnaWl6bXJ3dmduIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTg3OTA0NCwiZXhwIjoyMDc3NDU1MDQ0fQ.6z2o72SjD9hz1q-ezhe__rPPAmaOmpko5GjkHIU8Cn8'

# Now import and run the main script
from populate_10_songs import main

if __name__ == "__main__":
    main()
