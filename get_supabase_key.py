#!/usr/bin/env python3
"""
Helper to get the correct Supabase API key
"""

import os

print("=" * 70)
print("SUPABASE API KEY SETUP")
print("=" * 70)
print()
print("Your current .env file has:")
print("  SUPABASE_KEY=sb_secret_gaDibmLRke2DoFanuxiqng_-p-XCP71")
print()
print("❌ This key is INCOMPLETE and won't work.")
print()
print("=" * 70)
print("TO GET THE CORRECT KEY:")
print("=" * 70)
print()
print("1. Open: https://supabase.com/dashboard/project/cekkrqnleagiizmrwvgn/settings/api")
print()
print("2. Look for the 'service_role' section (NOT 'anon public')")
print()
print("3. Click the eye icon to reveal the key")
print()
print("4. Copy the FULL key - it should:")
print("   - Start with: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
print("   - Be 200-300 characters long")
print("   - Look like: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6...")
print()
print("5. Replace line 9 in .env file with:")
print("   SUPABASE_KEY=<paste_the_full_key_here>")
print()
print("=" * 70)
print()
print("After updating, run: python test_supabase_connection.py")
print()
