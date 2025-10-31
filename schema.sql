-- Supabase database schema for shazomi
-- Run this SQL in the Supabase SQL Editor to set up your database

-- Create fingerprint_hash table for audio fingerprints
CREATE TABLE IF NOT EXISTS fingerprint_hash (
    id BIGSERIAL PRIMARY KEY,
    hash_value BIGINT NOT NULL,
    time_offset REAL NOT NULL,
    song_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index on hash_value for fast lookups (critical for performance)
CREATE INDEX IF NOT EXISTS idx_hash_value ON fingerprint_hash (hash_value);

-- Create index on song_id for faster filtering
CREATE INDEX IF NOT EXISTS idx_song_id ON fingerprint_hash (song_id);

-- Create song_info table for song metadata
CREATE TABLE IF NOT EXISTS song_info (
    song_id TEXT PRIMARY KEY,
    artist TEXT,
    album TEXT,
    title TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index on title for searching
CREATE INDEX IF NOT EXISTS idx_title ON song_info (title);

-- Enable Row Level Security (RLS) for both tables
ALTER TABLE fingerprint_hash ENABLE ROW LEVEL SECURITY;
ALTER TABLE song_info ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations (since this is a service, not user-facing)
-- In production, you may want more restrictive policies
CREATE POLICY "Allow all operations on fingerprint_hash"
    ON fingerprint_hash
    FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow all operations on song_info"
    ON song_info
    FOR ALL
    USING (true)
    WITH CHECK (true);

-- Optional: Create a view for song statistics
CREATE OR REPLACE VIEW song_stats AS
SELECT
    si.song_id,
    si.artist,
    si.album,
    si.title,
    COUNT(h.id) as fingerprint_count,
    si.created_at
FROM song_info si
LEFT JOIN fingerprint_hash h ON si.song_id = h.song_id
GROUP BY si.song_id, si.artist, si.album, si.title, si.created_at
ORDER BY fingerprint_count DESC;
