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

-- Create listening_history table to track user song detections
CREATE TABLE IF NOT EXISTS listening_history (
    id BIGSERIAL PRIMARY KEY,
    uid TEXT NOT NULL,
    song_id TEXT NOT NULL,
    artist TEXT,
    album TEXT,
    title TEXT,
    match_score INTEGER NOT NULL,
    detected_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (song_id) REFERENCES song_info(song_id)
);

-- Create indexes for listening_history
CREATE INDEX IF NOT EXISTS idx_listening_uid ON listening_history (uid);
CREATE INDEX IF NOT EXISTS idx_listening_song_id ON listening_history (song_id);
CREATE INDEX IF NOT EXISTS idx_listening_detected_at ON listening_history (detected_at);

-- Enable RLS for listening_history
ALTER TABLE listening_history ENABLE ROW LEVEL SECURITY;

-- Create policy for listening_history
CREATE POLICY "Allow all operations on listening_history"
    ON listening_history
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

-- Optional: Create a view for user listening statistics
CREATE OR REPLACE VIEW user_listening_stats AS
SELECT
    uid,
    COUNT(*) as total_songs_detected,
    COUNT(DISTINCT song_id) as unique_songs,
    MAX(detected_at) as last_detection
FROM listening_history
GROUP BY uid
ORDER BY total_songs_detected DESC;
