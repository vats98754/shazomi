-- Add listening_history table to track user song detections

CREATE TABLE IF NOT EXISTS listening_history (
    id BIGSERIAL PRIMARY KEY,
    uid TEXT NOT NULL,
    song_id TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    title TEXT NOT NULL,
    match_score INTEGER NOT NULL,
    detected_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_song FOREIGN KEY (song_id) REFERENCES song_info(song_id) ON DELETE CASCADE
);

-- Create indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_listening_history_uid ON listening_history (uid);
CREATE INDEX IF NOT EXISTS idx_listening_history_song_id ON listening_history (song_id);
CREATE INDEX IF NOT EXISTS idx_listening_history_detected_at ON listening_history (detected_at DESC);

-- Enable RLS
ALTER TABLE listening_history ENABLE ROW LEVEL SECURITY;

-- Create policy
CREATE POLICY "Allow all operations on listening_history"
    ON listening_history
    FOR ALL
    USING (true)
    WITH CHECK (true);
