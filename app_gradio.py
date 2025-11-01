#!/usr/bin/env python3
"""
Shazomi - Music Recognition Web App
Beautiful Gradio interface with file upload and microphone recording
"""

import os
os.environ['SUPABASE_URL'] = 'https://cekkrqnleagiizmrwvgn.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNla2tycW5sZWFnaWl6bXJ3dmduIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTg3OTA0NCwiZXhwIjoyMDc3NDU1MDQ0fQ.6z2o72SjD9hz1q-ezhe__rPPAmaOmpko5GjkHIU8Cn8'

import gradio as gr
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import tempfile
import subprocess
import time
from fingerprinting import fingerprint_audio, best_match, score_match, SAMPLE_RATE
from supabase_storage import get_matches, get_info_for_song_id, get_supabase
import matplotlib.pyplot as plt
import io
from PIL import Image

# Custom CSS for stunning UI
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.main-title {
    text-align: center;
    color: #1DB954;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 0.5em;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 1.2em;
    margin-bottom: 2em;
}
.result-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2em;
    border-radius: 15px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}
.confidence-high {
    color: #1DB954;
    font-weight: bold;
}
.confidence-medium {
    color: #FFA500;
    font-weight: bold;
}
.confidence-low {
    color: #FF4444;
    font-weight: bold;
}
"""

def load_audio_file(audio_input) -> tuple:
    """Load audio from file or microphone recording"""
    if audio_input is None:
        return None, None

    # Gradio returns (sample_rate, audio_data) for microphone
    # or file path for file upload
    if isinstance(audio_input, str):
        # File upload - use ffmpeg to convert to 16kHz mono WAV
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        try:
            cmd = [
                'ffmpeg', '-i', audio_input,
                '-ar', str(SAMPLE_RATE),
                '-ac', '1',
                '-y', tmp_wav_path
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode != 0:
                raise Exception(f"Audio conversion failed")

            sample_rate, audio_data = wavfile.read(tmp_wav_path)
        finally:
            Path(tmp_wav_path).unlink(missing_ok=True)
    else:
        # Microphone recording
        sample_rate, audio_data = audio_input

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            from scipy.signal import resample
            new_len = int(len(audio_data) * SAMPLE_RATE / sample_rate)
            audio_data = resample(audio_data, new_len)
            sample_rate = SAMPLE_RATE

    # Convert to int16
    if audio_data.dtype != np.int16:
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = (audio_data * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1).astype(np.int16)

    return audio_data, sample_rate

def create_spectrogram(audio_data):
    """Create a beautiful spectrogram visualization"""
    from fingerprinting import my_spectrogram

    f, t, Sxx = my_spectrogram(audio_data)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10),
                  shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title('Audio Spectrogram', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return Image.open(buf)

def recognize_song(audio_input, progress=gr.Progress()):
    """Main recognition function"""
    try:
        if audio_input is None:
            return None, "Please upload a file or record audio", None

        progress(0, desc="Loading audio...")

        # Load audio
        audio_data, sample_rate = load_audio_file(audio_input)

        if audio_data is None:
            return None, "Failed to load audio", None

        duration = len(audio_data) / sample_rate

        progress(0.2, desc=f"Loaded {duration:.1f}s of audio...")

        # Create spectrogram
        progress(0.3, desc="Generating spectrogram...")
        spec_image = create_spectrogram(audio_data)

        # Generate fingerprints
        progress(0.4, desc="Generating fingerprints...")
        start_time = time.time()
        hashes = fingerprint_audio(audio_data, identifier="recorded")
        fp_time = time.time() - start_time

        if not hashes:
            return spec_image, "❌ No fingerprints generated from audio", None

        progress(0.6, desc=f"Generated {len(hashes):,} fingerprints...")

        # Search database
        progress(0.7, desc="Searching database...")
        matches = get_matches(hashes, threshold=1)

        if not matches:
            result_html = f"""
            <div style='text-align: center; padding: 2em;'>
                <h2 style='color: #FF4444;'>🔍 No Match Found</h2>
                <p style='font-size: 1.2em;'>Song not in database</p>
                <p style='color: #666;'>Fingerprints: {len(hashes):,}</p>
                <p style='color: #666;'>Duration: {duration:.1f}s</p>
            </div>
            """
            return spec_image, result_html, None

        # Score ALL matches
        progress(0.9, desc="Scoring matches...")

        scored_matches = []
        for song_id, offsets in matches.items():
            score = score_match(offsets)
            scored_matches.append((song_id, score, len(offsets)))

        # Sort by score descending
        scored_matches.sort(key=lambda x: x[1], reverse=True)

        if not scored_matches:
            return spec_image, "❌ No strong match found", None

        best_song_id, best_score, best_raw = scored_matches[0]

        # Get song info
        info = get_info_for_song_id(best_song_id)

        if not info:
            return spec_image, "❌ Could not retrieve song information", None

        artist, album, title = info

        # Determine confidence
        if best_score > 100:
            confidence = "VERY HIGH"
            confidence_color = "#1DB954"
            emoji = "🎯"
        elif best_score > 50:
            confidence = "HIGH"
            confidence_color = "#1DB954"
            emoji = "✅"
        elif best_score > 20:
            confidence = "MEDIUM"
            confidence_color = "#FFA500"
            emoji = "⚠️"
        else:
            confidence = "LOW"
            confidence_color = "#FF4444"
            emoji = "❓"

        # Create beautiful result HTML
        result_html = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 2em; border-radius: 15px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3); margin: 1em 0;'>
            <h1 style='margin: 0; font-size: 2.5em; text-align: center;'>{emoji} {title}</h1>
            <h2 style='margin: 0.5em 0; text-align: center; opacity: 0.9;'>by {artist}</h2>
            <h3 style='margin: 0.5em 0; text-align: center; opacity: 0.8;'>{album}</h3>

            <div style='margin-top: 2em; padding-top: 2em; border-top: 2px solid rgba(255,255,255,0.3);'>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1em; text-align: center;'>
                    <div>
                        <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>Match Score</p>
                        <p style='margin: 0.2em 0 0 0; font-size: 2em; font-weight: bold;'>{best_score:,}</p>
                    </div>
                    <div>
                        <p style='margin: 0; opacity: 0.8; font-size: 0.9em;'>Confidence</p>
                        <p style='margin: 0.2em 0 0 0; font-size: 2em; font-weight: bold; color: {confidence_color};'>{confidence}</p>
                    </div>
                </div>
            </div>

            <div style='margin-top: 2em; padding-top: 2em; border-top: 2px solid rgba(255,255,255,0.3);'>
                <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1em; text-align: center;'>
                    <div>
                        <p style='margin: 0; opacity: 0.8; font-size: 0.8em;'>Duration</p>
                        <p style='margin: 0.2em 0 0 0; font-size: 1.2em;'>{duration:.1f}s</p>
                    </div>
                    <div>
                        <p style='margin: 0; opacity: 0.8; font-size: 0.8em;'>Fingerprints</p>
                        <p style='margin: 0.2em 0 0 0; font-size: 1.2em;'>{len(hashes):,}</p>
                    </div>
                    <div>
                        <p style='margin: 0; opacity: 0.8; font-size: 0.8em;'>Matches</p>
                        <p style='margin: 0.2em 0 0 0; font-size: 1.2em;'>{len(matches[best_song_id]):,}</p>
                    </div>
                </div>
            </div>
        </div>
        """

        # Create detailed stats with ALL match scores
        top_matches_html = ""
        for i, (sid, sc, raw) in enumerate(scored_matches[:10], 1):
            match_info = get_info_for_song_id(sid)
            if match_info:
                m_artist, m_album, m_title = match_info
                match_name = f"{m_title} - {m_artist}"
            else:
                match_name = sid[:40]

            if i == 1:
                color = "#1DB954"
            elif i == 2:
                color = "#FFA500"
            elif i == 3:
                color = "#FF4444"
            else:
                color = "#666"

            top_matches_html += f"""
                <tr style='border-bottom: 1px solid #ddd;'>
                    <td style='padding: 0.5em; font-weight: bold; color: {color};'>#{i}</td>
                    <td style='padding: 0.5em;'>{match_name}</td>
                    <td style='padding: 0.5em; text-align: right;'>{sc:,}</td>
                    <td style='padding: 0.5em; text-align: right; color: #999;'>{raw:,}</td>
                </tr>
            """

        stats_html = f"""
        <div style='padding: 1em; background: #f8f9fa; border-radius: 10px; margin-top: 1em;'>
            <h3 style='margin-top: 0; color: #333;'>Recognition Statistics</h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr style='border-bottom: 1px solid #ddd;'>
                    <td style='padding: 0.5em;'><strong>Processing Time:</strong></td>
                    <td style='padding: 0.5em;'>{fp_time:.2f}s</td>
                </tr>
                <tr style='border-bottom: 1px solid #ddd;'>
                    <td style='padding: 0.5em;'><strong>Audio Duration:</strong></td>
                    <td style='padding: 0.5em;'>{duration:.1f}s</td>
                </tr>
                <tr style='border-bottom: 1px solid #ddd;'>
                    <td style='padding: 0.5em;'><strong>Total Fingerprints:</strong></td>
                    <td style='padding: 0.5em;'>{len(hashes):,}</td>
                </tr>
                <tr style='border-bottom: 1px solid #ddd;'>
                    <td style='padding: 0.5em;'><strong>Database Matches:</strong></td>
                    <td style='padding: 0.5em;'>{best_raw:,}</td>
                </tr>
                <tr>
                    <td style='padding: 0.5em;'><strong>Songs with Matches:</strong></td>
                    <td style='padding: 0.5em;'>{len(matches)}</td>
                </tr>
            </table>

            <h3 style='margin-top: 1.5em; color: #333;'>Top Matches (Ranked by Score)</h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr style='border-bottom: 2px solid #333; background: #e0e0e0;'>
                    <th style='padding: 0.5em; text-align: left;'>Rank</th>
                    <th style='padding: 0.5em; text-align: left;'>Song</th>
                    <th style='padding: 0.5em; text-align: right;'>Score</th>
                    <th style='padding: 0.5em; text-align: right;'>Raw Matches</th>
                </tr>
                {top_matches_html}
            </table>
        </div>
        """

        progress(1.0, desc="Done!")

        return spec_image, result_html, stats_html

    except Exception as e:
        import traceback
        error_html = f"""
        <div style='background: #ffebee; color: #c62828; padding: 1.5em; border-radius: 10px; border-left: 5px solid #c62828;'>
            <h3 style='margin-top: 0;'>❌ Error</h3>
            <p><strong>Message:</strong> {str(e)}</p>
            <details>
                <summary>Show Details</summary>
                <pre style='margin-top: 1em; background: white; padding: 1em; border-radius: 5px; overflow-x: auto;'>{traceback.format_exc()}</pre>
            </details>
        </div>
        """
        return None, error_html, None

def check_database_status():
    """Check database connection and song count"""
    try:
        client = get_supabase()
        # Use faster query with limit to avoid timeout
        songs = client.table('song_info').select('count', count='exact').limit(1).execute()

        # Skip fingerprint count as it's very slow with large databases
        status_html = f"""
        <div style='background: #e8f5e9; padding: 1em; border-radius: 10px; border-left: 5px solid #4caf50;'>
            <h4 style='margin-top: 0; color: #2e7d32;'>✅ Database Connected</h4>
            <p style='margin: 0.5em 0;'><strong>Songs:</strong> {songs.count:,}</p>
            <p style='margin: 0.5em 0; color: #666;'><em>Ready for recognition</em></p>
        </div>
        """
        return status_html
    except Exception as e:
        return f"""
        <div style='background: #ffebee; padding: 1em; border-radius: 10px; border-left: 5px solid #c62828;'>
            <h4 style='margin-top: 0; color: #c62828;'>❌ Database Error</h4>
            <p>{str(e)}</p>
        </div>
        """

# Create Gradio interface
with gr.Blocks(css=CUSTOM_CSS, title="Shazomi - Music Recognition") as app:
    gr.HTML("""
        <div style='text-align: center; margin-bottom: 2em;'>
            <h1 style='color: #1DB954; font-size: 3.5em; margin-bottom: 0.2em;'>🎵 Shazomi</h1>
            <p style='font-size: 1.3em; color: #666;'>AI-Powered Music Recognition</p>
            <p style='color: #999;'>Upload a song or record audio to identify it instantly</p>
        </div>
    """)

    with gr.Row():
        with gr.Column():
            gr.HTML("<h3 style='text-align: center;'>📤 Upload Audio File</h3>")
            file_input = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="Upload Audio (MP3, WAV, M4A, etc.)"
            )
            file_button = gr.Button("🔍 Recognize from File", variant="primary", size="lg")

        with gr.Column():
            gr.HTML("<h3 style='text-align: center;'>🎤 Record Audio</h3>")
            mic_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record Audio"
            )
            mic_button = gr.Button("🔍 Recognize from Microphone", variant="primary", size="lg")

    with gr.Row():
        spectrogram_output = gr.Image(label="Spectrogram Visualization", type="pil")

    with gr.Row():
        result_output = gr.HTML(label="Recognition Result")

    with gr.Row():
        stats_output = gr.HTML(label="Detailed Statistics")

    gr.HTML("<hr style='margin: 2em 0;'>")

    with gr.Row():
        db_status = gr.HTML(label="Database Status")
        check_db_btn = gr.Button("🔄 Check Database Status")

    gr.HTML("""
        <div style='text-align: center; margin-top: 2em; padding-top: 2em; border-top: 2px solid #eee; color: #999;'>
            <p>Powered by Shazam-style fingerprinting • Built with Gradio</p>
        </div>
    """)

    # Connect buttons
    file_button.click(
        fn=recognize_song,
        inputs=[file_input],
        outputs=[spectrogram_output, result_output, stats_output]
    )

    mic_button.click(
        fn=recognize_song,
        inputs=[mic_input],
        outputs=[spectrogram_output, result_output, stats_output]
    )

    check_db_btn.click(
        fn=check_database_status,
        outputs=[db_status]
    )

    # Show initial message without loading database
    db_status.value = """
        <div style='background: #e3f2fd; padding: 1em; border-radius: 10px; border-left: 5px solid #2196F3;'>
            <h4 style='margin-top: 0; color: #1976d2;'>🔄 Database Ready</h4>
            <p style='margin: 0.5em 0;'>Click "Check Database Status" to view song count</p>
        </div>
    """

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
