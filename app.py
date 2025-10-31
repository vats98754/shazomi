from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import io
import json
from scipy.io import wavfile
from scipy.signal import resample as scipy_resample

# Lazy import pydub inside the processing route so the app can start even if
# optional packages are missing. We'll handle ImportError in the route.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fingerprinting import fingerprint_audio, my_spectrogram, find_peaks, idxs_to_tf_pairs

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
FINGERPRINT_DB = os.path.join(BASE_DIR, 'fingerprints.json')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


def save_fingerprints(hashes):
    data = []
    if os.path.exists(FINGERPRINT_DB):
        with open(FINGERPRINT_DB, 'r') as f:
            try:
                data = json.load(f)
            except Exception:
                data = []

    data.extend(hashes)
    with open(FINGERPRINT_DB, 'w') as f:
        json.dump(data, f, indent=2)


def audiosegment_to_np(audio_seg):
    """Convert a pydub.AudioSegment-like object to a mono numpy array"""
    samples = np.array(audio_seg.get_array_of_samples())
    if audio_seg.channels > 1:
        samples = samples.reshape((-1, audio_seg.channels))
        samples = samples.mean(axis=1).astype(samples.dtype)
    return samples


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Accepts file upload (mp3/wav/webm), converts to mono 16kHz, fingerprints, and returns images and counts."""
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    f = request.files['file']
    filename = secure_filename(f.filename or 'upload')
    path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)

    # If the uploaded file is a WAV we can load it without ffmpeg using scipy
    ext = os.path.splitext(filename)[1].lower()
    samples = None
    if ext == '.wav':
        try:
            sr, data = wavfile.read(path)
        except Exception as e:
            return jsonify({'error': f'Could not read WAV file: {e}'}), 400

        # Convert to mono if needed
        if getattr(data, 'ndim', 1) > 1:
            data = data.mean(axis=1)

        # If sample rate differs, resample to 16000
        if sr != 16000:
            try:
                new_len = int(len(data) * 16000 / float(sr))
                data = scipy_resample(data, new_len)
            except Exception as e:
                return jsonify({'error': f'Could not resample WAV: {e}'}), 500

        # Ensure int16 format for fingerprinting
        if not np.issubdtype(data.dtype, np.integer):
            data = np.asarray(data)
            data = data / (np.max(np.abs(data)) + 1e-9)
            data = (data * 32767).astype(np.int16)

        samples = data.astype(np.int16)
    else:
        # Use pydub (requires ffmpeg/ffprobe) for other formats
        try:
            from pydub import AudioSegment
        except Exception:
            return jsonify({'error': 'pydub (and ffmpeg/ffprobe) required to process this file type. Install requirements and ffmpeg.'}), 500

        try:
            audio = AudioSegment.from_file(path)
        except Exception as e:
            return jsonify({'error': f'Could not load audio file: {e}'}), 500

        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        samples = audiosegment_to_np(audio)

    # Convert to float32 for spectrogram
    samples_f = samples.astype(np.float32)

    # Generate spectrogram and peaks
    f_axis, t_axis, Sxx = my_spectrogram(samples_f)
    peaks = find_peaks(Sxx)
    tf_peaks = idxs_to_tf_pairs(peaks, t_axis, f_axis)

    # Save spectrogram image
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(t_axis, f_axis, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    if len(tf_peaks) > 0:
        ax.scatter(tf_peaks[:, 1], tf_peaks[:, 0], c='r', s=5)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    img_buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(img_buf, format='png')
    plt.close(fig)
    img_buf.seek(0)

    img_name = f"spectrogram_{uuid_filename()}.png"
    img_path = os.path.join(UPLOAD_DIR, img_name)
    with open(img_path, 'wb') as out_f:
        out_f.write(img_buf.read())

    # Fingerprint (expecting int16 samples)
    hashes = fingerprint_audio(samples)
    save_fingerprints(hashes)

    return jsonify({
        'spectrogram': f'/uploads/{img_name}',
        'peaks': int(len(tf_peaks)),
        'hashes': int(len(hashes))
    })


def uuid_filename():
    import uuid as _u
    return _u.uuid4().hex


@app.route('/uploads/<path:p>')
def uploaded_file(p):
    return send_from_directory(UPLOAD_DIR, p)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
    # If the uploaded file is a WAV we can load it without ffmpeg using scipy
