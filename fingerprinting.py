"""
Audio fingerprinting using abracadabra algorithm
Adapted from https://github.com/notexactlyawe/abracadabra
"""

import uuid
import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
import logging

logger = logging.getLogger(__name__)

# Settings (from abracadabra)
SAMPLE_RATE = 16000  # Match Omi DevKit 2
FFT_WINDOW_SIZE = 0.2  # seconds
PEAK_BOX_SIZE = 15
POINT_EFFICIENCY = 0.8
TARGET_START = 0.05  # seconds
TARGET_T = 1.8  # seconds
TARGET_F = 4000  # Hz

# Windowing settings for robust matching
WINDOW_SIZE = 15.0  # seconds - size of each fingerprinting window
WINDOW_OVERLAP = 5.0  # seconds - overlap between windows


def my_spectrogram(audio):
    """Perform spectrogram calculation"""
    nperseg = int(SAMPLE_RATE * FFT_WINDOW_SIZE)
    return spectrogram(audio, SAMPLE_RATE, nperseg=nperseg)


def find_peaks(Sxx):
    """
    Find peaks in a spectrogram using maximum filter

    Args:
        Sxx: The spectrogram power values

    Returns:
        List of (frequency_idx, time_idx) tuples representing peaks
    """
    data_max = maximum_filter(Sxx, size=PEAK_BOX_SIZE, mode='constant', cval=0.0)
    peak_goodmask = (Sxx == data_max)
    y_peaks, x_peaks = peak_goodmask.nonzero()
    peak_values = Sxx[y_peaks, x_peaks]
    i = peak_values.argsort()[::-1]
    j = [(y_peaks[idx], x_peaks[idx]) for idx in i]

    # Calculate target number of peaks
    total = Sxx.shape[0] * Sxx.shape[1]
    peak_target = int((total / (PEAK_BOX_SIZE**2)) * POINT_EFFICIENCY)

    return j[:peak_target]


def idxs_to_tf_pairs(idxs, t, f):
    """Convert indices to time/frequency value pairs"""
    return np.array([(f[i[0]], t[i[1]]) for i in idxs])


def hash_point_pair(p1, p2):
    """Generate hash from two time/frequency points"""
    return hash((p1[0], p2[0], p2[1] - p1[1]))


def target_zone(anchor, points, width, height, t):
    """
    Generate target zone as described in Shazam paper

    Yields all points within a box that starts `t` seconds after the anchor point,
    with specified width and height.

    Args:
        anchor: The anchor point (freq, time)
        points: List of all points to search
        width: Width of target zone in seconds
        height: Height of target zone in Hz
        t: How many seconds after anchor to start target zone

    Yields:
        Points within the target zone
    """
    x_min = anchor[1] + t
    x_max = x_min + width
    y_min = anchor[0] - (height * 0.5)
    y_max = y_min + height

    for point in points:
        if point[0] < y_min or point[0] > y_max:
            continue
        if point[1] < x_min or point[1] > x_max:
            continue
        yield point


def hash_points(points, identifier="recorded"):
    """
    Generate all hashes for a list of peaks

    Args:
        points: List of (frequency, time) peaks
        identifier: Identifier for the audio (filename or "recorded")

    Returns:
        List of tuples: (hash, time_offset, song_id)
    """
    hashes = []
    song_id = uuid.uuid5(uuid.NAMESPACE_OID, identifier).int

    for anchor in points:
        for target in target_zone(anchor, points, TARGET_T, TARGET_F, TARGET_START):
            hashes.append((
                hash_point_pair(anchor, target),
                anchor[1],
                str(song_id)
            ))

    return hashes


def fingerprint_audio_window(audio_data: np.ndarray, identifier: str, window_offset: float = 0.0):
    """
    Fingerprint a single window of audio

    Args:
        audio_data: Raw PCM audio as numpy array (int16)
        identifier: Unique identifier for the audio
        window_offset: Time offset of this window in the original audio (seconds)

    Returns:
        List of tuples: (hash, time_offset, song_id) with adjusted time offsets
    """
    try:
        # Generate spectrogram
        f, t, Sxx = my_spectrogram(audio_data)

        # Find peaks
        peaks = find_peaks(Sxx)

        # Convert to time/frequency pairs
        peaks = idxs_to_tf_pairs(peaks, t, f)

        # Generate hashes
        hashes = hash_points(peaks, identifier)

        # Adjust time offsets by window position
        adjusted_hashes = []
        for hash_val, time_offset, song_id in hashes:
            adjusted_hashes.append((hash_val, time_offset + window_offset, song_id))

        return adjusted_hashes

    except Exception as e:
        logger.error(f"Error fingerprinting audio window: {e}", exc_info=True)
        return []


def fingerprint_audio(audio_data: np.ndarray, identifier: str = "recorded", use_windowing: bool = None):
    """
    Generate fingerprint hashes from raw audio data

    For long audio (>20s), automatically uses overlapping windows for robustness.
    For short audio (<20s), fingerprints the entire clip at once.

    Args:
        audio_data: Raw PCM audio as numpy array (int16)
        identifier: Unique identifier for the audio (file path for storage, "recorded" for recognition)
        use_windowing: Force windowing on/off. If None, auto-detect based on length.

    Returns:
        List of tuples: (hash, time_offset, song_id)
    """
    try:
        duration = len(audio_data) / SAMPLE_RATE
        logger.info(f"Fingerprinting audio: {len(audio_data)} samples ({duration:.1f}s)")

        # Auto-detect whether to use windowing
        if use_windowing is None:
            use_windowing = duration > 20.0

        if not use_windowing:
            # Short audio - fingerprint entire clip
            logger.info("Using single-pass fingerprinting")

            f, t, Sxx = my_spectrogram(audio_data)
            logger.info(f"Spectrogram shape: {Sxx.shape}")

            peaks = find_peaks(Sxx)
            logger.info(f"Found {len(peaks)} peaks")

            peaks = idxs_to_tf_pairs(peaks, t, f)
            hashes = hash_points(peaks, identifier)
            logger.info(f"Generated {len(hashes)} hashes")

            return hashes

        else:
            # Long audio - use overlapping windows
            logger.info(f"Using windowed fingerprinting (window={WINDOW_SIZE}s, overlap={WINDOW_OVERLAP}s)")

            window_samples = int(WINDOW_SIZE * SAMPLE_RATE)
            hop_samples = int((WINDOW_SIZE - WINDOW_OVERLAP) * SAMPLE_RATE)

            all_hashes = []
            window_count = 0

            for start_sample in range(0, len(audio_data) - window_samples + 1, hop_samples):
                end_sample = start_sample + window_samples
                window = audio_data[start_sample:end_sample]
                window_offset = start_sample / SAMPLE_RATE

                window_hashes = fingerprint_audio_window(window, identifier, window_offset)
                all_hashes.extend(window_hashes)
                window_count += 1

            # Handle last partial window if exists
            if len(audio_data) % hop_samples != 0:
                last_window = audio_data[-window_samples:]
                last_offset = (len(audio_data) - window_samples) / SAMPLE_RATE
                window_hashes = fingerprint_audio_window(last_window, identifier, last_offset)
                all_hashes.extend(window_hashes)
                window_count += 1

            logger.info(f"Processed {window_count} windows, generated {len(all_hashes)} total hashes")
            return all_hashes

    except Exception as e:
        logger.error(f"Error fingerprinting audio: {e}", exc_info=True)
        return []


def score_match(offsets):
    """
    Score a matched song using histogram of time deltas

    Args:
        offsets: List of (db_offset, sample_offset) tuples

    Returns:
        Score (size of largest histogram bin)
    """
    if len(offsets) < 2:
        return 0

    binwidth = 0.5
    tks = [x[0] - x[1] for x in offsets]

    hist, _ = np.histogram(
        tks,
        bins=np.arange(int(min(tks)), int(max(tks)) + binwidth + 1, binwidth)
    )

    return int(np.max(hist))


def best_match(matches):
    """
    Find the best matching song from a dictionary of matches

    Args:
        matches: Dict of song_id -> list of offset pairs

    Returns:
        Tuple of (song_id, score) for best match, or (None, 0) if no matches
    """
    if not matches:
        return None, 0

    matched_song = None
    best_score = 0

    for song_id, offsets in matches.items():
        if len(offsets) < best_score:
            # Can't beat current best score
            continue

        score = score_match(offsets)
        if score > best_score:
            best_score = score
            matched_song = song_id

    return matched_song, best_score
