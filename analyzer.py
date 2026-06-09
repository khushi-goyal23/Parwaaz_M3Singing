import numpy as np
import librosa
import torch
import torchcrepe
import os
import json
import whisper
import subprocess
import glob
import sys

TARGET_SR = 22050
BASE_DIR = "."


def separate_vocals(input_path, output_dir="separated"):
    os.makedirs(output_dir, exist_ok=True)

    # Use current interpreter so it works inside active venv/conda too.
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", output_dir, input_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "Unknown demucs error")
    except Exception:
        # Keep demo usable even if stem separation is unavailable.
        return input_path

    # Demucs model folder name can vary; search recursively.
    matches = glob.glob(os.path.join(output_dir, "**", "vocals.wav"), recursive=True)
    if matches:
        return matches[0]

    # Fallback for demos where only mixed track exists.
    return input_path

# ---------------- PREPROCESS ----------------
def load_and_preprocess(audio_path):
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    return {"audio": y, "sr": sr, "duration": duration}


# ---------------- PITCH ----------------
def extract_pitch(audio, sr):
    import torch
    import torchcrepe

    audio = torch.tensor(audio).float().unsqueeze(0)

    # predict pitch + confidence
    pitch, confidence = torchcrepe.predict(
        audio,
        sr,
        hop_length=160,
        fmin=50,
        fmax=1000,
        model='tiny',
        batch_size=1024,
        device='cpu',
        return_periodicity=True
    )

    frequency = pitch.squeeze().numpy()
    confidence = confidence.squeeze().numpy()

    # frame-accurate time axis from hop length
    time = (np.arange(len(frequency)) * 160) / float(sr)

    return {
        "time": time,
        "frequency": frequency,
        "confidence": confidence
    }


def hz_to_cents(freq):
    freq = np.where(freq <= 0, 1e-6, freq)
    return 1200 * np.log2(freq / 440.0)


def filter_pitch(pitch, th=0.5):
    mask = pitch["confidence"] >= th
    # If periodicity is low for the whole clip, keep all frames
    # so downstream stages can still run for demo purposes.
    if not np.any(mask):
        mask = np.ones_like(pitch["confidence"], dtype=bool)
    return {"cents": hz_to_cents(pitch["frequency"][mask]), "time": pitch["time"][mask]}


# ---------------- ALIGN ----------------
def align_pitch(ref, user):
    ref = ref.reshape(-1, 1)
    user = user.reshape(-1, 1)

    D, wp = librosa.sequence.dtw(X=ref.T, Y=user.T)
    wp = np.array(wp)[::-1]

    return ref[wp[:, 0]].flatten(), user[wp[:, 1]].flatten()

#------------ RHYTHM----------------------

def extract_onsets(audio, sr):
    frames = librosa.onset.onset_detect(y=audio, sr=sr)
    return librosa.frames_to_time(frames, sr=sr)


def compute_rhythm(ref_on, user_on):

    if len(ref_on) == 0 or len(user_on) == 0:
        return 500, 1.0   # fallback (bad rhythm)

    m = min(len(ref_on), len(user_on))
    ref = ref_on[:m]
    user = user_on[:m]

    deviation = np.abs(ref - user) * 1000
    mean_dev = np.mean(deviation)

    tempo_ratio = user_on[-1] / ref_on[-1] if ref_on[-1] > 0 else 1.0

    return mean_dev, tempo_ratio


#------------- STABILITY -----------------

def compute_stability(cents):
    return float(np.std(cents))


#-----------LYRICS---------------------

whisper_model = None


def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        try:
            whisper_model = whisper.load_model("base")
        except Exception:
            whisper_model = None
    return whisper_model


def transcribe(audio_path):
    model = get_whisper_model()
    if model is None:
        return ""
    try:
        result = model.transcribe(audio_path)
        return result.get("text", "")
    except FileNotFoundError:
        # Common on Windows when ffmpeg is not on PATH.
        return ""
    except Exception:
        return ""


#------------LYRICS SCORE-----------


def compute_wer(ref, user):
    ref_words = ref.split()
    user_words = user.split()

    m = min(len(ref_words), len(user_words))
    errors = sum(r != u for r, u in zip(ref_words[:m], user_words[:m]))

    return errors / max(1, len(ref_words))

#---------EARLY DETECT --------------


def detect_pitch_tendency(ref, user):
    diff = user - ref

    mean_diff = np.mean(diff)

    if mean_diff < -30:
        return "You are consistently singing flat (below pitch)."
    elif mean_diff > 30:
        return "You are consistently singing sharp (above pitch)."
    else:
        return "Pitch alignment is generally balanced."


#---------- EARLY OR FAST

def detect_timing_tendency(ref_on, user_on):

    if len(ref_on) == 0 or len(user_on) == 0:
        return "Timing could not be evaluated."

    m = min(len(ref_on), len(user_on))
    ref = ref_on[:m]
    user = user_on[:m]

    diff = user - ref   # signed difference (seconds)

    mean_shift = np.mean(diff)

    if mean_shift < -0.05:
        return "You tend to sing early (ahead of beat)."
    elif mean_shift > 0.05:
        return "You tend to sing late (behind the beat)."
    else:
        return "Timing is generally aligned."


# ---------------- KEY ----------------
def compute_key_offset(ref, user):
    ref_m = np.median(ref)
    user_m = np.median(user)
    offset = int(round((user_m - ref_m) / 100))
    return max(-6, min(6, offset))


def apply_key_correction(user, offset):
    return user - offset * 100


# ---------------- METRICS ----------------
def compute_pitch_metrics(ref, user):
    diff = np.abs(ref - user)
    mean_error = np.mean(diff)
    in_range = np.mean(diff <= 50) * 100
    return mean_error, in_range


# ---------------- CACHE ----------------
def save_cache(data, path):
    d = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f)


def load_cache(path):
    with open(path) as f:
        d = json.load(f)
    for k in d:
        if isinstance(d[k], list):
            d[k] = np.array(d[k])
    return d



def generate_timeline_feedback(ref, user):
    feedback = []
    window = 50  # frames per chunk
    
    for i in range(0, len(ref), window):
        r = ref[i:i+window]
        u = user[i:i+window]
        
        if len(r) == 0:
            continue
        
        error = np.mean(np.abs(r - u))
        
        if error < 30:
            msg = "Good pitch"
        elif error < 80:
            msg = "Slightly off"
        else:
            msg = "Off pitch"
        
        feedback.append({
            "start": round(i/100, 2),
            "end": round((i+window)/100, 2),
            "message": msg
        })
    
    return feedback


# ---------------- PIPELINES ----------------
def process_reference(path, cache):

    if os.path.exists(cache):
        return load_cache(cache)

    print("Running Demucs (this may take time)...")

    vocal_path = separate_vocals(path)

    data = load_and_preprocess(vocal_path)

    pitch = filter_pitch(extract_pitch(data["audio"], data["sr"]))
    onsets = extract_onsets(data["audio"], data["sr"])
    lyrics = transcribe(vocal_path)

    out = {
        "cents": pitch["cents"],
        "onsets": onsets,
        "lyrics": lyrics
    }

    save_cache(out, cache)
    return out


def process_user(path, cache):

    if os.path.exists(cache):
        return load_cache(cache)

    data = load_and_preprocess(path)

    raw_pitch = extract_pitch(data["audio"], data["sr"])
    pitch = filter_pitch(raw_pitch)
    onsets = extract_onsets(data["audio"], data["sr"])
    lyrics = transcribe(path)
    live_timeline = build_live_note_timeline(raw_pitch)

    out = {
        "cents": pitch["cents"],
        "time": pitch["time"],
        "live_timeline": live_timeline,
        "onsets": onsets,
        "lyrics": lyrics
    }

    save_cache(out, cache)
    return out

# -------- SCALE + TRANSITIONS --------

MAJOR_SCALES = {
    "C Major": ['C','D','E','F','G','A','B'],
    "G Major": ['G','A','B','C','D','E','F#'],
    "D Major": ['D','E','F#','G','A','B','C#'],
    "A Major": ['A','B','C#','D','E','F#','G#'],
    "E Major": ['E','F#','G#','A','B','C#','D#'],
}

def detect_scale(notes):
    from collections import Counter
    
    note_counts = Counter(notes)
    unique_notes = set(note_counts.keys())
    
    best_scale = None
    best_score = 0
    
    for scale, scale_notes in MAJOR_SCALES.items():
        score = len(unique_notes.intersection(scale_notes))
        
        if score > best_score:
            best_score = score
            best_scale = scale
    
    return best_scale if best_scale else "Unknown"


def get_note_transitions(notes):
    transitions = []
    
    for i in range(len(notes) - 1):
        if notes[i] != notes[i+1]:
            transitions.append(f"{notes[i]} → {notes[i+1]}")
    
    return transitions[:10]

# -------- NOTE UTILITIES --------

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def cents_to_note(cents):
    freq = 440 * (2 ** (cents / 1200))
    midi = int(round(69 + 12 * np.log2(freq / 440)))
    return NOTE_NAMES[midi % 12]


def hz_to_note(freq):
    if freq <= 0:
        return None
    midi = int(round(69 + 12 * np.log2(freq / 440.0)))
    return NOTE_NAMES[midi % 12]


def extract_notes_with_time(cents_array, time_array):
    if len(cents_array) == 0 or len(time_array) == 0:
        return []
    notes = [cents_to_note(c) for c in cents_array]
    
    result = []
    current_note = notes[0]
    start_time = time_array[0]
    
    for i in range(1, len(notes)):
        if notes[i] != current_note:
            end_time = time_array[i]
            
            result.append({
                "note": current_note,
                "start": float(start_time),
                "end": float(end_time),
                "duration": float(end_time - start_time)
            })
            
            current_note = notes[i]
            start_time = time_array[i]
    
    result.append({
        "note": current_note,
        "start": float(start_time),
        "end": float(time_array[-1]),
        "duration": float(time_array[-1] - start_time)
    })
    
    return result


def get_note_durations(note_segments):
    return [
        {"note": seg["note"], "duration": round(seg["duration"], 2)}
        for seg in note_segments if seg["duration"] > 0.1
    ]


def get_note_timeline(note_segments):
    return [
        {
            "note": seg["note"],
            "start": round(float(seg["start"]), 2),
            "end": round(float(seg["end"]), 2),
        }
        for seg in note_segments
        if seg["duration"] > 0.1
    ]


def build_live_note_timeline(pitch, conf_th=0.45, min_duration=0.08):
    freq = pitch["frequency"]
    conf = pitch["confidence"]
    time = pitch["time"]
    if len(time) == 0:
        return []

    # Smooth note jitter with a short local median on MIDI values.
    midi_vals = np.full(len(time), np.nan)
    for i in range(len(time)):
        if conf[i] >= conf_th and freq[i] > 0:
            midi_vals[i] = 69 + 12 * np.log2(freq[i] / 440.0)

    smoothed = np.full(len(time), np.nan)
    for i in range(len(time)):
        if np.isnan(midi_vals[i]):
            continue
        lo = max(0, i - 3)
        hi = min(len(time), i + 4)
        window = midi_vals[lo:hi]
        valid = window[~np.isnan(window)]
        smoothed[i] = np.median(valid) if len(valid) > 0 else midi_vals[i]

    labels = []
    for i in range(len(time)):
        if np.isnan(smoothed[i]):
            labels.append("Rest")
        else:
            midi = int(round(smoothed[i]))
            labels.append(NOTE_NAMES[midi % 12])

    timeline = []
    start_idx = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            start_t = float(time[start_idx])
            end_t = float(time[i])
            if (end_t - start_t) >= min_duration:
                timeline.append({
                    "note": current,
                    "start": round(start_t, 2),
                    "end": round(end_t, 2),
                })
            start_idx = i
            current = labels[i]

    start_t = float(time[start_idx])
    frame_dt = (float(time[-1]) - float(time[-2])) if len(time) > 1 else 0.01
    end_t = float(time[-1]) + max(frame_dt, 0.01)
    if (end_t - start_t) >= min_duration:
        timeline.append({
            "note": current,
            "start": round(start_t, 2),
            "end": round(end_t, 2),
        })

    return timeline

def compute_rhythm_score(rhythm_dev):
    # Linear + bounded
    if rhythm_dev < 100:
        return 90
    elif rhythm_dev < 200:
        return 75
    elif rhythm_dev < 400:
        return 60
    elif rhythm_dev < 800:
        return 40
    elif rhythm_dev < 1500:
        return 20
    else:
        return 5


def _ensure_non_empty_cents(cents):
    # Keep analysis resilient for difficult/noisy uploads.
    return cents if len(cents) > 0 else np.array([0.0], dtype=float)

def analyze_full(user_path, reference_path):

    # clear cache
    for p in [
        BASE_DIR + "/cache/reference/ref.json",
        BASE_DIR + "/cache/user/user.json"
    ]:
        if os.path.exists(p):
            os.remove(p)

    ref = process_reference(reference_path, BASE_DIR + "/cache/reference/ref.json")
    user = process_user(user_path, BASE_DIR + "/cache/user/user.json")
    ref["cents"] = _ensure_non_empty_cents(ref["cents"])
    user["cents"] = _ensure_non_empty_cents(user["cents"])

    # KEY NORMALIZATION
    offset = compute_key_offset(ref["cents"], user["cents"])
    user_corr = apply_key_correction(user["cents"], offset)

    # ALIGNMENT
    ref_a, user_a = align_pitch(ref["cents"], user_corr)

    # ---------- NOTE ANALYSIS ----------

    # Use the user's original detected pitch timestamps for live UI sync.
    user_time = user.get("time")
    if user_time is None or len(user_time) != len(user["cents"]):
        user_time = np.linspace(0, len(user["cents"]) / 100, len(user["cents"]))
    note_segments = extract_notes_with_time(user["cents"], user_time)
    notes = [seg["note"] for seg in note_segments]

    scale = detect_scale(notes)
    transitions = get_note_transitions(notes)
    durations = get_note_durations(note_segments)
    note_timeline = user.get("live_timeline", get_note_timeline(note_segments))

    pitch_tendency = detect_pitch_tendency(ref_a, user_a)

    timing_tendency = detect_timing_tendency(ref["onsets"], user["onsets"])

    timeline = generate_timeline_feedback(ref_a, user_a)

    # ---------------- METRICS ----------------
    pitch_error, pitch_acc = compute_pitch_metrics(ref_a, user_a)

    rhythm_dev, tempo = compute_rhythm(ref["onsets"], user["onsets"])

    stability = compute_stability(user_a)

    wer = compute_wer(ref["lyrics"], user["lyrics"])

    # ---------------- SCORES ----------------

    pitch_score = pitch_acc

    rhythm_score = compute_rhythm_score(rhythm_dev)

    lyrics_score = max(0, 100 * np.exp(-wer * 2))

    stability_score = max(0, 100 * np.exp(-stability / 200))

    # Clamp all
    pitch_score = max(0, min(100, pitch_score))
    rhythm_score = max(0, min(100, rhythm_score))
    lyrics_score = max(0, min(100, lyrics_score))
    stability_score = max(0, min(100, stability_score))

    # FINAL SCORE
    final_score = (
        0.4 * pitch_score +
        0.3 * rhythm_score +
        0.2 * lyrics_score +
        0.1 * stability_score
    )

    return {
        "pitch_accuracy": float(pitch_score),
        "pitch_error": float(pitch_error),
        "rhythm_deviation_ms": float(rhythm_dev),
        "tempo_ratio": float(tempo),
        "stability": float(stability),
        "lyrics_error": float(wer),
        "key_offset": int(offset),
        "final_score": float(final_score),
        "ref_contour": ref_a.tolist(),
        "user_contour": user_a.tolist(),
        "pitch_tendency": pitch_tendency,
        "timing_tendency": timing_tendency, 
        "detected_scale": scale,
        "note_transitions": transitions,
        "note_durations": durations,
        "note_timeline": note_timeline,
        "timeline_feedback": timeline
    }

def analyze_pitch(user_path, reference_path):

    result = analyze_full(user_path, reference_path)

    return {
        "mean_error_cents": float(result["pitch_error"]),
        "in_range_percent": float(result["pitch_accuracy"]),

        # ✅ USE ALREADY COMPUTED SCORES
        "rhythm_score": float(compute_rhythm_score(result["rhythm_deviation_ms"])),
        "lyrics_score": float(max(0, 100 * np.exp(-result["lyrics_error"] * 2))),

        "final_score": float(result["final_score"]),
        "feedback": generate_feedback(result),
        "pitch_contour": [float(x) for x in result["user_contour"]],
        "pitch_tendency": result["pitch_tendency"],
        "timing_tendency": result["timing_tendency"],

        "detected_scale": result["detected_scale"],
        "note_transitions": result["note_transitions"],
        "note_durations": result["note_durations"],
        "note_timeline": result["note_timeline"],
        "timeline_feedback": result["timeline_feedback"]
    }

def generate_feedback(r):

    feedback = []

    if r["pitch_accuracy"] < 60:
        feedback.append("You are frequently off-pitch. Focus on matching notes.")
    elif r["pitch_accuracy"] < 80:
        feedback.append("Pitch is decent but needs refinement.")

    if r["rhythm_deviation_ms"] > 250:
        feedback.append("Timing is inconsistent. Practice with a metronome.")
    elif r["rhythm_deviation_ms"] > 120:
        feedback.append("Slight timing issues detected.")

    if r["lyrics_error"] > 0.4:
        feedback.append("Lyrics accuracy is low.")
    elif r["lyrics_error"] > 0.2:
        feedback.append("Minor lyric mistakes.")

    if r["stability"] > 200:
        feedback.append("Your voice is unstable on sustained notes.")

    if not feedback:
        return "Strong performance overall."

    return " ".join(feedback)