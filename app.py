from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import shutil
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Utility functions
# -----------------------------
def bandpass_filter(data, sr, lowcut=100, highcut=3000, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val

# -----------------------------
# Processing logic
# -----------------------------
def process_audio_logic(input_path, noise):
    audio, sr = librosa.load(input_path, sr=None)

    # Convert stereo → mono (VERY IMPORTANT)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    print("Loaded:", np.min(audio), np.max(audio))

    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=min(noise / 50, 0.9)
    )

    print("After NR:", np.min(reduced), np.max(reduced))

    filtered = bandpass_filter(reduced, sr)

    enhanced = normalize_audio(filtered)

    print("Final:", np.min(enhanced), np.max(enhanced))
    print("NaN check:", np.isnan(enhanced).any())

    return enhanced.astype(np.float32), sr

# -----------------------------
# Streaming helper
# -----------------------------
def iterfile(path):
    with open(path, "rb") as f:
        yield from f

# -----------------------------
# API route
# -----------------------------
from fastapi.responses import FileResponse

@app.post("/process-audio/")
async def process_audio(
    file: UploadFile = File(...),
    noise: int = Form(25)
):
    input_path = "input.wav"
    output_path = "output.wav"

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process audio
    audio, sr = process_audio_logic(input_path, noise)

    # Save processed audio
    sf.write(output_path, audio, sr, subtype='PCM_16')

    print("Returning processed audio ✅")

    # ✅ THIS MUST BE INSIDE FUNCTION
    return FileResponse(output_path, media_type="audio/wav")