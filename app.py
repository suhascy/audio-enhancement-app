from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import shutil
import librosa
import noisereduce as nr
import soundfile as sf

# ✅ FIRST create app
app = FastAPI()

# ✅ THEN add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ THEN define routes
@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):

    input_path = "input.wav"
    output_path = "output.wav"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    audio, sr = librosa.load(input_path, sr=None)

    reduced_noise = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False
    )

    sf.write(output_path, reduced_noise, sr)

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="processed.wav"
    )