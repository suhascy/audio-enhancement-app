import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut=85, highcut=3000, fs=16000):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))
# ===== Imports =====
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt


# ===== Utility Functions =====
def bandpass_filter(data, sr, lowcut=85, highcut=3000, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def normalize_audio(audio):
    return audio / max(abs(audio))


# ===== Main Processing =====
audio, sr = librosa.load("recording.wav", sr=None)

reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.9)

filtered_audio = bandpass_filter(reduced_noise, sr)

enhanced_audio = normalize_audio(filtered_audio)

sf.write("final_output.wav", enhanced_audio, sr)

print("Processing Complete!")
