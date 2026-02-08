import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

print("✅ Script started")

SAMPLE_RATE = 16000
DURATION = 5

print("🎙️ About to record...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype=np.float32
)

print("⏳ Recording...")
sd.wait()
print("✅ Recording finished")

audio = audio.flatten()

print("🧠 Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")

print("📝 Transcribing...")
segments, info = model.transcribe(audio, language="en")

print("\nRESULT:")
for segment in segments:
    print(segment.text)
