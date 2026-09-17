import os
from pathlib import Path
import cv2
import librosa
import numpy as np
from tqdm import tqdm

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_PATH, "..", "..", "data"))
INPUT_DIR = os.path.join(DATA_DIR, "segmented")
OUTPUT_DIR = os.path.join(DATA_DIR, "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 128
N_MELS = 128
FMIN = 20.0  # Eliminates continuous DC/zero-frequency band at 0 Hz


def wav_to_image(filename):
    # Load audio preserving native sampling rate (96 kHz or 192 kHz)
    y, sr = librosa.load(filename, sr=None, mono=True)

    # Remove DC component
    y = y - np.mean(y)

    # Peak normalization
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Dynamic FFT window configuration based on sampling rate
    if sr >= 170000:  # Ultrasonic microphone (192 kHz)
        n_fft = 4096
        hop_length = 512
    elif sr >= 80000:  # Standard audible microphone (96 kHz)
        n_fft = 2048
        hop_length = 256
    else:
        n_fft = 1024
        hop_length = 128

    fmax = float(sr // 2)

    # Log-Mel Spectrogram computation
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=N_MELS,
        power=2.0,
        fmin=FMIN,
        fmax=fmax,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Continuous [0, 1] normalization
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    if (mel_max - mel_min) > 0:
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = np.zeros_like(mel_db)

    # Convert to uint8 (0-255)
    mel_img = (255.0 * mel_norm).astype(np.uint8)

    # Vertical flip to align 0 Hz at the image bottom (OpenCV convention)
    mel_img = np.flipud(mel_img)

    # Spatial resize to target dimensions (128x128)
    mel_resized = cv2.resize(
        mel_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
    )

    # Expand to 3 RGB channels for CNN compatibility
    return cv2.cvtColor(mel_resized, cv2.COLOR_GRAY2RGB)


def create_images():
    wav_files = list(Path(INPUT_DIR).rglob("*.wav"))
    print(f"[INFO] {len(wav_files)} .wav audio files identified.")

    for wav in tqdm(wav_files):
        relative = wav.relative_to(INPUT_DIR)
        output = Path(OUTPUT_DIR) / relative
        output = output.with_suffix(".png")

        output.parent.mkdir(parents=True, exist_ok=True)

        if output.exists():
            continue

        try:
            img = wav_to_image(str(wav))
            cv2.imwrite(str(output), img)
        except Exception as e:
            print(f"[ERROR] Failed to process {wav}: {e}")


if __name__ == "__main__":
    create_images()