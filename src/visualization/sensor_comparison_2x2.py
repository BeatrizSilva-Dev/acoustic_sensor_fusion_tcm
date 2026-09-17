import os
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_PATH, "..", ".."))

ROOT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "segmented",
    "drill_4mm_06_batch_00_collet_1_30-01-2025",
)
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

INTERNAL_CHANNEL = "Tr4"
EXTERNAL_CHANNEL = "Tr1"

NORMAL_HOLE = 1
JAM_HOLE = 25
FOLDER_TYPE = "ultrasonic_mics"


def find_audio(root_dir, hole_number, channel, folder_type, jam=False):
  hole = str(hole_number).zfill(5)
  for root, _, files in os.walk(root_dir):
    if folder_type.lower() not in root.lower():
      continue
    for f in files:
      if not f.lower().endswith(".wav"):
        continue
      if channel.lower() not in f.lower():
        continue
      if f"hole{hole}" not in f.lower():
        continue
      if jam:
        if "_jam.wav" in f.lower():
          return os.path.join(root, f)
      else:
        if "_jam.wav" not in f.lower():
          return os.path.join(root, f)
  return None


files_to_plot = [
    find_audio(ROOT_PATH, NORMAL_HOLE, INTERNAL_CHANNEL, FOLDER_TYPE, jam=False),
    find_audio(ROOT_PATH, JAM_HOLE, INTERNAL_CHANNEL, FOLDER_TYPE, jam=True),
    find_audio(ROOT_PATH, NORMAL_HOLE, EXTERNAL_CHANNEL, FOLDER_TYPE, jam=False),
    find_audio(ROOT_PATH, JAM_HOLE, EXTERNAL_CHANNEL, FOLDER_TYPE, jam=True),
]

if any(p is None for p in files_to_plot):
  raise FileNotFoundError("One or more target audio files could not be located.")


matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.labelweight": "bold",
})

fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.4))

labels = [
    "(a) Internal — Normal",
    "(b) Internal — jamming event",
    "(c) External — Normal",
    "(d) External — jamming event",
]

img = None

for i, path in enumerate(files_to_plot):
  row = i // 2
  col = i % 2
  ax = axes[row, col]

  y, sr = librosa.load(path, sr=None, mono=True)

  y = y - np.mean(y)
  peak = np.max(np.abs(y))
  if peak > 0:
    y = y / peak

  # FFT parameters for ultrasonic signals
  n_fft = 4096
  hop_length = 512

  D = librosa.amplitude_to_db(
      np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)),
      ref=np.max,
  )

  img = librosa.display.specshow(
      D,
      sr=sr,
      hop_length=hop_length,
      x_axis="time",
      y_axis="linear",
      ax=ax,
      cmap="magma",
      vmin=-80,
      vmax=0,
  )

  ax.set_title(labels[i], pad=5, fontweight="bold")

  # Full ultrasonic frequency range up to 96 kHz (Nyquist for 192 kHz)
  ax.set_ylim(0, 96000)

  if col == 0:
    ax.set_ylabel("Frequency (Hz)")
  else:
    ax.set_ylabel("")

  if row == 1:
    ax.set_xlabel("Time (s)")
  else:
    ax.set_xlabel("")

  ax.tick_params(
      axis="both", which="both", labelbottom=True, labelleft=(col == 0)
  )


plt.subplots_adjust(
    left=0.10,
    right=0.86,
    top=0.92,
    bottom=0.11,
    hspace=0.30,
    wspace=0.14,
)

cbar_ax = fig.add_axes([0.88, 0.12, 0.02, 0.78])
cbar = fig.colorbar(img, cax=cbar_ax, format="%+2.0f dB")
cbar.set_label("Intensity (dB)", weight="bold")

output_pdf = os.path.join(PLOTS_DIR, "spectrogram_normal_vs_jam_drill06.pdf")
output_png = os.path.join(PLOTS_DIR, "spectrogram_normal_vs_jam_drill06.png")

plt.savefig(output_pdf, dpi=400, bbox_inches="tight")
plt.savefig(output_png, dpi=300, bbox_inches="tight")
plt.close()

print(f"Spectrograms successfully generated in: {PLOTS_DIR}")