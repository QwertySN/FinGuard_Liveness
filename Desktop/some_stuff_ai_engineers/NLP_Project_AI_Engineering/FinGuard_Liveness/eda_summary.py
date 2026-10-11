"""FinGuard Liveness | EDA Summary: статистики + сетка спектрограмм."""
import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

AUDIO_DIR = "data/audio/bonafide"
files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.flac")))

if not files:
    raise SystemExit("Нет .flac файлов в data/audio/bonafide/")

durs, srs, cents, zcrs, rmss = [], [], [], [], []
for path in files:
    y, sr = librosa.load(path, sr=None)
    durs.append(len(y) / sr)
    srs.append(sr)
    cents.append(float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()))
    zcrs.append(float(librosa.feature.zero_crossing_rate(y)[0].mean()))
    rmss.append(float(librosa.feature.rms(y=y)[0].mean()))

print("=" * 60)
print(f"EDA Summary | n = {len(files)} файлов")
print("=" * 60)
print(f"Sampling rate     : {int(np.mean(srs))} Hz")
print(f"Длительность      : mean {np.mean(durs):.2f} c | min {np.min(durs):.2f} | max {np.max(durs):.2f}")
print(f"Spectral centroid : mean {np.mean(cents):.0f} Hz | std {np.std(cents):.0f}")
print(f"ZCR               : mean {np.mean(zcrs):.4f} | std {np.std(zcrs):.4f}")
print(f"RMS energy        : mean {np.mean(rmss):.4f} | std {np.std(rmss):.4f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for ax, path in zip(axes.flat, files[:4]):
    y, sr = librosa.load(path, sr=None)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel", cmap="magma", ax=ax)
    ax.set_title(os.path.basename(path), fontsize=9)
    ax.label_outer()
fig.suptitle("Bonafide-голоса (LibriSpeech subset): mel-спектрограммы")
fig.tight_layout()
fig.savefig("eda_summary.png", dpi=150)
print("Сохранено: eda_summary.png")