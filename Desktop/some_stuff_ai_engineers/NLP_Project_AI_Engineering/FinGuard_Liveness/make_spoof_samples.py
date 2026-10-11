"""
FinGuard Liveness: генерация spoof-образцов для Live Demo.
Берёт 10 реальных голосов и сохраняет их аугментированные копии
(pitch shift + time stretch + шум) в data/audio/spoof как .flac.
"""
import glob
import os

import librosa
import numpy as np
import soundfile as sf

SRC_DIR = "data/audio/bonafide"
DST_DIR = "data/audio/spoof"
SAMPLE_RATE = 16000


def make_spoof(y, sr):
    """Применяет TTS-подобные трансформации к аудио."""
    y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=4.0)
    y = librosa.effects.time_stretch(y=y, rate=1.1)
    return y + np.random.normal(0, 0.005, len(y))


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.flac")))[:10]
    if not files:
        raise SystemExit("Нет .flac в data/audio/bonafide/")
    for i, path in enumerate(files):
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        sf.write(os.path.join(DST_DIR, f"spoof_{i:03d}.flac"), make_spoof(y, sr), sr)
        print(f"Сохранён spoof_{i:03d}.flac")
    print(f"Готово: {len(files)} spoof-файлов в {DST_DIR}")


if __name__ == "__main__":
    main()