"""FinGuard Liveness | EDA Visual: 4 взгляда на голос + статистики корпуса."""
import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

AUDIO_DIR = "data/audio/bonafide"
files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.flac")))
path = files[0]
y, sr = librosa.load(path, sr=None)

# ---------- ФИГУРА 1: четыре взгляда на один файл ----------
fig, axes = plt.subplots(2, 2, figsize=(13, 7))

# 1. Волна — самый понятный взгляд
librosa.display.waveshow(y, sr=sr, ax=axes[0, 0], color='#2E86AB')
axes[0, 0].set_title('1. Волна: голос как колебание')
axes[0, 0].set_xlabel('время (сек)')

# 2. Мел-спектрограмма (оставляем, она уже в отчёте)
S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
im = librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=axes[0, 1])
axes[0, 1].set_title('2. Спектрограмма: «рентген» голоса')
fig.colorbar(im, ax=axes[0, 1], format='%+02.0f dB')

# 3. Средний спектр: где живёт энергия
D = np.abs(librosa.stft(y))
freqs = librosa.fft_frequencies(sr=sr)
mean_db = librosa.amplitude_to_db(D.mean(axis=1), ref=D.mean(axis=1).max())
axes[1, 0].plot(freqs / 1000, mean_db, color='#A23B72')
axes[1, 0].set_xlim(0, 8)
axes[1, 0].set_title('3. Средний спектр: «диета» голоса')
axes[1, 0].set_xlabel('частота (кГц)')
axes[1, 0].set_ylabel('дБ')
axes[1, 0].grid(alpha=0.3)

# 4. Тон F0: мелодия речи
f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
times = librosa.times_like(f0, sr=sr)
axes[1, 1].plot(times, f0, color='#F18F01', lw=1.5)
axes[1, 1].set_title('4. Тон (F0): «плавающая» интонация')
axes[1, 1].set_xlabel('время (сек)')
axes[1, 1].set_ylabel('Гц')
axes[1, 1].grid(alpha=0.3)

fig.suptitle('EDA bonafide-файла: четыре взгляда на один голос')
fig.tight_layout()
fig.savefig('eda_four_views.png', dpi=150)
print('Сохранено: eda_four_views.png')

# ---------- ФИГУРА 2: паспорт корпуса (30 файлов) ----------
durs, cents = [], []
for p in files:
    yy, srr = librosa.load(p, sr=None)
    durs.append(len(yy) / srr)
    cents.append(float(librosa.feature.spectral_centroid(y=yy, sr=srr)[0].mean()))

fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
ax2[0].hist(durs, bins=10, color='#2E86AB', edgecolor='white')
ax2[0].set_title('Длительность записей (сек)')
ax2[1].hist(cents, bins=10, color='#A23B72', edgecolor='white')
ax2[1].set_title('Спектральный центроид, Гц («яркость» голоса)')
fig2.suptitle('Паспорт корпуса: распределения по 30 bonafide-файлам')
fig2.tight_layout()
fig2.savefig('eda_corpus_hist.png', dpi=150)
print('Сохранено: eda_corpus_hist.png')