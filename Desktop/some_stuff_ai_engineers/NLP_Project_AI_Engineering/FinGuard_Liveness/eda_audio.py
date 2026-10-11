import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Берём первый скачанный файл
audio_path = "data/audio/bonafide/librispeech_000.flac"

if not os.path.exists(audio_path):
    print(f"Файл {audio_path} не найден. Убедись, что скачивание прошло успешно.")
else:
    print(f"Загружаем {audio_path}...")
    
    # Загружаем аудио (librosa сама сконвертирует FLAC в numpy-массив)
    y, sr = librosa.load(audio_path, sr=None)
    
    # Создаём Mel-спектрограмму (то, как "видит" звук нейросеть)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    # Рисуем
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.title('Mel Spectrogram - Real Voice (LibriSpeech)')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    
    # Сохраняем картинку в папку с проектом
    save_path = "eda_spectrogram.png"
    plt.savefig(save_path)
    print(f"Спектрограмма сохранена как {save_path}")
    print("Готово!")