import os
from datasets import load_dataset, Audio
from tqdm import tqdm

# Создаём папку
os.makedirs("data/audio/bonafide", exist_ok=True)
os.makedirs("data/audio/spoof", exist_ok=True)

print("Быстрая загрузка подмножества LibriSpeech (30 файлов)...")

# streaming=True - данные идут потоком из интернета
ds = load_dataset(
    "openslr/librispeech_asr", 
    split="test.clean", 
    streaming=True
)

# МАГИЯ: Отключаем автоматическое декодирование аудио.
# Это убирает требование установить torchcodec и ускоряет процесс в разы.
ds = ds.cast_column("audio", Audio(decode=False))

count = 0
# .take(30) берет ровно 30 первых элементов из бесконечного стрима
for item in tqdm(ds.take(30), total=30, desc="Сохраняем аудио"):
    # Достаем сырые байты файла
    audio_bytes = item['audio']['bytes']
    
    # LibriSpeech хранит аудио в формате FLAC, поэтому сохраняем с правильным расширением
    save_path = f"data/audio/bonafide/librispeech_{count:03d}.flac"
    
    # Просто записываем байты на диск (мгновенно!)
    with open(save_path, "wb") as f:
        f.write(audio_bytes)
        
    count += 1

print(f"\nГотово! {count} файлов сохранены.")
print(f"Путь: {os.path.abspath('data/audio/bonafide')}")