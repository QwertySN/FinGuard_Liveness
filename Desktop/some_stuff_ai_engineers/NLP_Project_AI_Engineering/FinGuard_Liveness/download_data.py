"""pip install datasets huggingface_hub torch torchvision torchaudio opencv-python pan
FinGuard Liveness: Data Downloader
Скачивает подмножества реальных датасетов через HuggingFace API.
"""
import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import shutil
from tqdm import tqdm

# Создаём структуру папок
DATA_DIR = Path("data")
(DATA_DIR / "video" / "real").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "video" / "fake").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "audio" / "bonafide").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "audio" / "spoof").mkdir(parents=True, exist_ok=True)

def download_celeb_df():
    """
    Celeb-DF v2: реальные видео знаменитостей + deepfake-атаки.
    Размер полного датасета: ~10 GB. Скачаем только 20 видео для старта.
    """
    print("\n🎥 Скачиваем Celeb-DF (подмножество для быстрого старта)...")
    
    # HuggingFace-хостинг Celeb-DF (от сообщества)
    # Скачаем только несколько файлов через API
    repo_id = "Reachout/Celeb-DF-v2"
    
    try:
        # Получаем список файлов в репозитории
        from huggingface_hub import list_repo_files
        files = list_repo_files(repo_id)
        
        # Берём первые 10 видео из real и 10 из fake
        real_files = [f for f in files if 'Celeb-real' in f and f.endswith('.mp4')][:10]
        fake_files = [f for f in files if 'Celeb-synthesis' in f and f.endswith('.mp4')][:10]
        
        print(f"  Найдено: {len(real_files)} real + {len(fake_files)} fake видео")
        
        # Скачиваем real видео
        for fname in tqdm(real_files, desc="Real videos"):
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="dataset",
                local_dir=str(DATA_DIR / "video" / "real")
            )
        
        # Скачиваем fake видео
        for fname in tqdm(fake_files, desc="Fake videos"):
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="dataset",
                local_dir=str(DATA_DIR / "video" / "fake")
            )
        
        print("✅ Видео скачаны успешно!")
        
    except Exception as e:
        print(f"⚠️ Не удалось скачать Celeb-DF: {e}")
        print("  Альтернатива: используем FaceForensics++ через HuggingFace datasets")
        download_faceforensics()

def download_faceforensics():
    """
    FaceForensics++: 1000 оригинальных видео + 4 типа манипуляций.
    Загружаем через datasets библиотеку - самый надёжный способ.
    """
    print("\n🎥 Скачиваем FaceForensics++ (subset)...")
    
    try:
        # Загружаем только маленький subset для теста
        ds = load_dataset(
            "deepmind/faceforensics_pp",
            split="train[:20]",  # Первые 20 примеров
            trust_remote_code=True
        )
        
        for i, item in enumerate(tqdm(ds, desc="Processing videos")):
            # Сохраняем видео
            if 'video' in item:
                video_path = DATA_DIR / "video" / "real" / f"real_{i}.mp4"
                with open(video_path, 'wb') as f:
                    f.write(item['video']['bytes'])
        
        print("✅ FaceForensics++ subset скачан!")
        
    except Exception as e:
        print(f"⚠️ FaceForensics++ недоступен: {e}")
        print("  Создаём тестовые плейсхолдеры для разработки...")
        create_placeholders()

def download_asvspoof():
    """
    ASVspoof: реальные и поддельные голоса.
    Используем HuggingFace datasets - самый надёжный источник.
    """
    print("\n🎤 Скачиваем ASVspoof (subset)...")
    
    try:
        # Пытаемся загрузить ASVspoof
        ds = load_dataset(
            "ProgramComputer/asvspoof",
            "LA",  # Logical Access
            split="train[:50]",  # Первые 50 примеров
            trust_remote_code=True
        )
        
        bonafide_count = 0
        spoof_count = 0
        
        for item in tqdm(ds, desc="Processing audio"):
            # Определяем тип: реальный или поддельный
            is_spoof = item.get('label', 0) == 1 or 'spoof' in str(item.get('system_id', '')).lower()
            
            target_dir = DATA_DIR / "audio" / ("spoof" if is_spoof else "bonafide")
            prefix = "spoof" if is_spoof else "bonafide"
            count = spoof_count if is_spoof else bonafide_count
            
            audio_path = target_dir / f"{prefix}_{count}.wav"
            
            # Сохраняем аудио
            if 'audio' in item and 'bytes' in item['audio']:
                with open(audio_path, 'wb') as f:
                    f.write(item['audio']['bytes'])
                
                if is_spoof:
                    spoof_count += 1
                else:
                    bonafide_count += 1
        
        print(f"✅ Аудио скачаны: {bonafide_count} bonafide + {spoof_count} spoof")
        
    except Exception as e:
        print(f"⚠️ ASVspoof недоступен: {e}")
        print("  Пробуем LibriSpeech (чистые голоса для baseline)...")
        download_librispeech()

def download_librispeech():
    """
    LibriSpeech: 1000 часов чистой английской речи.
    Идеально как baseline для "реальных" голосов.
    """
    print("\n🎤 Скачиваем LibriSpeech (test subset)...")
    
    try:
        ds = load_dataset(
            "openslr/librispeech_asr",
            split="test.clean[:30]",
            trust_remote_code=True
        )
        
        for i, item in enumerate(tqdm(ds, desc="Processing audio")):
            audio_path = DATA_DIR / "audio" / "bonafide" / f"librispeech_{i}.wav"
            # Сохраняем через torchaudio или просто bytes
            if 'audio' in item and 'bytes' in item['audio']:
                with open(audio_path, 'wb') as f:
                    f.write(item['audio']['bytes'])
        
        print("✅ LibriSpeech subset скачан!")
        
    except Exception as e:
        print(f"⚠️ LibriSpeech недоступен: {e}")
        create_placeholders()

def create_placeholders():
    """
    Создаём файлы-плейсхолдеры, чтобы можно было разрабатывать код
    даже если датасеты временно недоступны.
    """
    print("\n📦 Создаём placeholder файлы для разработки...")
    
    # Создаём пустые файлы как маркеры
    for i in range(10):
        (DATA_DIR / "video" / "real" / f"placeholder_real_{i}.mp4").touch()
        (DATA_DIR / "video" / "fake" / f"placeholder_fake_{i}.mp4").touch()
        (DATA_DIR / "audio" / "bonafide" / f"placeholder_bonafide_{i}.wav").touch()
        (DATA_DIR / "audio" / "spoof" / f"placeholder_spoof_{i}.wav").touch()
    
    print("✅ Плейсхолдеры созданы. Структура готова к разработке.")

if __name__ == "__main__":
    print("=" * 60)
    print("🛡️  FinGuard Liveness - Data Downloader")
    print("=" * 60)
    
    # Качаем видео
    download_celeb_df()
    
    # Качаем аудио
    download_asvspoof()
    
    print("\n" + "=" * 60)
    print("🎉 Скачивание завершено!")
    print("=" * 60)
    print(f"\n📂 Структура данных:")
    print(f"  Видео (real): {len(list((DATA_DIR / 'video/real').glob('*')))} файлов")
    print(f"  Видео (fake): {len(list((DATA_DIR / 'video/fake').glob('*')))} файлов")
    print(f"  Аудио (bonafide): {len(list((DATA_DIR / 'audio/bonafide').glob('*')))} файлов")
    print(f"  Аудио (spoof): {len(list((DATA_DIR / 'audio/spoof').glob('*')))} файлов")