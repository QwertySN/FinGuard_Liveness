"""
FinGuard Liveness | Streamlit-презентация и Live Demo.
Интерфейс для защиты CV-проекта: бизнес-контекст, EDA, метрики, демо.
"""
import io
import os

import librosa
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(page_title="FinGuard Liveness", page_icon="🛡️", layout="wide")

IMG_SIZE = 128
SAMPLE_RATE = 16000


class SpoofDetectorCNN(nn.Module):
    """Точная копия архитектуры из train_cnn.py.

    ВАЖНО: имена слоёв (features / classifier) обязаны совпадать
    с ключами сохранённого файла finguard_cnn.pth, иначе
    load_state_dict упадёт с ошибкой Missing/Unexpected keys.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


@st.cache_resource
def load_model():
    """Загружает обученные веса в архитектуру-двойник."""
    model = SpoofDetectorCNN()
    if os.path.exists("finguard_cnn.pth"):
        model.load_state_dict(
            torch.load("finguard_cnn.pth", map_location="cpu", weights_only=True)
        )
    model.eval()
    return model


def preprocess_audio(uploaded_file):
    """Превращает загруженное аудио в тензор Mel-спектрограммы 1x1x128x128."""
    raw = uploaded_file.getvalue()
    if len(raw) == 0:
        raise ValueError("файл пустой (0 байт) — это заглушка, а не аудио")
    audio_bytes = io.BytesIO(raw)
    y, sr = librosa.load(audio_bytes, sr=SAMPLE_RATE)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
    tensor = torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0)
    if tensor.shape[2] < IMG_SIZE:
        tensor = torch.nn.functional.pad(tensor, (0, IMG_SIZE - tensor.shape[2]))
    else:
        tensor = tensor[:, :, :IMG_SIZE]
    return tensor.unsqueeze(0)


def main():
    st.title("🛡️ FinGuard Liveness")
    st.caption("Audio Anti-Spoofing модуль для банковского KYC | AI Engineering • 2026")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏠 О проекте", "📊 EDA и данные", "🧠 Метрики", "💬 Live Demo"]
    )

    with tab1:
        st.header("Бизнес-проблема")
        st.write(
            "Мошенники клонируют голоса клиентов (TTS, voice cloning) и обходят "
            "системы видеоидентификации (KYC). Модуль FinGuard Liveness отличает "
            "живой голос от синтезированного и является CV-частью экосистемы "
            "FinGuard: при вердикте SPOOF NLP-ассистент блокирует сессию."
        )
        st.subheader("Архитектура")
        st.write("- Вход: аудио 16 кГц → Mel-спектрограмма 128×128 («рентген» голоса).")
        st.write("- Модель: 2D CNN, 3 сверточных блока + Dropout 0.5.")
        st.write("- Обоснование: спектрограмма сохраняет пространственные артефакты "
                 "вокодера, которые теряются в MFCC и сырой волне.")

    with tab2:
        st.header("Исследовательский анализ данных")
        st.write("Реальный датасет LibriSpeech (30 bonafide-файлов). Spoof-класс "
                 "получен аугментацией: pitch shift, time stretch, шум.")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("assets/eda_four_views.png"):
                st.image("assets/eda_four_views.png",
                         caption="Паспорт голоса: 4 взгляда на bonafide-файл")
            else:
                st.warning("Не найден assets/eda_four_views.png")
        with col2:
            if os.path.exists("assets/eda_corpus_hist.png"):
                st.image("assets/eda_corpus_hist.png",
                         caption="Распределения по корпусу")
            else:
                st.warning("Не найден assets/eda_corpus_hist.png")

    with tab3:
        st.header("Сравнение моделей (бонус +10%)")
        st.dataframe(
            {
                "Модель": ["Random Forest (baseline)", "2D CNN (основная)"],
                "Признаки": ["MFCC (13 coeffs)", "Mel-Spectrogram 128×128"],
                "Accuracy (test)": ["1.00", "1.00"],
                "Роль": ["Точка отсчёта", "Финальное решение"],
            },
            use_container_width=True,
        )
        st.info("Идеальные метрики ожидаемы для PoC на аугментации: трансформации "
                "оставляют явные спектральные следы. Следующий шаг — реальный "
                "ASVspoof 2021, где метрики станут реалистичными (85–95%).")
        if os.path.exists("assets/model_metrics.png"):
            st.image("assets/model_metrics.png", caption="Confusion Matrix и ROC-кривая")

    with tab4:
        st.header("Live Demo: детектор поддельного голоса")
        uploaded = st.file_uploader("Загрузите аудио (.wav или .flac)", type=["wav", "flac"])
        if st.button("🔍 Проверить голос"):
            try:
                with st.spinner("Анализирую спектральные признаки..."):
                    tensor = preprocess_audio(uploaded)
                    model = load_model()
                    with torch.no_grad():
                        logits = model(tensor)
                        probs = torch.softmax(logits, dim=1)[0]
                        pred = int(torch.argmax(logits, dim=1).item())
                label = "🟢 REAL (живой голос)" if pred == 0 else "🔴 SPOOF (подделка)"
                st.markdown(f"### Вердикт: {label}")
                st.progress(float(probs[pred]))
                st.write(f"Уверенность модели: **{probs[pred].item() * 100:.2f}%**")
            except Exception as exc:
                st.error(f"Не удалось прочитать аудио: {exc}")
                st.info("Проверьте, что файл не пустой и имеет формат .wav или .flac. "
                        "Пустые placeholder-файлы (0 байт) не являются аудио.")


if __name__ == "__main__":
    main()