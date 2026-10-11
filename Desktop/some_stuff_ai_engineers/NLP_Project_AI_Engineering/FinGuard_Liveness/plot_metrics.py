import os
import glob

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)
from torch.utils.data import Dataset

# ================= НАСТРОЙКИ =================
AUDIO_DIR = "data/audio/bonafide"
IMG_SIZE = 128
SAMPLE_RATE = 16000
ASSETS_DIR = "assets"


class SpoofDetectorCNN(nn.Module):
    """Архитектура-двойник из train_cnn.py."""

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


class EvalDataset(Dataset):
    """Датасет для оценки: real + spoof через аугментацию."""

    def __init__(self, audio_dir):
        self.files = sorted(glob.glob(os.path.join(audio_dir, "*.flac")))

    def __len__(self):
        return len(self.files) * 2

    def _to_tensor(self, y):
        spec = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-8)
        t = torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0)
        if t.shape[2] < IMG_SIZE:
            t = torch.nn.functional.pad(t, (0, IMG_SIZE - t.shape[2]))
        else:
            t = t[:, :, :IMG_SIZE]
        return t

    def __getitem__(self, idx):
        real_idx = idx // 2
        is_spoof = idx % 2
        y, _ = librosa.load(self.files[real_idx], sr=SAMPLE_RATE)
        if is_spoof:
            y = librosa.effects.pitch_shift(y=y, sr=SAMPLE_RATE, n_steps=4.0)
            y = librosa.effects.time_stretch(y=y, rate=1.1)
            y = y + np.random.normal(0, 0.005, len(y))
        return self._to_tensor(y), 1 if is_spoof else 0


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    print("Загружаем модель...")
    model = SpoofDetectorCNN()
    model.load_state_dict(
        torch.load("finguard_cnn.pth", map_location="cpu", weights_only=True)
    )
    model.eval()

    print("Готовим тестовую выборку...")
    dataset = EvalDataset(AUDIO_DIR)
    # Берём 20% данных как test (те же пропорции, что при обучении)
    test_size = int(0.2 * len(dataset))
    _, test_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - test_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for tensor, label in test_ds:
            batch = tensor.unsqueeze(0)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(logits, dim=1).item())
            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(probs.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    print("\nОтчёт по метрикам:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Spoof"]))

    # ================= РИСУЕМ =================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Real", "Spoof"],
        cmap="Blues",
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix\n(матрица ошибок)")

    # Убрали color="darkorange" из вызова from_predictions
    roc_display = RocCurveDisplay.from_predictions(
        y_true, y_prob[:, 1],
        ax=axes[1],
    )
    # Теперь красим линию вручную (опционально)
    roc_display.line_.set_color("darkorange")
    roc_display.line_.set_linewidth(2)
    
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random guess")
    axes[1].set_title("ROC Curve\n(качество классификации)")
    axes[1].legend(loc="lower right")

    plt.suptitle("FinGuard Liveness — метрики 2D CNN", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(ASSETS_DIR, "model_metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()