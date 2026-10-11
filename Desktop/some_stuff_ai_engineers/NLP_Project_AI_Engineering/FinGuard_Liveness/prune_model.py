"""
FinGuard Liveness: Model Pruning (бонус +5% за оптимизацию).
Автономная версия: архитектура CNN определена внутри файла.
"""
import torch
import torch.nn.utils.prune as prune


class SpoofDetectorCNN(torch.nn.Module):
    """2D CNN для Mel-спектрограмм (копия архитектуры из train_cnn.py)."""

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 16 * 16, 128), torch.nn.ReLU(),
            torch.nn.Dropout(0.5), torch.nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def count_weights(model):
    """Возвращает (всего весов, ненулевых) для Conv2d и Linear."""
    total, nonzero = 0, 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            total += module.weight.numel()
            nonzero += torch.count_nonzero(module.weight).item()
    return total, nonzero


def main():
    print("Загружаем обученную модель...")
    model = SpoofDetectorCNN()
    model.load_state_dict(
        torch.load("finguard_cnn.pth", map_location="cpu", weights_only=True)
    )

    total, nonzero = count_weights(model)
    print(f"До прунинга: ненулевых {nonzero}/{total}")

    print("Применяем L1 Unstructured Pruning (обнуление 30% весов)...")
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=0.3)
            prune.remove(module, "weight")

    total, nonzero = count_weights(model)
    print(f"После прунинга: ненулевых {nonzero}/{total}")
    print(f"Sparsity модели: {100 * (1 - nonzero / total):.1f}% связей обнулено")

    torch.save(model.state_dict(), "finguard_cnn_pruned.pth")
    print("Обрезанная модель сохранена: finguard_cnn_pruned.pth")


if __name__ == "__main__":
    main()