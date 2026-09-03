"""
Zoptymalizowana sieć PointNet do klasyfikacji ModelNet10.
Zawiera:
 - Poprawny Input T-Net z regularyzacją ortogonalności
 - Pełny globalny wektor cech (1024-d)
 - Augmentację chmur punktów w locie (rotacja Z, jittering)
 - Harmonogram uczenia (StepLR)
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_loader import load_and_sample_off

RAW_CACHE_DIR = Path("cache_raw")

# WCZYTYWANIE I CACHE
def load_raw_pointclouds(dataset_path="ModelNet10", num_points=1000):
    data_dir = Path(dataset_path)
    categories = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    train_points, train_labels = [], []
    test_points, test_labels = [], []

    for split in ["train", "test"]:
        target_points = train_points if split == "train" else test_points
        target_labels = train_labels if split == "train" else test_labels

        for category in categories:
            cat_dir = data_dir / category / split
            if not cat_dir.exists():
                continue
            off_files = list(cat_dir.glob("*.off"))
            print(f"[{split}] {category}: {len(off_files)} plików")

            for file_path in off_files:
                try:
                    pcd = load_and_sample_off(str(file_path), num_points=num_points)
                    pts = np.asarray(pcd.points, dtype=np.float32)
                    target_points.append(pts)
                    target_labels.append(category)
                except Exception as e:
                    print(f"Błąd dla pliku {file_path}: {e}")

    return (
        np.array(train_points, dtype=np.float32),
        np.array(train_labels),
        np.array(test_points, dtype=np.float32),
        np.array(test_labels),
        categories,
    )


def load_raw_pointclouds_cached(dataset_path="ModelNet10", num_points=1000):
    RAW_CACHE_DIR.mkdir(exist_ok=True)
    suffix = f"{num_points}p"
    files = {
        "train_x": RAW_CACHE_DIR / f"train_points_{suffix}.npy",
        "train_y": RAW_CACHE_DIR / f"train_labels_{suffix}.npy",
        "test_x": RAW_CACHE_DIR / f"test_points_{suffix}.npy",
        "test_y": RAW_CACHE_DIR / f"test_labels_{suffix}.npy",
        "categories": RAW_CACHE_DIR / f"categories_{suffix}.npy",
    }

    if all(f.exists() for f in files.values()):
        print("Wczytywanie surowych chmur punktów z cache...")
        return (
            np.load(files["train_x"]),
            np.load(files["train_y"]),
            np.load(files["test_x"]),
            np.load(files["test_y"]),
            np.load(files["categories"]).tolist(),
        )

    print("Brak cache - pierwsza ekstrakcja...")
    train_x, train_y, test_x, test_y, categories = load_raw_pointclouds(dataset_path, num_points)

    np.save(files["train_x"], train_x)
    np.save(files["train_y"], train_y)
    np.save(files["test_x"], test_x)
    np.save(files["test_y"], test_y)
    np.save(files["categories"], np.array(categories))
    print("Zapisano cache surowych chmur punktów.")

    return train_x, train_y, test_x, test_y, categories


# DATASET I AUGMENTACJA
class PointCloudDataset(Dataset):
    def __init__(self, points, labels, label_to_idx, augment=False):
        # points shape wejściowe: (N_samples, num_points, 3)
        self.points = points
        self.labels = [label_to_idx[l] for l in labels]
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def _augment(self, pts):
        # 1. Losowa rotacja wokół osi pionowej (Z)
        theta = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,             0,              1]
        ], dtype=np.float32)
        pts = np.dot(pts, rot_matrix)

        # 2. Jittering (drobny szum gaussowski)
        noise = np.clip(np.random.normal(0, 0.02, size=pts.shape), -0.05, 0.05).astype(np.float32)
        return pts + noise

    def __getitem__(self, idx):
        pts = self.points[idx]
        if self.augment:
            pts = self._augment(pts)

        # Transpozycja na (3, num_points) pod Conv1d
        pts_tensor = torch.from_numpy(pts).float().transpose(0, 1)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return pts_tensor, label_tensor

# ARCHITEKTURA POINTNET
class TNet3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 9),
        )

    def forward(self, x):
        batch_size = x.size(0)
        feat = self.conv(x)
        global_feat = torch.max(feat, dim=2)[0]
        matrix = self.fc(global_feat)
        identity = torch.eye(3, device=x.device).flatten().unsqueeze(0)
        return (matrix + identity).view(batch_size, 3, 3)


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=10, use_tnet=True):
        super().__init__()
        self.use_tnet = use_tnet
        if use_tnet:
            self.tnet = TNet3d()

        # Ekstrakcja cech lokalnych
        self.feat_conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        # Ekstrakcja cech globalnych
        self.feat_conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(),
        )

        # Klasyfikator MLP
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        trans = None
        if self.use_tnet:
            trans = self.tnet(x)
            x = torch.bmm(trans, x)

        x = self.feat_conv1(x)
        x = self.feat_conv2(x)
        x = torch.max(x, dim=2)[0]  # Global Max Pooling -> (B, 1024)
        return self.classifier(x), trans


def feature_transform_regularizer(trans):
    """Kara za nieliniowość/brak ortogonalności transformacji T-Neta: ||I - AA^T||^2"""
    if trans is None:
        return 0.0
    d = trans.size(1)
    identity = torch.eye(d, device=trans.device).unsqueeze(0).repeat(trans.size(0), 1, 1)
    loss = torch.norm(identity - torch.bmm(trans, trans.transpose(2, 1)), dim=(1, 2)).mean()
    return loss

# 4. TRENING I EWALUACJA
def train_and_evaluate_pointnet(num_points=1000, epochs=50, batch_size=32, lr=1e-3, use_tnet=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_x, train_y, test_x, test_y, categories = load_raw_pointclouds_cached(
        dataset_path="ModelNet10", num_points=num_points
    )

    label_to_idx = {c: i for i, c in enumerate(categories)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    train_ds = PointCloudDataset(train_x, train_y, label_to_idx, augment=True)
    test_ds = PointCloudDataset(test_x, test_y, label_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    model = PointNetClassifier(num_classes=len(categories), use_tnet=use_tnet).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs, test_accs = [], [], []

    print(f"\nTrening przez {epochs} epok ({len(train_ds)} obiektów)...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for points, labels in train_loader:
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, trans = model(points)

            cls_loss = criterion(outputs, labels)
            reg_loss = feature_transform_regularizer(trans) * 0.001
            loss = cls_loss + reg_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * points.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += points.size(0)

        scheduler.step()

        train_loss = epoch_loss / total
        train_acc = correct / total * 100
        test_acc = _evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"  Epoka {epoch:2d}/{epochs} | lr={scheduler.get_last_lr()[0]:.5f} | "
              f"loss={train_loss:.4f} | train_acc={train_acc:5.2f}% | "
              f"test_acc={test_acc:5.2f}% | {time.time()-t0:.1f}s")

    y_true, y_pred = _predict_all(model, test_loader, device)
    y_true_labels = [idx_to_label[i] for i in y_true]
    y_pred_labels = [idx_to_label[i] for i in y_pred]

    final_acc = accuracy_score(y_true_labels, y_pred_labels) * 100
    print(f"\nFinalna dokładność PointNet: {final_acc:.2f}%")
    print(classification_report(y_true_labels, y_pred_labels))

    _plot_training_curve(train_losses, train_accs, test_accs, num_points)
    _plot_confusion_matrix(y_true_labels, y_pred_labels, categories, final_acc, num_points)

    return final_acc, model


def _evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for points, labels in loader:
            points, labels = points.to(device), labels.to(device)
            outputs, _ = model(points)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += points.size(0)
    return correct / total * 100


def _predict_all(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            outputs, _ = model(points)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return y_true, y_pred


# 5. WYKRESY
def _plot_training_curve(losses, train_accs, test_accs, num_points):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(losses, color="indigo", linewidth=2)
    axes[0].set_title("Krzywa uczenia (Loss)")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(train_accs, label="Train", color="tab:blue")
    axes[1].plot(test_accs, label="Test", color="tab:red")
    axes[1].set_title("Dokładność (%)")
    axes[1].set_xlabel("Epoka")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/pointnet_training_curve.png", dpi=300)
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, categories, acc, num_points):
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=categories, yticklabels=categories)
    plt.title(f"Confusion Matrix - PointNet, Acc: {acc:.2f}% (N={num_points})")
    plt.xlabel("Predykcja")
    plt.ylabel("Prawda")
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/confusion_matrix_pointnet.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    train_and_evaluate_pointnet(num_points=1000, epochs=50, batch_size=64, lr=1e-3, use_tnet=True)