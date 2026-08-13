import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from main import load_dataset_and_extract_features


def train_and_save_mlp():
    print("🧠 Rozpoczynanie procesu uczenia sieci neuronowej MLP...")

    # Utworzenie folderu na wyniki
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)

    # 1. Wczytanie i ekstrakcja cech z ModelNet10
    train_data, test_data, categories = load_dataset_and_extract_features("ModelNet10", num_points=1000)

    y_train = np.array(train_data['labels'])
    y_test = np.array(test_data['labels'])

    # Wagi cech (Wzmocnienie FPFH dla lepszej detekcji detali)
    w_pca, w_d2, w_fpfh = 1.0, 1.0, 2.0

    X_train_fused = np.hstack([
        np.array(train_data['PCA']) * w_pca,
        np.array(train_data['D2']) * w_d2,
        np.array(train_data['FPFH']) * w_fpfh
    ])

    X_test_fused = np.hstack([
        np.array(test_data['PCA']) * w_pca,
        np.array(test_data['D2']) * w_d2,
        np.array(test_data['FPFH']) * w_fpfh
    ])

    # 2. Standaryzacja
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fused)
    X_test_scaled = scaler.transform(X_test_fused)

    # 3. Architektura sieci MLP (256 -> 128 -> 64 neurony)
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )

    print("🚀 Uczenie sieci neuronowej (Feature Fusion: 66 wymiarów)...")
    mlp.fit(X_train_scaled, y_train)

    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100

    print(f"\n🔥 DOKŁADNOŚĆ SIECI Z JEDNOSTKOWYM WAŻENIEM FPFH: {acc:.2f}% 🔥\n")

    # 4. Zapisanie wyuczonego modelu i scalera na dysk
    joblib.dump(mlp, results_dir / 'mlp_model.joblib')
    joblib.dump(scaler, results_dir / 'mlp_scaler.joblib')
    print(f"💾 Zapisano model i scaler w folderze: {results_dir}/")

    # 5. Generowanie wykresu uczenia (Loss Curve)
    plt.figure(figsize=(8, 5))
    plt.plot(mlp.loss_curve_, color='indigo', linewidth=2, label='Funkcja straty (Loss)')
    plt.title('Krzywa uczenia sieci MLP na sklejonych deskryptorach 3D')
    plt.xlabel('Epoka')
    plt.ylabel('Błąd (Loss)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / 'mlp_learning_curve.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    train_and_save_mlp()