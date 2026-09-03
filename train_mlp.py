from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from cache import cache_memory
from main import load_dataset_and_extract_features

def train_mlp_classifier(num_points=1000):
  print("Rozpoczynanie procesu uczenia sieci neuronowej MLP...")

  # 1. Wczytanie danych z cache
  train_data, test_data, categories = cache_memory(
      load_dataset_and_extract_features,
      dataset_path="ModelNet10",
      num_points=num_points,
  )

  y_train = np.array(train_data["labels"])
  y_test = np.array(test_data["labels"])

  # 2. Sklejenie cech w stosunku 1:1 (66 wymiarów)
  X_train_fused = np.hstack(
      [train_data["PCA"], train_data["D2"], train_data["FPFH"]]
  )
  X_test_fused = np.hstack(
      [test_data["PCA"], test_data["D2"], test_data["FPFH"]]
  )

  # 3. Standaryzacja wektora cech
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train_fused)
  X_test_scaled = scaler.transform(X_test_fused)

  # 4. Architektura sieci MLP
  mlp = MLPClassifier(
      hidden_layer_sizes=(256, 128, 64),
      activation="relu",
      solver="adam",
      alpha=0.001,
      max_iter=1000,
      random_state=42,
      early_stopping=True,
  )

  print("Uczenie sieci neuronowej MLP na fuzji deskryptorów (66 wymiarów)...")
  mlp.fit(X_train_scaled, y_train)

  # 5. Ewaluacja i raport
  y_pred = mlp.predict(X_test_scaled)
  acc = accuracy_score(y_test, y_pred) * 100

  print(f"DOKŁADNOŚĆ SIECI MLP (Fuzja cech): {acc:.2f}%")
  print("Raport klasyfikacji:")
  print(classification_report(y_test, y_pred))

  results_dir = Path("results")
  results_dir.mkdir(exist_ok=True)

  # 6. Zapis wykresu krzywej uczenia (Loss)
  plt.figure(figsize=(8, 5))
  plt.plot(
      mlp.loss_curve_,
      color="indigo",
      linewidth=2,
      label="Funkcja straty (Loss)",
  )
  plt.title("Krzywa uczenia sieci MLP na fuzji deskryptorów 3D")
  plt.xlabel("Epoka")
  plt.ylabel("Wartość funkcji straty (Loss)")
  plt.grid(True, linestyle="--", alpha=0.7)
  plt.legend()
  plt.tight_layout()

  loss_save_path = results_dir / "mlp_learning_curve.png"
  plt.savefig(loss_save_path, dpi=300)
  plt.close()
  print(f"Krzywa uczenia zapisana w: {loss_save_path}")

  # 7. Generowanie i zapis Macierzy Pomyłek (Confusion Matrix)
  cm = confusion_matrix(y_test, y_pred, labels=categories)

  plt.figure(figsize=(10, 8))
  sns.heatmap(
      cm,
      annot=True,
      fmt="d",
      cmap="Blues",
      xticklabels=categories,
      yticklabels=categories,
  )
  plt.title(f"Macierz pomyłek - MLP (Fuzja cech, Accuracy: {acc:.2f}%)")
  plt.xlabel("Przewidywana kategoria (Predicted)")
  plt.ylabel("Prawdziwa kategoria (True)")
  plt.tight_layout()

  cm_save_path = results_dir / "confusion_matrix_mlp.png"
  plt.savefig(cm_save_path, dpi=300)
  plt.close()
  print(f"Macierz pomyłek zapisana w: {cm_save_path}")


if __name__ == "__main__":
  train_mlp_classifier(num_points=1000)