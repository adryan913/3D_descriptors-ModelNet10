from pathlib import Path
import numpy as np
import open3d as o3d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from cache import cache_memory
from data_loader import load_and_sample_off
from descriptors import deskryptor_d2, deskryptor_fpfh, deskryptor_pca
from main import load_dataset_and_extract_features

def train_models(dataset_path="ModelNet10", num_points=1000, k=5):
  # Wczytanie wyłącznie zbioru treningowego z cache
  train_data, _, categories = cache_memory(load_dataset_and_extract_features, dataset_path=dataset_path, num_points=num_points)
  y_train = np.array(train_data["labels"])
  models = {}

  # Trening k-NN dla pojedynczych deskryptorów
  for desc in ["PCA", "D2", "FPFH"]:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(np.array(train_data[desc]), y_train)
    models[desc] = knn

  # Trening k-NN dla fuzji cech ze standaryzacją
  scaler = StandardScaler()
  X_fused = np.hstack([train_data["PCA"], train_data["D2"], train_data["FPFH"]])
  X_fused_scaled = scaler.fit_transform(X_fused)
  knn_fused = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
  knn_fused.fit(X_fused_scaled, y_train)
  models["FUSION"] = knn_fused
  return models, scaler, categories

def predict_and_visualize(file_path, models, scaler, num_points=1000):
  file_path = Path(file_path)
  if not file_path.exists():
    print(f"BŁĄD: Plik '{file_path}' nie istnieje!")
    return

  # Przetworzenie siatki do chmury punktów
  pcd = load_and_sample_off(str(file_path), num_points=num_points)

  # Ekstrakcja cech badanego obiektu
  features = {
      "PCA": deskryptor_pca(pcd),
      "D2": deskryptor_d2(pcd),
      "FPFH": deskryptor_fpfh(pcd),
  }

  # Predykcje pojedynczych modeli
  preds = {
      desc: models[desc].predict(feat.reshape(1, -1))[0]
      for desc, feat in features.items()
  }

  # Predykcja fuzji z wykorzystaniem dopasowanego wcześniej scalera
  fused_raw = np.hstack(list(features.values())).reshape(1, -1)
  preds["FUSION"] = models["FUSION"].predict(scaler.transform(fused_raw))[0]

  # Odczyt etykiety z nazwy folderu
  true_label = (
      file_path.parent.parent.name
      if file_path.parent.name in ("test", "train")
      else file_path.parent.name
  )

  # Prezentacja wyników
  print(f"Obiekt: {file_path.name} | Prawdziwa kategoria: {true_label}")
  for desc, pred in preds.items():
    label = "FUZJA (skal.)" if desc == "FUSION" else desc
    print(f"Predykcja [{label}]: {pred}")

  # Wizualizacja 3D
  pcd.paint_uniform_color([0.15, 0.65, 0.25])
  o3d.visualization.draw_geometries(
      [pcd],
      window_name=f"{file_path.name} | True: {true_label} | Fuzja:"
      f" {preds['FUSION']}",
  )

if __name__ == "__main__":
  models, scaler, categories = train_models(
      dataset_path="ModelNet10", num_points=1000, k=5
  )
  test_object = "ModelNet10/monitor/test/monitor_0563.off"
  predict_and_visualize(test_object, models, scaler, num_points=1000)