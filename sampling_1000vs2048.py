import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from cache import cache_memory
from main import load_dataset_and_extract_features

def compare_sampling():
  descriptors = ["PCA", "D2", "FPFH"]
  train_1000, test_1000, _ = cache_memory(load_dataset_and_extract_features, dataset_path="ModelNet10", num_points=1000)
  y_train_1000 = np.array(train_1000["labels"])
  y_test_1000 = np.array(test_1000["labels"])
  acc_1000 = []
  for desc in descriptors:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(train_1000[desc]), y_train_1000)
    y_pred = knn.predict(np.array(test_1000[desc]))
    score = accuracy_score(y_test_1000, y_pred) * 100
    acc_1000.append(score)

  train_2048, test_2048, _ = cache_memory(load_dataset_and_extract_features, dataset_path="ModelNet10", num_points=2048)
  y_train_2048 = np.array(train_2048["labels"])
  y_test_2048 = np.array(test_2048["labels"])
  acc_2048 = []
  for desc in descriptors:
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(train_2048[desc]), y_train_2048)
    y_pred = knn.predict(np.array(test_2048[desc]))
    score = accuracy_score(y_test_2048, y_pred) * 100
    acc_2048.append(score)

  plt.figure(figsize=(8, 6))
  plt.bar([0, 1, 2], acc_1000, width=0.30, label="1000 punktów")
  plt.bar([0.3, 1.3, 2.3], acc_2048, width=0.30, label="2048 punktów")
  plt.xticks([0.15, 1.15, 2.15], descriptors)
  plt.ylabel("Dokładność (%)")
  plt.title("Porównanie dokładności próbek: 1000 vs 2048 punktów")
  plt.legend()
  plt.grid()
  plt.savefig("results/sampling_1000vs2048.png")
  plt.close()

if __name__ == "__main__":
  compare_sampling()