from pathlib import Path
import numpy as np
CACHE_DIR = Path("cache")

def cache_memory(load_data_fn, dataset_path="ModelNet10", num_points=1000):
  """Sprawdza, czy wyliczone cechy znajdują się już w plikach pamięci podręcznej (.npy).
  Jeśli istnieją -> wczytuje je w ułamku sekundy.
  Jeśli nie -> wykonuje ekstrakcję cech i zapisuje je na przyszłość.
  """
  CACHE_DIR.mkdir(exist_ok=True)

  # Połączono przypisanie listy
  files_to_check = [
      CACHE_DIR / f"train_pca_{num_points}p.npy",
      CACHE_DIR / f"train_d2_{num_points}p.npy",
      CACHE_DIR / f"train_fpfh_{num_points}p.npy",
      CACHE_DIR / f"train_labels_{num_points}p.npy",
      CACHE_DIR / f"test_pca_{num_points}p.npy",
      CACHE_DIR / f"test_d2_{num_points}p.npy",
      CACHE_DIR / f"test_fpfh_{num_points}p.npy",
      CACHE_DIR / f"test_labels_{num_points}p.npy",
  ]

  # Sprawdzenie czy wszystkie pliki .npy istnieją
  if all(f.exists() for f in files_to_check):
    print("Znaleziono zapisane pliki .npy w pamięci. Wczytywanie")

    train_data = {
        "PCA": np.load(CACHE_DIR / f"train_pca_{num_points}p.npy"),
        "D2": np.load(CACHE_DIR / f"train_d2_{num_points}p.npy"),
        "FPFH": np.load(CACHE_DIR / f"train_fpfh_{num_points}p.npy"),
        "labels": np.load(
            CACHE_DIR / f"train_labels_{num_points}p.npy"
        ).tolist(),
    }

    test_data = {
        "PCA": np.load(CACHE_DIR / f"test_pca_{num_points}p.npy"),
        "D2": np.load(CACHE_DIR / f"test_d2_{num_points}p.npy"),
        "FPFH": np.load(CACHE_DIR / f"test_fpfh_{num_points}p.npy"),
        "labels": np.load(
            CACHE_DIR / f"test_labels_{num_points}p.npy"
        ).tolist(),
    }

    categories = sorted(list(set(train_data["labels"])))
    print("Załadowano dane z plików .npy\n")
    return train_data, test_data, categories

  # Kod poniżej wykonuje się TYLKO WTEDY, gdy brak chociaż jednego pliku .npy
  print(f"Brak pamięci dla N={num_points}. Trwa pierwsza ekstrakcja")
  train_data, test_data, categories = load_data_fn(dataset_path, num_points=num_points)

  # Zapisujemy każdy deskryptor do osobnego pliku .npy
  print("Zapisywanie deskryptorów do plików .npy")

  # Zapis zbioru treningowego
  np.save(CACHE_DIR / f"train_pca_{num_points}p.npy", train_data["PCA"])
  np.save(CACHE_DIR / f"train_d2_{num_points}p.npy", train_data["D2"])
  np.save(CACHE_DIR / f"train_fpfh_{num_points}p.npy", train_data["FPFH"])
  np.save(CACHE_DIR / f"train_labels_{num_points}p.npy", train_data["labels"])

  # Zapis zbioru testowego
  np.save(CACHE_DIR / f"test_pca_{num_points}p.npy", test_data["PCA"])
  np.save(CACHE_DIR / f"test_d2_{num_points}p.npy", test_data["D2"])
  np.save(CACHE_DIR / f"test_fpfh_{num_points}p.npy", test_data["FPFH"])
  np.save(CACHE_DIR / f"test_labels_{num_points}p.npy", test_data["labels"])

  print("Zapis .npy zakończony sukcesem!\n")
  return train_data, test_data, categories

if __name__ == "__main__":
  from main import load_dataset_and_extract_features

  print("Testowanie modułu cache")
  train_data, test_data, categories = cache_memory(
      load_dataset_and_extract_features,
      dataset_path="ModelNet10",
      num_points=1000,
  )
  print(f" Sukces! Załadowano kategorie: {categories}")