import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_loader import load_and_sample_off
from descriptors import deskryptor_pca, deskryptor_d2, deskryptor_fpfh
from cache import cache_memory


def load_dataset_and_extract_features(dataset_path="ModelNet10", num_points=1000):
    """
    Przechodzi przez zbiór ModelNet10 (osobno train i test),
    wyciąga deskryptory PCA, D2, FPFH i zwraca je z etykietami.
    """
    data_dir = Path(dataset_path)

    # Listy na cechy treningowe i testowe dla każdego deskryptora
    train_data = {'PCA': [], 'D2': [], 'FPFH': [], 'labels': []}
    test_data = {'PCA': [], 'D2': [], 'FPFH': [], 'labels': []}

    # Wyciągamy nazwy kategorii (np. chair, bed, desk...)
    categories = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Znaleziono kategoria obiektu ({len(categories)}): {categories}\n")

    for split in ['train', 'test']:
        current_target = train_data if split == 'train' else test_data
        print(f"--- Processing split: {split.upper()} ---")

        for category in categories:
            cat_dir = data_dir / category / split
            if not cat_dir.exists():
                continue

            off_files = list(cat_dir.glob("*.off"))
            print(f"Przetwarzanie kategorii '{category}' ({len(off_files)} plików)...")

            for file_path in off_files:
                try:
                    # 1. Wczytanie i preprocessing chmury punktów
                    pcd = load_and_sample_off(str(file_path), num_points=num_points)

                    # 2. Wyliczenie deskryptorów
                    pca_feat = deskryptor_pca(pcd)
                    d2_feat = deskryptor_d2(pcd)
                    fpfh_feat = deskryptor_fpfh(pcd)

                    # 3. Dodanie do odpowiednich zbiorów
                    current_target['PCA'].append(pca_feat)
                    current_target['D2'].append(d2_feat)
                    current_target['FPFH'].append(fpfh_feat)
                    current_target['labels'].append(category)

                except Exception as e:
                    print(f"Błąd dla pliku {file_path}: {e}")

    return train_data, test_data, categories


def evaluate_and_plot(X_train, X_test, y_train, y_test, descriptor_name, categories, k_neighbors=5):
    """
    Trenuje k-NN, oblicza Accuracy, drukuje raport i zapisuje Macierz Pomyłek do pliku PNG.
    """
    print(f"\n==========================================")
    print(f" EVALUATION: {descriptor_name} (k-NN, k={k_neighbors})")
    print(f"==========================================")

    # Inicjalizacja i trening klasyfikatora k-NN z metryką euklidesową
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean')
    knn.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = knn.predict(X_test)

    # Wyliczenie dokładności (Accuracy)
    acc = accuracy_score(y_test, y_pred)
    print(f" Dokładność (Accuracy) [{descriptor_name}]: {acc * 100:.2f}%\n")
    print("Raport klasyfikacji:")
    print(classification_report(y_test, y_pred))

    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred, labels=categories)

    # Rysowanie i zapis macierzy pomyłek
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.title(f'Macierz pomyłek - {descriptor_name} (Accuracy: {acc * 100:.2f}%)')
    plt.xlabel('Przewidywana kategoria (Predicted)')
    plt.ylabel('Prawdziwa kategoria (True)')
    plt.tight_layout()

    # Stworzenie folderu na wyniki
    os.makedirs("results", exist_ok=True)
    save_path = f"results/confusion_matrix_{descriptor_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" Zapisano macierz pomyłek w: {save_path}")


if __name__ == "__main__":
    DATASET_PATH = "ModelNet10"  # Ścieżka do pobranego zbioru

    print("Rozpoczynanie pełnego potoku klasyfikacji 3D...")

    # WCZYTANIE Z PAMIĘCI CACHE:
    train_data, test_data, categories = cache_memory(
        load_dataset_and_extract_features,
        dataset_path=DATASET_PATH,
        num_points=1000
    )

    # Konwersja na tablice NumPy
    y_train = np.array(train_data['labels'])
    y_test = np.array(test_data['labels'])

    # Ewaluacja każdego deskryptora osobno
    descriptors = ['PCA', 'D2', 'FPFH']

    for desc_name in descriptors:
        X_train = np.array(train_data[desc_name])
        X_test = np.array(test_data[desc_name])

        evaluate_and_plot(X_train, X_test, y_train, y_test, desc_name, categories, k_neighbors=5)

    print("\n ZAKOŃCZONO PEŁNY PROCES! Wszystkie wyniki i macierze pomyłek znajdują się w folderze 'results/'.")