from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from cache import cache_memory
from main import evaluate_and_plot, load_dataset_and_extract_features

def knn_accuracy(X_train, X_test, y_train, y_test, k=5):
    """
    Pomocnicza funkcja trenująca klasyfikator k-NN i zwracająca dokładność (Accuracy) w %.
    Używa metryki euklidesowej do mierzenia odległości w przestrzeni cech.
    """
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)  # Dopasowanie modelu do danych treningowych
    # Predykcja na zbiorze testowym i obliczenie odsetka poprawnych rozpoznań
    return accuracy_score(y_test, knn.predict(X_test)) * 100


def evaluate_fusion(num_points=1000, k_neighbors=5):
    """
    Główna funkcja eksperymentu:
    1. Wczytuje wyekstrahowane cechy chmur punktów.
    2. Ocenia pojedyncze deskryptory (PCA, D2, FPFH).
    3. Tworzy fuzje cech i bada wpływ standaryzacji (StandardScaler).
    4. Generuje macierz pomyłek oraz wykres porównawczy.
    """
    print("Wczytywanie danych z cache...")
    # Pobranie danych z pamięci podręcznej (lub ich wyliczenie, jeśli brak cache)
    train_data, test_data, categories = cache_memory(
        load_dataset_and_extract_features,
        dataset_path="ModelNet10",
        num_points=num_points,
    )

    # Etykiety klas obiektów (np. stół, krzesło, łóżko)
    y_train = np.array(train_data["labels"])
    y_test = np.array(test_data["labels"])
    results = {}

    # KROK 1: Ewaluacja każdego deskryptora z osobna
    for desc in ["PCA", "D2", "FPFH"]:
        results[desc] = knn_accuracy(
            np.array(train_data[desc]),
            np.array(test_data[desc]),
            y_train,
            y_test,
            k_neighbors,
        )
        print(f"Pojedynczy [{desc}]: {results[desc]:.2f}%")

    # KROK 2: Fuzja cech (sklejenie wektorów horyzontalnie: PCA + D2 + FPFH)
    X_train_fused = np.hstack([train_data["PCA"], train_data["D2"], train_data["FPFH"]])
    X_test_fused = np.hstack([test_data["PCA"], test_data["D2"], test_data["FPFH"]])

    # Wariant A: Fuzja bez normalizacji/skalowania cech
    # (Cechy o większych wartościach liczbowych mogą zdominować metrykę euklidesową)
    results["Fuzja (bez skalowania)"] = knn_accuracy(
        X_train_fused, X_test_fused, y_train, y_test, k_neighbors
    )
    print(f"Fuzja bez skalowania: {results['Fuzja (bez skalowania)']:.2f}%")

    # Wariant B: Fuzja ze standaryzacją (z-score: średnia=0, odchylenie std=1)
    # Wyrównuje wagi poszczególnych deskryptorów w k-NN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fused)  # Dopasowanie na train i transformacja
    X_test_scaled = scaler.transform(X_test_fused)  # Transformacja testu parametrami z train

    results["Fuzja (ze skalowaniem)"] = knn_accuracy(
        X_train_scaled, X_test_scaled, y_train, y_test, k_neighbors
    )
    print(f"Fuzja ze skalowaniem: {results['Fuzja (ze skalowaniem)']:.2f}%")

    # KROK 3: Generowanie macierzy pomyłek (Confusion Matrix)
    evaluate_and_plot(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        "FUZJA (PCA+D2+FPFH,skalowane)",
        categories,
        k_neighbors=k_neighbors,
    )

    # KROK 4: Rysowanie i zapis wykresu słupkowego z wynikami
    plot_summary(results, num_points)
    return results


def plot_summary(results, num_points):
    """
    Rysuje wykres słupkowy podsumowujący uzyskane dokładności.
    """
    # Formatowanie etykiet (podział długich nazw na dwie linie)
    labels = [l.replace(" (", "\n(") for l in results.keys()]
    values = list(results.values())
    colors = ["blue"] * 3 + ["red", "green"]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values, color=colors)

    # Dodanie etykiet tekstowych z wartościami procentowymi nad słupkami
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1.5,
            f"{yval:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.ylabel("Dokładność (%)")
    plt.title(
        "Wpływ fuzji deskryptorów i standaryzacji na dokładność k-NN\n"
        f"(N={num_points} pkt)"
    )
    plt.ylim(0, 100)
    plt.grid(axis="y")
    plt.tight_layout()

    # Zapis wykresu do katalogu 'results'
    Path("results").mkdir(exist_ok=True)
    save_path = "results/feature_fusion.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\nWykres zapisany w: {save_path}")


if __name__ == "__main__":
    evaluate_fusion(num_points=1000, k_neighbors=5)