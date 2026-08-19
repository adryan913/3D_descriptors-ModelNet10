import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from cache import cache_memory
from main import load_dataset_and_extract_features

def knn_tune():
    # wczytanie danych przez cache
    train_data, test_data, categories = cache_memory(load_dataset_and_extract_features,dataset_path="ModelNet10", num_points=1000)

    # przeksztalcenie danych treningowych i testowych na tablice
    y_train = np.array(train_data["labels"])
    y_test = np.array(test_data["labels"])

    # zdefiniowanie listy badanych wartosci k
    k_val = [1,3,5,7,10,15,20,25,30]

    # zdefiniowanie listy deskryptorów
    descriptors = ["PCA", "D2", "FPFH"]

    # przygotowanie okna wykresu
    plt.figure(figsize=(10,6))

    # pętla po descriptors
    for desc in descriptors:
        # dla każdego deskryptora pobierz wartosci train i test
        x_train = np.array(train_data[desc])
        x_test = np.array(test_data[desc])

        accuracies = []

        for k in k_val:
            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            acc = accuracy_score(y_test, y_pred) * 100
            accuracies.append(acc)
            print(f"Deskryptor: {desc:<5} | k={k:<2} -> Dokładność: {acc:.2f}%")
        plt.plot(k_val, accuracies, label=desc)

    plt.title("Wpływ liczby sąsiadów na dokładność klasyfikacji")
    plt.xlabel("Liczba sąsiadów (k)")
    plt.ylabel("Dokładność (%)")
    plt.xticks((k_val))
    plt.grid(True)
    plt.legend()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/knn_k_tuning.png")
    plt.close()
    print("Zapisano wykres")

if __name__ == "__main__":
    knn_tune()