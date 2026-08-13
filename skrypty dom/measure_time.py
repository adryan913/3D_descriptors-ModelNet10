import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from main import load_and_sample_off
from descriptors import deskryptor_pca, deskryptor_d2, deskryptor_fpfh
from pathlib import Path


def measure_descriptor_times(sample_files, num_points=1000):
    pcds = [load_and_sample_off(f, num_points=num_points) for f in sample_files]

    times = {'PCA': 0.0, 'D2': 0.0, 'FPFH': 0.0}

    # Pomiar PCA
    t0 = time.time()
    for pcd in pcds:
        _ = deskryptor_pca(pcd)
    times['PCA'] = (time.time() - t0) / len(pcds) * 1000  # czas na 1 próbkę w ms

    # Pomiar D2
    t0 = time.time()
    for pcd in pcds:
        _ = deskryptor_d2(pcd)
    times['D2'] = (time.time() - t0) / len(pcds) * 1000

    # Pomiar FPFH
    t0 = time.time()
    for pcd in pcds:
        _ = deskryptor_fpfh(pcd)
    times['FPFH'] = (time.time() - t0) / len(pcds) * 1000

    return times


if __name__ == "__main__":
    # Pobranie 100 przykładowych plików do rzetelnego testu wydajności
    files = list(Path("../ModelNet10").rglob("*.off"))[:100]

    print(" Mierzenie średniego czasu ekstrakcji cech dla 1 obiektu...")
    avg_times = measure_descriptor_times(files, num_points=1000)

    print("\n--- ŚREDNI CZAS EKSTRAKCJI CECH (dla 1 obiektu 3D) ---")
    for desc, t_ms in avg_times.items():
        print(f"  {desc:5s}: {t_ms:.2f} ms")

    # Wykres czas vs dokładność
    accuracies = {'PCA': 37.00, 'D2': 55.73, 'FPFH': 60.02}

    plt.figure(figsize=(8, 6))
    for desc in ['PCA', 'D2', 'FPFH']:
        plt.scatter(avg_times[desc], accuracies[desc], s=200, label=desc)
        plt.annotate(f"{desc}\n({avg_times[desc]:.1f} ms, {accuracies[desc]}%)",
                     (avg_times[desc], accuracies[desc]),
                     textcoords="offset points", xytext=(10, -5), ha='left')

    plt.title('Kompromis wydajnościowy: Czas obliczeń vs Dokładność')
    plt.xlabel('Średni czas ekstrakcji cech na 1 obiekt [ms]')
    plt.ylabel('Dokładność / Accuracy [%]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../results/time_vs_accuracy.png', dpi=300)
    plt.show()
    print("\nZapisano wykres do: results/time_vs_accuracy.png")