# 3D Point Cloud Classification - ModelNet10

Projekt analizy porównawczej klasycznych deskryptorów geometrii 3D (PCA, D2, FPFH) w zadaniu klasyfikacji chmur punktów ze zbioru ModelNet10 przy użyciu algorytmu k-NN.

## Struktura projektu
* `descriptors.py` – implementacja ekstrakcji cech geometrycznych (PCA, D2, FPFH).
* `data_loader.py` – wczytywanie, centrowanie i skalowanie chmur punktów.
* `cache.py` – moduł pamięci podręcznej zapisujący wyekstrahowane cechy do plików `.npy`.
* `knn_tune.py` – analiza wpływu parametru k na dokładność klasyfikacji.
* `sampling_1000vs2048.py` – porównanie gęstości próbkowania (1000 vs 2048 punktów).
* `main.py` – główny potok klasyfikacji i generowanie macierzy pomyłek.