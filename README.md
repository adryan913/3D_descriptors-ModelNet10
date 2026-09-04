# Klasyfikacja Obiektów 3D: Inżynieria Cech vs Głębokie Uczenie (PointNet)

Repozytorium zawiera kompleksowy projekt badawczy skupiający się na przetwarzaniu i klasyfikacji chmur punktów 3D z wykorzystaniem zbioru **ModelNet10**. Projekt zestawia klasyczne deskryptory geometryczne z nowoczesnymi architekturami głębokiego uczenia.

Projekt został zrealizowany w ramach akademickich praktyk badawczych w Instytucie Telekomunikacji Multimedialnej Politechniki Poznańskiej.
## Przegląd Projektu

Głównym celem projektu jest ewaluacja i porównanie różnych podejść do rozpoznawania obiektów 3D:
1. **Klasyczna inżynieria cech:** Implementacja deskryptorów PCA, Shape Distribution (D2) oraz Fast Point Feature Histograms (FPFH).
2. **Fuzja cech:** Połączenie cech lokalnych i globalnych (66-wymiarowy wektor) z wykorzystaniem standaryzacji oraz nieliniowej klasyfikacji (MLP).
3. **Głębokie uczenie (Deep Learning):** Implementacja architektury **PointNet** (z modułem T-Net) operującej bezpośrednio na surowych współrzędnych przestrzennych.

## Struktura Repozytorium

* `data_loader.py` - Preprocessing siatek, równomierne próbkowanie (N=1000) i normalizacja do sfery jednostkowej.
* `descriptors.py` - Matematyczna implementacja PCA i D2 oraz integracja deskryptora FPFH z biblioteki Open3D.
* `cache.py` - System pamięci podręcznej dla plików binarnych `.npy`, redukujący czas ekstrakcji z kilku minut do ułamka sekundy.
* `main.py` - Główny potok ewaluacji pojedynczych deskryptorów z użyciem klasyfikatora k-NN.
* `knn_tune.py` - Optymalizacja parametru k dla klasyfikatora k-NN.
* `sampling_1000vs2048.py` - Analiza wpływu gęstości próbkowania chmury punktów na skuteczność.
* `fusion.py` - Horyzontalna fuzja cech i analiza wpływu `StandardScaler`.
* `train_mlp.py` - Trening wielowarstwowej sieci neuronowej (MLP) na zunifikowanym wektorze 66 cech.
* `pointnet.py` - Implementacja sieci PointNet (T-Net, współdzielone MLP) w PyTorch.
* `predicter.py` - Moduł inferencji do predykcji w czasie rzeczywistym i interaktywnej wizualizacji Open3D dla pojedynczych próbek `.off`.

## Instalacja i Uruchomienie

**1. Klonowanie repozytorium i instalacja zależności:**
```bash
git clone https://github.com/adryan913/3D_descriptors-ModelNet10.git
cd 3D_descriptors-ModelNet10
pip install -r requirements.txt
```

**2. Przygotowanie zbioru danych:**
Pobierz zbiór [ModelNet10](https://modelnet.cs.princeton.edu/) i umieść rozpakowany folder `ModelNet10` w głównym katalogu projektu.

**3. Uruchamianie skryptów:**
* Ewaluacja bazowa dla klasycznych deskryptorów: `python main.py`
* Trening sieci MLP na fuzji cech: `python train_mlp.py`
* Trening modelu PointNet: `python pointnet.py`
* Interaktywna predykcja: `python predicter.py`

## Wyniki (Benchmark)

Eksperymenty przeprowadzono na zbiorze testowym ModelNet10. Wyniki obrazują przeskok jakościowy od prostych deskryptorów globalnych do zaawansowanych sieci głębokich.

| Metoda | Wymiar wektora cech | Klasyfikator | Dokładność testowa |
|---|---|---|---|
| **PCA** | 3 | k-NN (k=5) | **37.33%** |
| **D2 (Shape Distribution)** | 30 | k-NN (k=5) | **56.17%** |
| **FPFH** | 33 | k-NN (k=5) | **61.12%** |
| **Fuzja (bez skalowania)** | 66 | k-NN (k=5) | **61.12%** |
| **Fuzja (StandardScaler)** | 66 | k-NN (k=5) | **70.37%** |
| **Fuzja + MLP** | 66 | MLP (256 -> 128 -> 64) | **77.42%** |
| **PointNet** | 1000 * 3 | PointNet (T-Net + 1024-d) | **90.09%** |

### Kluczowe Wnioski
* **Kluczowa rola standaryzacji:** Bezpośrednie połączenie cech (PCA + D2 + FPFH) bez skalowania nie poprawia wyników względem FPFH, ponieważ wartości o większym rzędzie wielkości całkowicie dominują metrykę odległości w k-NN. Zastosowanie `StandardScaler` podnosi dokładność o ponad 9 punktów procentowych.
* **Modelowanie nieliniowe (MLP):** Zastosowanie wielowarstwowej sieci neuronowej na zunifikowanych cechach pozwala modelowi nauczyć się adaptacyjnych wag i nieliniowych korelacji, przewyższając algorytmy oparte na metryce odległości.
* **Przewaga Deep Learning:** Sieć PointNet deklasuje klasyczne metody inżynierii cech. Operując bezpośrednio na surowych współrzędnych XYZ i samodzielnie ekstrahując cechy lokalne i globalne, osiąga skuteczność na poziomie **90.09%**.