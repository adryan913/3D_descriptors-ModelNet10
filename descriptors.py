import numpy as np
import open3d as o3d


def deskryptor_pca(pcd):
    """
    Wyznacza macierz kowariancji, wartości własne (lambda_1 >= lambda_2 >= lambda_3)
    oraz wskaźniki geometryczne: Linearness, Planarity, Sphericity.
    """
    # Wyciągnięcie punktów jako tablica numpy (n*3)
    punkty = np.asarray(pcd.points)

    # Obliczenie macierzy kowariancji 3x3 dla punktów X, Y, Z
    cov_matrix = np.cov(punkty, rowvar=False)

    # wyznaczanie wartości własnych, tylko wartości rzeczywiste
    # np.linalg.eigh jest do macierzy symetrycznych
    eigenvalues, _ = np.linalg.eigh(cov_matrix)

    # Sortowanie wartości własnych malejąco: l1 >= l2 >= l3
    eigenvalues = np.sort(eigenvalues)[::-1]
    l1, l2, l3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]

    # Zabezpieczenie przed dzieleniem przez zero
    if l1 <= 0:
        return np.array([0.0, 0.0, 0.0])

    # Wskaźniki geometryczne podane w pliku praktyk
    linearness = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    sphericity = l3 / l1

    return np.array([linearness, planarity, sphericity], dtype=float)


def deskryptor_d2(pcd, num_pairs=10000, num_bins=30):
    """
    Losuje N par punktów, mierzy ich odległości euklidesowe
    i buduje unormowany histogram.
    """
    punkty = np.asarray(pcd.points)
    num_points = len(punkty)

    # Losowanie indeksów par punktów bez pętli for
    idx1 = np.random.randint(0, num_points, size=num_pairs)
    idx2 = np.random.randint(0, num_points, size=num_pairs)

    # Obliczenie odległości euklidesowych między parami
    distances = np.linalg.norm(punkty[idx1] - punkty[idx2], axis=1)

    # Histogram w zakresie [0, 2] (maksymalna odległość w sferze jednostkowej r=1 wynosi 2)
    hist, _ = np.histogram(distances, bins=num_bins, range=(0.0, 2.0))
    hist = hist / hist.sum()
    return hist


def deskryptor_fpfh(pcd, k_neighbors=30):
    """
    Estymuje wektory normalne dla K sąsiadów, a następnie wylicza
    33-wymiarowy histogram FPFH i uśrednia go dla całej chmury.
    """
    # Estymacja wektorów normalnych
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))

    # Obliczenie FPFH za pomocą Open3D
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )

    # Uśrednienie po wszystkich punktach obiektu -> wektor 33-elementowy
    fpfh_vector = np.mean(fpfh.data, axis=1)

    return fpfh_vector


if __name__ == "__main__":
    from data_loader import load_and_sample_off

    test_file = "ModelNet10/chair/train/chair_0001.off"
    pcd = load_and_sample_off(test_file, num_points=1000)

    pca_feat = deskryptor_pca(pcd)
    print(f"1. PCA (3 cechy float): {pca_feat}")
    d2_feat = deskryptor_d2(pcd)
    print(f"2. D2 (histogram):   długość = {len(d2_feat)}, suma = {np.sum(d2_feat):.2f}")
    fpfh_feat = deskryptor_fpfh(pcd)
    print(f"3. FPFH (histogram): długość = {len(fpfh_feat)}")