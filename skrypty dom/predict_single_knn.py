import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from sklearn.neighbors import KNeighborsClassifier

from data_loader import load_and_sample_off
from descriptors import deskryptor_pca, deskryptor_d2, deskryptor_fpfh
from main import load_dataset_and_extract_features


class ShapeClassifier3D:
    def __init__(self, dataset_path="ModelNet10", num_points=1000, k_neighbors=5):
        print(" Inicjalizacja demo k-NN... Wczytywanie bazy i trenowanie modeli...")
        self.num_points = num_points
        self.k_neighbors = k_neighbors

        train_data, test_data, self.categories = load_dataset_and_extract_features(dataset_path, num_points=num_points)

        y_all = np.array(train_data['labels'] + test_data['labels'])

        self.models = {}
        for desc in ['PCA', 'D2', 'FPFH']:
            X_all = np.array(train_data[desc] + test_data[desc])
            knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric='euclidean')
            knn.fit(X_all, y_all)
            self.models[desc] = knn

        print(" Modele k-NN gotowe do predykcji!\n")

    def predict_and_visualize(self, file_path):
        file_path = Path(file_path)
        if not file_path.exists():
            print(f" BŁĄD: Plik '{file_path}' nie istnieje!")
            return

        print(f" Analiza obiektu: {file_path.name}")

        pcd = load_and_sample_off(str(file_path), num_points=self.num_points)

        pca_feat = deskryptor_pca(pcd).reshape(1, -1)
        d2_feat = deskryptor_d2(pcd).reshape(1, -1)
        fpfh_feat = deskryptor_fpfh(pcd).reshape(1, -1)

        pred_pca = self.models['PCA'].predict(pca_feat)[0]
        pred_d2 = self.models['D2'].predict(d2_feat)[0]
        pred_fpfh = self.models['FPFH'].predict(fpfh_feat)[0]

        print("\n" + "=" * 40)
        print(f"  WYNIKI KLASYFIKACJI k-NN DLA: {file_path.name}")
        print("=" * 40)
        print(f"  Prawdziwa kategoria (Folder): {file_path.parent.parent.name}")
        print(f"  --------------------------------------")
        print(f"  Predykcja [PCA]  : {pred_pca}")
        print(f"  Predykcja [D2]   : {pred_d2}")
        print(f"  Predykcja [FPFH] : {pred_fpfh}")
        print("=" * 40 + "\n")

        print(" Otwieranie okna wizualizacji 3D (zamknij okno, aby kontynuować)...")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"Obiekt: {file_path.name} | FPFH Pred: {pred_fpfh}"
        )


if __name__ == "__main__":
    demo = ShapeClassifier3D(dataset_path="../ModelNet10", num_points=1000, k_neighbors=5)
    test_object = "ModelNet10/bed/test/bed_0591.off"
    demo.predict_and_visualize(test_object)