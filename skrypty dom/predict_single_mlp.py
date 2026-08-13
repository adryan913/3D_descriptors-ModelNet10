import numpy as np
import joblib
from pathlib import Path

from data_loader import load_and_sample_off
from descriptors import deskryptor_pca, deskryptor_d2, deskryptor_fpfh

CATEGORIES = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']


def predict_single_file_mlp(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f" Plik '{file_path}' nie istnieje!")
        return

    print(f"\n🔍 Analiza obiektu przez sztuczną sieć neuronową: {path.name}")

    model_path = Path('../results/mlp_model.joblib')
    scaler_path = Path('../results/mlp_scaler.joblib')

    if not model_path.exists() or not scaler_path.exists():
        print(" Uruchom najpierw skrypt 'train_neural_net.py', aby wytrenować i zapisać model!")
        return

    mlp = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Próbkowanie i wyliczanie deskryptorów
    pcd = load_and_sample_off(str(path), num_points=1000)

    w_pca, w_d2, w_fpfh = 1.0, 1.0, 2.0
    pca_f = deskryptor_pca(pcd) * w_pca
    d2_f = deskryptor_d2(pcd) * w_d2
    fpfh_f = deskryptor_fpfh(pcd) * w_fpfh

    sample_fused = np.hstack([pca_f, d2_f, fpfh_f]).reshape(1, -1)
    sample_scaled = scaler.transform(sample_fused)

    predicted_class = mlp.predict(sample_scaled)[0]
    probabilities = mlp.predict_proba(sample_scaled)[0]

    print("=" * 60)
    print(f" OSTATECZNA PREDYKCJA SIECI MLP: {predicted_class.upper()}")
    print("=" * 60)
    print(" Rozkład prawdopodobieństwa dla klas:")

    prob_pairs = sorted(zip(CATEGORIES, probabilities), key=lambda x: x[1], reverse=True)
    for cat, prob in prob_pairs:
        bar = "█" * int(prob * 25)
        print(f"  {cat:12s} : {prob * 100:6.2f}%  {bar}")
    print("=" * 60)


if __name__ == "__main__":
    # Testowy plik toalety
    test_file = "../ModelNet10/night_stand/test/night_stand_0278.off"
    predict_single_file_mlp(test_file)