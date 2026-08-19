import os
from pathlib import Path
import open3d as o3d
import numpy as np
np.random.seed(42)
o3d.utility.random.seed(42)

def preprocess_point_cloud(pcd):
    """
    Normalizacja pozycji i skali chmury punktów:
    1. Przesunięcie środka ciężkości do [0,0,0]
    2. Przeskalowanie do wnętrza sfery jednostkowej
    """
    # Konwersja wektora punktów Open3D na tablicę NumPy (jako widok danych, bez zbędnej kopii w pamięci)
    points = np.asarray(pcd.points)

    # 1. Wycentrowanie w [0,0,0]
    center = np.mean(points, axis=0)
    points_centered = points - center

    # 2. Skalowanie (promień sfery jednostkowej = 1)
    max_distance = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
    points_normalized = points_centered / max_distance

    # Przypisanie zaktualizowanych punktów z powrotem do obiektu Open3D
    pcd.points = o3d.utility.Vector3dVector(points_normalized)
    return pcd


def load_and_sample_off(file_path, num_points=1000):
    """
    Wczytuje siatkę trójkątów (.off), wykonuje próbkowanie powierzchni
    do zadanej liczby punktów oraz stosuje preprocessing.
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pcd = preprocess_point_cloud(pcd)
    return pcd


def save_point_cloud_capture(pcd, output_filename="capture.png", output_dir="captures"):
    """
    Generuje obraz zrzutu ekranu 3D dla podanej chmury punktów
    i zapisuje go w wybranym folderze.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    full_save_path = str(output_path / output_filename)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_float_buffer(False)
    vis.capture_screen_image(full_save_path)
    vis.destroy_window()

    print(f" Zapisano zrzut ekranu w: {full_save_path}")


# --- TEST DZIAŁANIA ---
if __name__ == "__main__":
    test_file = "ModelNet10/monitor/train/monitor_0005.off"

    if os.path.exists(test_file):
        print(f"Wczytywanie i przetwarzanie pliku: {test_file}...")
        pcd = load_and_sample_off(test_file, num_points=1000)

        # Wyświetlanie szczegółowych statystyk w konsoli
        pts = np.asarray(pcd.points)
        print(f"Sukces! Liczba punktów w chmurze: {pts.shape[0]}")
        print(f"Środek ciężkości (powinien być bliski [0,0,0]): {np.mean(pts, axis=0)}")
        print(f"Maksymalna odległość od środka (max r <= 1.0): {np.max(np.sqrt(np.sum(pts ** 2, axis=1))):.4f}")

        # Zapisanie zrzutu ekranu w folderze captures/
        save_point_cloud_capture(pcd, output_filename="capture_monitor_0005.png", output_dir="captures")

        # Otwarcie okienka wizualizacji 3D
        print("\nOtwieranie okna z wizualizacją 3D... (Zamknij okno, aby zakończyć test)")
        o3d.visualization.draw_geometries([pcd], window_name="Sprobkowana i wycentrowana chmura 3D")
    else:
        print(f"Nie znaleziono pliku testowego: {test_file}")