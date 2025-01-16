import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from PIL import Image
import os

# 1.1 Laden der Daten
def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)
def data_overview(x_train, y_train):
    categories = [
        "T-shirt/Top", "Hose", "Pullover", "Kleid", "Mantel",
        "Sandalen", "Hemd", "Sneaker", "Tasche", "Halbschuhe"
    ]

    # 1.1.1 Dimensionen der Daten
    print(f"Dimensionen der Trainingsdaten: {x_train.shape}")
    print(f"Dimensionen der Testdaten: {y_train.shape}")

    # 1.1.2 Häufigkeiten pro Kategorie
    unique, counts = np.unique(y_train, return_counts=True)
    print("Häufigkeiten pro Kategorie:")
    for category, count in zip(categories, counts):
        print(f"{category}: {count}")

    # 1.1.3 Pixel des 10. Bildes und die Kategorie
    index = 9  # 10. Bild (0-basierte Indexierung)
    print("\nDetails zum 10. Bild:")
    print(f"Pixelwerte:\n{x_train[index]}")
    print(f"Kategorie: {categories[y_train[index]]}")

# 1.2 Visualisierung der Daten
def visualize_images(x_train, y_train, output_dir="visualization"):
    os.makedirs(output_dir, exist_ok=True)


    for i in range(100):
        img = Image.fromarray(x_train[i])
        label = y_train[i]
        img.save(f"{output_dir}/{i}_{label}.jpeg")

    print(f"Die ersten 100 Bilder wurden in {output_dir} gespeichert.")


def export_images_by_category(x_train, y_train, output_dir="by_category"):
    os.makedirs(output_dir, exist_ok=True)

    categories = [
        "T-shirt/Top", "Hose", "Pullover", "Kleid", "Mantel",
        "Sandalen", "Hemd", "Sneaker", "Tasche", "Halbschuhe"
    ]

    for category in range(10):
        category_dir = os.path.join(output_dir, categories[category])
        os.makedirs(category_dir, exist_ok=True)

        category_images = x_train[y_train == category]
        for i, img in enumerate(category_images[:10]):
            img = Image.fromarray(img)
            img.save(f"{category_dir}/{i}.jpeg")

    print(f"Bilder wurden nach Kategorien in {output_dir} gespeichert.")

if __name__ == "__main__":
    (x_train, y_train), _ = load_data()
    visualize_images(x_train, y_train)
    export_images_by_category(x_train, y_train)
