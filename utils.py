import os

def create_directory(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    print(f"Verzeichnis {dir_path} wurde erstellt.")
