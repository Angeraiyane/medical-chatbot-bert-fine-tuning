import os


def create_directories(*directories):
    """
    Crée les dossiers s'ils n'existent pas.
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def print_separator():
    print("=" * 60)