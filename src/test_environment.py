import torch


def check_environment():
    """
    Vérifie l'installation de PyTorch et la disponibilité d'un GPU.
    """

    print("=" * 50)
    print("Medical Chatbot - Environment Check")
    print("=" * 50)

    print(f"PyTorch Version : {torch.__version__}")

    if torch.cuda.is_available():
        print("GPU Disponible : Oui")
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU Disponible : Non")
        print("Le projet utilisera le CPU.")

    print("=" * 50)


if __name__ == "__main__":
    check_environment()