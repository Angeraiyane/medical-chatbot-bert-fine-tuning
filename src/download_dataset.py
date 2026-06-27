from datasets import load_dataset


def main():
    print("Téléchargement du dataset MedQuAD...")

    dataset = load_dataset("lavita/MedQuAD")

    print("\nDataset téléchargé avec succès !")
    print(dataset)

    print("\nPremier exemple :")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()