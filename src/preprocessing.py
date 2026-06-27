from datasets import load_dataset
import pandas as pd


def load_data():
    """
    Charge le dataset MedQuAD.
    """
    dataset = load_dataset("lavita/MedQuAD")
    return dataset["train"].to_pandas()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le dataset.
    """

    print("=" * 60)
    print("Nettoyage du dataset")
    print("=" * 60)

    print(f"Nombre de lignes avant nettoyage : {len(df)}")

    # Conserver uniquement les colonnes utiles
    df = df[["question", "answer"]]

    # Supprimer les valeurs manquantes
    df = df.dropna()

    # Supprimer les doublons
    df = df.drop_duplicates()

    # Réinitialiser les index
    df = df.reset_index(drop=True)

    print(f"Nombre de lignes après nettoyage : {len(df)}")

    return df


def main():

    df = load_data()

    df = clean_data(df)

    print("\nAperçu :")
    print(df.head())

    print("\nInformations :")
    print(df.info())

    # Sauvegarde
    df.to_csv("data/processed/medical_qa.csv", index=False)

    print("\nDataset sauvegardé dans :")
    print("data/processed/medical_qa.csv")


if __name__ == "__main__":
    main()