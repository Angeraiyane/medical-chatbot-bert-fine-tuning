from datasets import load_dataset
import pandas as pd


def main():

    print("=" * 60)
    print("Chargement du dataset...")
    print("=" * 60)

    dataset = load_dataset("lavita/MedQuAD")

    df = dataset["train"].to_pandas()

    print("\nPremières lignes :")
    print(df.head())

    print("\nInformations générales :")
    print(df.info())

    print("\nDimensions :")
    print(df.shape)

    print("\nColonnes :")
    print(df.columns.tolist())

    print("\nValeurs manquantes :")
    print(df.isnull().sum())

    print("\nStatistiques :")
    print(df.describe(include="all"))

    print("\nTypes de questions :")
    print(df["question_type"].value_counts())


if __name__ == "__main__":
    main()