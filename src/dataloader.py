from torch.utils.data import DataLoader

from dataset import MedicalQADataset

BATCH_SIZE = 8


def main():

    dataset = MedicalQADataset(
        "data/processed/medical_qa.csv"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("=" * 60)
    print("Informations")
    print("=" * 60)

    print("Nombre d'exemples :", len(dataset))

    print("Batch size :", BATCH_SIZE)

    print("Nombre de batches :", len(dataloader))

    print("=" * 60)

    batch = next(iter(dataloader))

    print("\nDimensions")

    print(batch["input_ids"].shape)

    print(batch["attention_mask"].shape)

    print("\nPremière question :")

    print(batch["question"][0])

    print("\nPremière réponse :")

    print(batch["answer"][0][:300], "...")


if __name__ == "__main__":
    main()