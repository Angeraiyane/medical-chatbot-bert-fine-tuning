import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256


class MedicalQADataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        question = self.data.iloc[index]["question"]
        answer = self.data.iloc[index]["answer"]

        encoding = self.tokenizer(
            question,
            answer,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {

            "input_ids": encoding["input_ids"].squeeze(),

            "attention_mask": encoding["attention_mask"].squeeze(),

            "question": question,

            "answer": answer

        }


def main():

    dataset = MedicalQADataset("data/processed/medical_qa.csv")

    print("=" * 60)

    print("Nombre d'exemples :", len(dataset))

    print("=" * 60)

    sample = dataset[0]

    print("\nQuestion :")

    print(sample["question"])

    print("\nRéponse :")

    print(sample["answer"][:200], "...")

    print("\nInput IDs :")

    print(sample["input_ids"])

    print("\nAttention Mask :")

    print(sample["attention_mask"])


if __name__ == "__main__":

    main()