from transformers import AutoTokenizer
import pandas as pd

MODEL_NAME = "bert-base-uncased"


def main():

    print("=" * 60)
    print("Chargement du tokenizer BERT")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_csv("data/processed/medical_qa.csv")

    question = df.iloc[0]["question"]

    print("\nQuestion :")
    print(question)

    encoding = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )

    print("\nInput IDs :")
    print(encoding["input_ids"])

    print("\nAttention Mask :")
    print(encoding["attention_mask"])

    print("\nTokens :")
    print(tokenizer.convert_ids_to_tokens(
        encoding["input_ids"][0]
    ))


if __name__ == "__main__":
    main()