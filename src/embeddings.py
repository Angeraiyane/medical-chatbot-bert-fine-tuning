import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

print("=" * 60)
print("Chargement du dataset...")
print("=" * 60)

df = pd.read_csv("data/processed/medical_qa.csv")

print(df.head())

print("\nChargement du modèle...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Création des embeddings...")

embeddings = model.encode(
    df["question"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

print(embeddings.shape)

np.save("models/question_embeddings.npy", embeddings)

print("\nEmbeddings sauvegardés.")