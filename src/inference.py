import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/processed/medical_qa.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("models/medical_index.faiss")

print("="*50)
print("Medical Chatbot")
print("="*50)

while True:

    question = input("\nVous : ")

    if question.lower() == "exit":
        break

    vector = model.encode([question])

    distances, indices = index.search(vector, k=1)

    answer = df.iloc[indices[0][0]]["answer"]

    print("\nChatbot :")

    print(answer)