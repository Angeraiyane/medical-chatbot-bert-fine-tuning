import faiss
import numpy as np

embeddings = np.load("models/question_embeddings.npy")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, "models/medical_index.faiss")

print("Index créé avec succès.")

print(index.ntotal)