import os
import pickle
from deepface import DeepFace

DATASET = "dataset"
OUTPUT = "embeddings.pkl"

database = {}

for student in os.listdir(DATASET):
    person_path = os.path.join(DATASET, student)
    embeds = []

    for img in os.listdir(person_path):
        path = os.path.join(person_path, img)

        result = DeepFace.represent(
            img_path=path,
            model_name="Facenet",
            enforce_detection=False
        )

        embeds.append(result[0]["embedding"])

    database[student] = embeds

with open(OUTPUT, "wb") as f:
    pickle.dump(database, f)

print("✅ Embeddings saved to embeddings.pkl")
