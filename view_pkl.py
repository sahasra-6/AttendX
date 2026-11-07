import pickle

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print("\n===== ALL STUDENT EMBEDDINGS =====\n")

for person in data:
    print(f"\nSTUDENT: {person}")
    print("Total embeddings:", len(data[person]))

    for i, emb in enumerate(data[person]):
        print(f"\nEmbedding {i+1}:")
        print(emb[:15])   # first 15 values (to keep readable)
