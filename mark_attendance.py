import cv2
import pickle
import numpy as np
from deepface import DeepFace

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    db = pickle.load(f)

# Cosine similarity
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Average embeddings per student (IMPORTANT FIX)
avg_db = {}
for person in db:
    avg_db[person] = np.mean(db[person], axis=0)

# Webcam
cap = cv2.VideoCapture(0)

stable_name = ""
stable_count = 0

THRESHOLD = 0.65     # similarity threshold
STABLE_FRAMES = 5   # how many frames required for confirmation

print("Attendance system started... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Extract live face embedding
        live = DeepFace.represent(
            frame,
            model_name="Facenet",
            enforce_detection=False
        )

        live_vec = live[0]["embedding"]

        best_name = "Unknown"
        best_score = 0

        # Compare with each student
        for person in avg_db:
            vec = avg_db[person]
            score = cosine(live_vec, vec)

            if score > best_score:
                best_score = score
                best_name = person

        # Debug similarity
        print("Match:", best_name, "Score:", round(best_score, 3))

        # Thresholding
        if best_score < THRESHOLD:
            best_name = "Unknown"

        # Stability check
        if best_name == stable_name:
            stable_count += 1
        else:
            stable_name = best_name
            stable_count = 0

        # Display result
        if stable_count > STABLE_FRAMES:
            color = (0, 255, 0) if stable_name != "Unknown" else (0, 0, 255)
            cv2.putText(frame, stable_name, (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    except Exception as e:
        print("Detection error")

    cv2.imshow("Attendance Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
