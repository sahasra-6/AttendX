import cv2
import pickle
import numpy as np
from deepface import DeepFace
from datetime import datetime
import csv
import time

with open("embeddings.pkl","rb") as f:
    db = pickle.load(f)

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

avg_db={}
for p in db:
    avg_db[p]=np.mean(db[p],axis=0)

cap=cv2.VideoCapture(0)

THRESHOLD=0.35
CHECK_INTERVAL=10   # seconds

last_check=time.time()

print("Continuous presence tracking started")

with open("presence_log.csv","w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["Time","Name","Status"])

while True:
    ret,frame=cap.read()
    if not ret:
        break

    name="Unknown"

    try:
        live=DeepFace.represent(frame,model_name="Facenet",enforce_detection=False)
        live_vec=live[0]["embedding"]

        best_score=0
        best_name="Unknown"

        for p in avg_db:
            s=cosine(live_vec,avg_db[p])
            if s>best_score:
                best_score=s
                best_name=p

        if best_score>THRESHOLD:
            name=best_name

    except:
        pass

    now=time.time()

    # Every X seconds log presence
    if now-last_check>CHECK_INTERVAL:
        status="Present" if name!="Unknown" else "Absent"
        timestamp=datetime.now().strftime("%H:%M:%S")

        with open("presence_log.csv","a",newline="") as f:
            writer=csv.writer(f)
            writer.writerow([timestamp,name,status])

        print(timestamp,status)

        last_check=now

    cv2.putText(frame,name,(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Presence Tracking",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()