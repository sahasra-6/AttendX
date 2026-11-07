import cv2
import os

# Ask student details
student_id = input("Enter Student Name_RollNo: ")
save_path = f"dataset/{student_id}"

# Create folder for student
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Open webcam
cap = cv2.VideoCapture(0)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# count = 0
# Start count from existing images (append instead of overwrite)
existing_images = len(os.listdir(save_path))
count = existing_images

MAX_IMAGES = 60

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))

        cv2.imwrite(f"{save_path}/{count}.jpg", face)
        count += 1

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Face Data Collection", frame)

    if count >= MAX_IMAGES:
        break

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
print("Face data collection completed!")
