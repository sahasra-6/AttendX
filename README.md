
# AttendX – Face Recognition Attendance System

AttendX is a Python-based attendance management system that uses face recognition to identify students and record their attendance.

## Features

- Face recognition using DeepFace and FaceNet.
- Teacher dashboard to start attendance sessions.
- Automatic attendance recording for recognized students.
- Registration of new students by teachers.
- Student dashboard to check attendance using roll number or name.
- Admin dashboard to view registered students and attendance records.
- Experimental liveness and anti-spoofing checks.

## Technologies Used

- Python
- Streamlit
- OpenCV
- DeepFace
- MediaPipe
- NumPy
- Pandas
- Pickle
- CSV

## Project Structure

```text
face_attendance_system/
├── frontend.py
├── capture_faces.py
├── generate_embeddings.py
├── mark_attendance.py
├── presence_tracking.py
├── calculate_duration.py
├── view_pkl.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project directory:

```bash
cd face_attendance_system
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate it on Windows:

```bash
venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run frontend.py
```

## Privacy Notice

Student face images, face embeddings, and attendance records are excluded from this public repository. Use biometric data only with appropriate consent and authorization.

## Future Improvements

- Stronger presentation attack detection.
- Improved attendance analytics.
- Enhanced security and authentication.
- Cloud database integration.

## Author

Developed as an academic project.
