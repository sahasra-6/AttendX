import os
import pickle
import time
from datetime import datetime, date

# Load MediaPipe first
import mediapipe as mp

# Other libraries
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from deepface import DeepFace
import random


# ============================================================
# AttendX - Dynamic Face Recognition Attendance
# Existing students are recognized automatically.
# Unknown students can be registered by the teacher during class.
# ============================================================

DB_FILE = "students.pkl"
ATTENDANCE_FILE = "attendance_records.csv"
LEGACY_EMBEDDINGS_FILE = "embeddings.pkl"

ATTENDANCE_COLUMNS = ["Date", "Time", "RollNo", "Name", "Session", "Status"]
MODEL_NAME = "Facenet"
COSINE_SIMILARITY_THRESHOLD = 0.65
UNKNOWN_CONFIRMATIONS = 3
PROCESS_EVERY_N_FRAMES = 8


def normalize(value):
    return str(value).strip().casefold()


def load_students():
    """Load the new database, or migrate the old embeddings.pkl format."""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            st.error(f"Could not read {DB_FILE}: {exc}")
            return {}

    migrated = {}
    if os.path.exists(LEGACY_EMBEDDINGS_FILE):
        try:
            with open(LEGACY_EMBEDDINGS_FILE, "rb") as f:
                old_db = pickle.load(f)

            # Old format: {"StudentName_RollNo": [embedding, ...]}
            for old_key, vectors in old_db.items():
                old_key = str(old_key)
                if "_" in old_key:
                    name, roll_no = old_key.rsplit("_", 1)
                else:
                    name, roll_no = old_key, old_key

                migrated[normalize(roll_no)] = {
                    "name": name.replace("_", " ").strip(),
                    "roll_no": roll_no.strip(),
                    "embeddings": [np.asarray(v, dtype=np.float32) for v in vectors],
                }
            save_students(migrated)
            st.info("Existing embeddings.pkl was imported into students.pkl.")
        except Exception as exc:
            st.warning(f"Could not migrate old embeddings: {exc}")

    return migrated


def save_students(database):
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)


def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame(columns=ATTENDANCE_COLUMNS).to_csv(ATTENDANCE_FILE, index=False)


def load_attendance():
    ensure_attendance_file()
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        for col in ATTENDANCE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[ATTENDANCE_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=ATTENDANCE_COLUMNS)


def mark_present(roll_no, name, session_name):
    """Append one Present record per student per date and class session."""
    ensure_attendance_file()
    today = date.today().isoformat()
    df = load_attendance()

    duplicate = df[
        (df["Date"].astype(str) == today)
        & (df["RollNo"].astype(str).map(normalize) == normalize(roll_no))
        & (df["Session"].astype(str).map(normalize) == normalize(session_name))
        & (df["Status"].astype(str) == "Present")
    ]
    if not duplicate.empty:
        return False

    now = datetime.now()
    new_row = {
        "Date": today,
        "Time": now.strftime("%H:%M:%S"),
        "RollNo": str(roll_no),
        "Name": str(name),
        "Session": str(session_name),
        "Status": "Present",
    }
    pd.DataFrame([new_row]).to_csv(
        ATTENDANCE_FILE, mode="a", header=not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0,
        index=False
    )
    return True


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def identify_student(live_embedding, database):
    best_roll = None
    best_score = -1.0

    for roll_key, record in database.items():
        vectors = record.get("embeddings", [])
        if not vectors:
            continue
        scores = [cosine_similarity(live_embedding, vector) for vector in vectors]
        score = max(scores) if scores else -1.0
        if score > best_score:
            best_score = score
            best_roll = roll_key

    if best_roll is not None and best_score >= COSINE_SIMILARITY_THRESHOLD:
        return best_roll, best_score
    return None, best_score


def eye_aspect_ratio(landmarks, eye_indices, width, height):
    """Calculate a simple eye-openness ratio from MediaPipe FaceMesh points."""
    points = [(landmarks[i].x * width, landmarks[i].y * height) for i in eye_indices]
    vertical_1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    vertical_2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)


def face_yaw_proxy(landmarks):
    """
    Estimate left/right head turn from nose position relative to cheek landmarks.
    This is a lightweight challenge signal, not a certified liveness model.
    """
    left_cheek = landmarks[234].x
    right_cheek = landmarks[454].x
    nose = landmarks[1].x
    span = right_cheek - left_cheek
    if abs(span) < 1e-5:
        return 0.5
    return (nose - left_cheek) / span


def run_camera_attendance(database, session_name):
    """
    Teacher-led attendance with randomized active liveness challenge.

    A face must pass a blink + randomized head-turn challenge before the
    embedding is compared with registered students or offered for registration.
    Static photos cannot perform the challenge. This is a prototype PAD check,
    not a guarantee against sophisticated replay/deepfake attacks.
    """
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not camera.isOpened():
        camera.release()
        st.error(
            "Camera could not be opened. Close other apps using the webcam, "
            "check Windows Camera permissions, and try again."
        )
        return None

    preview = st.empty()
    message = st.empty()
    frame_count = 0
    recorded_this_run = set()
    pending_embedding = None
    stop_reason = "Camera session ended."

    # Random order makes a prerecorded sequence harder to reuse.
    challenge_steps = ["blink", "turn"]
    random.shuffle(challenge_steps)
    current_step = 0
    blink_seen_closed = False
    blink_completed = False
    neutral_yaw_samples = []
    neutral_yaw = None
    turn_completed = False
    stable_face_frames = 0
    waiting_for_face_change = False
    no_face_frames = 0
    last_embedding = None

    mp_face_mesh = mp.solutions.face_mesh
    # MediaPipe FaceMesh landmark indices for the two eyes.
    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]

    message.info(
        "Liveness check: follow the random instructions on screen. "
        "Attendance is not saved until the challenge passes and identity is checked."
    )

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as mesh:
            while True:
                ok, frame = camera.read()
                if not ok:
                    stop_reason = "Could not read a frame from the camera."
                    break

                frame_count += 1
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = mesh.process(rgb)

                label = "Position your face in the camera"
                color = (0, 200, 255)

                if result.multi_face_landmarks:
                    no_face_frames = 0
                    if waiting_for_face_change:
                        label = "Attendance saved. Next student, please."
                        color = (0, 180, 0)
                    elif len(result.multi_face_landmarks) != 1:
                        label = "REJECTED: only one person must be in frame"
                        color = (0, 0, 255)
                        stable_face_frames = 0
                        neutral_yaw_samples = []
                    else:
                        landmarks = result.multi_face_landmarks[0].landmark
                        stable_face_frames += 1

                        left_ear = eye_aspect_ratio(landmarks, left_eye, width, height)
                        right_ear = eye_aspect_ratio(landmarks, right_eye, width, height)
                        ear = (left_ear + right_ear) / 2.0
                        yaw = face_yaw_proxy(landmarks)

                        if neutral_yaw is None:
                            neutral_yaw_samples.append(yaw)
                            label = "Keep your head straight - calibrating"
                            color = (0, 200, 255)
                            if len(neutral_yaw_samples) >= 12:
                                neutral_yaw = float(np.median(neutral_yaw_samples))
                                instruction = challenge_steps[current_step]
                                label = "Liveness: " + ("BLINK" if instruction == "blink" else "TURN HEAD TO ONE SIDE")
                        else:
                            instruction = challenge_steps[current_step] if current_step < len(challenge_steps) else None

                            if instruction == "blink":
                                # Closed-eye frame followed by open-eye frame counts as a blink.
                                if ear < 0.19:
                                    blink_seen_closed = True
                                    label = "Liveness check: blink now"
                                elif blink_seen_closed and ear > 0.22:
                                    blink_completed = True
                                    current_step += 1
                                    label = "Blink detected - good!"
                                else:
                                    label = "Liveness check: BLINK"
                                color = (255, 180, 0)

                            elif instruction == "turn":
                                # Accept a clear turn in either direction relative to calibrated neutral.
                                if abs(yaw - neutral_yaw) > 0.075:
                                    turn_completed = True
                                    current_step += 1
                                    label = "Head movement detected - good!"
                                else:
                                    label = "Liveness check: TURN YOUR HEAD TO ONE SIDE"
                                color = (255, 180, 0)

                            if current_step >= len(challenge_steps):
                                label = "LIVENESS PASSED - checking identity..."
                                color = (0, 180, 0)

                                # Run DeepFace's FasNet anti-spoofing check BEFORE
                                # identity matching or allowing teacher registration.
                                # Fail closed: if PAD is unavailable or errors, do not
                                # create a pending embedding and do not show registration.
                                if frame_count % 3 == 0:
                                    try:
                                        pad_faces = DeepFace.extract_faces(
                                            img_path=frame,
                                            detector_backend="opencv",
                                            enforce_detection=True,
                                            align=True,
                                            anti_spoofing=True,
                                        )

                                        if len(pad_faces) != 1:
                                            label = "REJECTED: exactly one face is required"
                                            color = (0, 0, 255)
                                            continue

                                        if pad_faces[0].get("is_real") is not True:
                                            label = "REJECTED: possible photo/video spoof"
                                            color = (0, 0, 255)
                                            stop_reason = (
                                                "Anti-spoofing rejected this face. "
                                                "No attendance was saved and registration was blocked."
                                            )
                                            break

                                        # Only a face accepted by PAD reaches FaceNet.
                                        faces = DeepFace.represent(
                                            img_path=frame,
                                            model_name=MODEL_NAME,
                                            detector_backend="opencv",
                                            enforce_detection=True,
                                            align=True,
                                        )
                                        if len(faces) == 1:
                                            last_embedding = np.asarray(faces[0]["embedding"], dtype=np.float32)
                                            roll_key, score = identify_student(last_embedding, database)

                                            if roll_key is not None:
                                                record = database[roll_key]
                                                label = f"VERIFIED: {record['name']} ({record['roll_no']})"
                                                color = (0, 180, 0)
                                                if roll_key not in recorded_this_run:
                                                    added = mark_present(record["roll_no"], record["name"], session_name)
                                                    recorded_this_run.add(roll_key)
                                                    if added:
                                                        message.success(
                                                            f"Attendance saved for {record['name']} ({record['roll_no']})."
                                                        )
                                                    else:
                                                        message.info(
                                                            f"{record['name']} is already present for this session."
                                                        )
                                                # Keep the teacher's session running for the next student.
                                                # Wait until this face leaves the frame before starting a fresh challenge.
                                                waiting_for_face_change = True
                                                current_step = len(challenge_steps)
                                            else:
                                                pending_embedding = last_embedding
                                                stop_reason = (
                                                    "Liveness passed, but this face is not registered. "
                                                    "Teacher registration is required before attendance can be saved."
                                                )
                                                break
                                    except Exception as exc:
                                        # Fail closed if the PAD model is unavailable,
                                        # incompatible, or cannot make a reliable decision.
                                        label = "ANTI-SPOOF CHECK FAILED - attendance NOT saved"
                                        color = (0, 0, 255)
                                        stop_reason = (
                                            "Anti-spoofing could not complete. "
                                            "No attendance was saved and registration was blocked. "
                                            f"Details: {exc}"
                                        )
                                        break

                else:
                    # Require a fresh randomized challenge for the next person.
                    no_face_frames += 1
                    label = "No face detected - attendance NOT saved"
                    color = (0, 0, 255)
                    stable_face_frames = 0

                    if waiting_for_face_change and no_face_frames >= 12:
                        waiting_for_face_change = False
                        challenge_steps = ["blink", "turn"]
                        random.shuffle(challenge_steps)
                        current_step = 0
                        blink_seen_closed = False
                        blink_completed = False
                        neutral_yaw_samples = []
                        neutral_yaw = None
                        turn_completed = False

                cv2.putText(
                    frame, label[:85], (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2
                )
                preview.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )

                

                if frame_count > 2400:
                    stop_reason = "Camera timed out. No unverified attendance was saved."
                    break

    finally:
        camera.release()
        
        preview.empty()

    message.info(stop_reason)
    return pending_embedding


def show_admin(database, attendance):
    st.subheader("System overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registered students", len(database))
    c2.metric("Present entries", len(attendance[attendance["Status"] == "Present"]))
    c3.metric("Attendance sessions", attendance["Session"].nunique() if not attendance.empty else 0)
    st.divider()
    st.subheader("Registered students")
    if database:
        students_df = pd.DataFrame([
            {"Name": v.get("name", ""), "Roll Number": v.get("roll_no", ""), "Face Samples": len(v.get("embeddings", []))}
            for v in database.values()
        ])
        st.dataframe(students_df, use_container_width=True, hide_index=True)
    else:
        st.info("No students registered yet.")
    st.subheader("Attendance history")
    st.dataframe(attendance.sort_values(["Date", "Time"], ascending=False) if not attendance.empty else attendance,
                 use_container_width=True, hide_index=True)


def show_teacher(database, attendance):
    st.subheader("Start a class attendance session")
    session_name = st.text_input("Class / Subject / Session name", placeholder="Example: DBMS - 3rd Period")
    st.caption("Click Start.The camera checks a randomized blink + head-turn challenge before face matching.")
    if st.button(" Start Face Recognition Attendance", type="primary", use_container_width=True):
        if not session_name.strip():
            st.warning("Enter a class or session name first.")
        else:
            pending = run_camera_attendance(database, session_name.strip())
            if pending is not None:
                st.session_state["pending_new_embedding"] = pending
                st.session_state["pending_session_name"] = session_name.strip()
                st.rerun()
            else:
                st.info("Camera session ended. No new student was registered. Only verified, matched students can be marked present.")

    if "pending_new_embedding" in st.session_state:
        st.divider()
        st.subheader("New student detected — register details")
        st.warning("Liveness challenge passed, but the face is not in the student database. Verify the student's identity before registering.")
        with st.form("new_student_registration"):
            new_name = st.text_input("Student full name")
            new_roll = st.text_input("Student roll number")
            confirm = st.form_submit_button("Register student & mark present", type="primary")

        if confirm:
            name = new_name.strip()
            roll = new_roll.strip()
            if not name or not roll:
                st.error("Both student name and roll number are required.")
            elif normalize(roll) in database:
                st.error("This roll number is already registered. Start attendance again to let the system recognize this student.")
            else:
                vector = np.asarray(st.session_state["pending_new_embedding"], dtype=np.float32)
                database[normalize(roll)] = {
                    "name": name,
                    "roll_no": roll,
                    "embeddings": [vector],
                }
                save_students(database)
                mark_present(roll, name, st.session_state["pending_session_name"])
                del st.session_state["pending_new_embedding"]
                del st.session_state["pending_session_name"]
                st.success(f"{name} ({roll}) registered and marked Present.")
                st.rerun()

    st.divider()
    st.subheader("Attendance records")
    latest = load_attendance()
    st.dataframe(latest.sort_values(["Date", "Time"], ascending=False) if not latest.empty else latest,
                 use_container_width=True, hide_index=True)


def show_student(database, attendance):
    st.subheader("Find your attendance")
    lookup = st.text_input("Enter your roll number or full name", placeholder="Example: 22CSE123")
    if not lookup.strip():
        st.info("Enter your roll number (recommended) or your registered full name.")
        return

    key = normalize(lookup)
    matches = []
    if key in database:
        matches = [database[key]]
    else:
        matches = [v for v in database.values() if normalize(v.get("name", "")) == key]

    if not matches:
        st.warning("No registered student found. Check the spelling or use your roll number.")
        return
    if len(matches) > 1:
        st.warning("More than one student has this name. Please search using your unique roll number.")
        return

    student = matches[0]
    roll = str(student.get("roll_no", ""))
    student_rows = attendance[attendance["RollNo"].astype(str).map(normalize) == normalize(roll)].copy()

    st.success(f"Welcome, {student.get('name', '')}!")
    c1, c2 = st.columns(2)
    c1.metric("Roll number", roll)
    c2.metric("Present sessions", int((student_rows["Status"] == "Present").sum()))

    if student_rows.empty:
        st.info("Your profile is registered, but no attendance has been recorded yet.")
    else:
        st.subheader("Your attendance history")
        st.dataframe(student_rows.sort_values(["Date", "Time"], ascending=False),
                     use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="AttendX", page_icon="📷", layout="wide")
    st.markdown(
    """
    <h1 style="
        text-align: center;
        font-size: 45px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 30px;
    ">
        AttendX
    </h1>
    """,
    unsafe_allow_html=True
)
    database = load_students()
    attendance = load_attendance()

    role = st.sidebar.selectbox("Open dashboard", ["Admin", "Teacher", "Student"])
    if role == "Teacher":
        st.header("Teacher Dashboard")
        show_teacher(database, attendance)
    elif role == "Student":
        st.header("Student Dashboard")
        show_student(database, attendance)
    else:
        st.header("Admin Dashboard")
        show_admin(database, attendance)


if __name__ == "__main__":
    main()
