from flask import Flask, render_template, request, redirect, session, jsonify, flash
import sqlite3
import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
import speech_recognition as sr
import mediapipe as mp
from collections import deque
import tensorflow as tf
import threading
from datetime import datetime

# =================================================
# APP CONFIG
# =================================================
app = Flask(__name__)
app.secret_key = "secret123"

from pydub import AudioSegment





# ================= REALTIME ALERT FLAG =================
realtime_alert_flag = {}
EMERGENCY_VOICE_KEYWORDS = [
    "help",
    "save me",
    "danger",
    "emergency",
    "doctor",
    "hospital",
    "pain",
    "not safe"
]



EMERGENCY_GESTURES = [
    "I need help",
    "Emergency",
    "I feel uncomfortable"
]

# =================================================
# DATABASE INITIALIZATION WITH MIGRATION
# =================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SENDER_EMAIL = "shreelakshmi112.k@gmail.com"
SENDER_PASSWORD = "bivhdztfumfpbzuu"

def send_alert_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print(f"📧 Email sent successfully to {to_email}")

    except smtplib.SMTPAuthenticationError:
        print("❌ Gmail authentication failed – check APP PASSWORD")
    except Exception as e:
        print(f"❌ Email failed: {e}")


def get_caretaker_email(patient_name):
    db = get_db()
    row = db.execute("""
        SELECT caretaker_email 
        FROM patient 
        WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
          AND caretaker_email IS NOT NULL
          AND caretaker_email != ''
    """, (patient_name,)).fetchone()
    db.close()
    return row["caretaker_email"] if row else None




def check_and_add_column(table_name, column_name, column_type):
    """Check if column exists, add if it doesn't"""
    db = get_db()
    cursor = db.cursor()
    
    try:
        # Try to select the column
        cursor.execute(f"SELECT {column_name} FROM {table_name} LIMIT 1")
        print(f"✅ Column '{column_name}' already exists in '{table_name}'")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        print(f"➕ Adding column '{column_name}' to '{table_name}'...")
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        db.commit()
        print(f"✅ Column '{column_name}' added to '{table_name}'")
    
    db.close()

def init_db():
    db = get_db()
    c = db.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS patient (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age TEXT,
        location TEXT,
        phone TEXT,
        password TEXT,
        caretaker_name TEXT,
        caretaker_phone TEXT,
        caretaker_email TEXT
    )
    """)



    c.execute("""
    CREATE TABLE IF NOT EXISTS caretaker (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        phone TEXT,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        msg_type TEXT,
        message TEXT,
        timestamp TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT,
        tablet_name TEXT,
        task_time TEXT,
        status TEXT,
        timestamp TEXT
    )
    """)

    db.commit()
    db.close()
    
    # Check and add timestamp columns if needed
    check_and_add_column("messages", "timestamp", "TEXT")
    
    check_and_add_column("patient", "caretaker_email", "TEXT")

    check_and_add_column("tasks", "timestamp", "TEXT")
    
    # Initialize timestamps for existing records
    initialize_timestamps()

def initialize_timestamps():
    """Set default timestamps for existing records that don't have them"""
    db = get_db()
    cursor = db.cursor()
    
    # Initialize messages timestamps
    cursor.execute("UPDATE messages SET timestamp = ? WHERE timestamp IS NULL", 
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
    
    # Initialize tasks timestamps
    cursor.execute("UPDATE tasks SET timestamp = ? WHERE timestamp IS NULL", 
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
    
    db.commit()
    db.close()
    print("✅ Timestamps initialized for existing records")

init_db()

# =================================================
# HELPER FUNCTIONS
# =================================================
def get_current_timestamp():
    """Get current timestamp in string format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =================================================
# LOAD EMOTION MODEL
# =================================================
try:
    emotion_model = load_model("emotion_model.h5", compile=False)
    print("✅ Emotion model loaded successfully")
except:
    print("⚠️  Emotion model not found. Using dummy model.")
    emotion_model = None

EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
except:
    print("⚠️  Haar cascade not found")
    face_cascade = None

# =================================================
# HOME
# =================================================
@app.route("/")
def index():
    return render_template("index.html")

# =================================================
# PATIENT REGISTER
# =================================================
@app.route("/patient_register", methods=["GET", "POST"])
def patient_register():
    if request.method == "POST":
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
    INSERT INTO patient
    (name, age, location, phone, password, caretaker_name, caretaker_phone, caretaker_email)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (
    request.form["name"].strip().lower(),
    request.form["age"].strip(),
    request.form["location"].strip(),
    request.form["phone"].strip(),
    request.form["password"].strip(),
    request.form["caretaker_name"].strip(),
    request.form["caretaker_phone"].strip(),
    request.form["caretaker_email"].strip().lower()
))

        db.commit()
        db.close()
        return redirect("/patient_login?registered=true")

    return render_template("patient_register.html")


# =================================================
# PATIENT LOGIN
# =================================================
@app.route("/patient_login", methods=["GET", "POST"])
def patient_login():
    if request.method == "POST":
        # ✅ Clean inputs
        name = request.form.get("name", "").strip().lower()
        password = request.form.get("password", "").strip()

        db = get_db()
        user = db.execute("""
            SELECT * FROM patient
            WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
              AND TRIM(password) = TRIM(?)
        """, (name, password)).fetchone()
        db.close()

        # 🔍 Debug
        print("LOGIN ATTEMPT →", name, password)
        print("USER FOUND →", user)

        if user:
            session.clear()
            session["patient"] = user["name"]
            session["patient_id"] = user["id"]

            print("✅ LOGIN SUCCESS")
            return redirect("/patient_dashboard")

        print("❌ LOGIN FAILED")
        return render_template(
            "patient_login.html",
            error="Invalid username or password"
        )

    return render_template("patient_login.html")



# =================================================
# PATIENT DASHBOARD
# =================================================
@app.route("/patient_dashboard")
def patient_dashboard():
    if "patient" not in session:
        return redirect("/patient_login")
    
    db = get_db()
    # Get pending tasks count
    tasks_count = db.execute("""
        SELECT COUNT(*) as count FROM tasks 
        WHERE patient_name=? AND status='Pending'
    """, (session["patient"],)).fetchone()["count"]
    
    # Get recent messages - using simple query without timestamp sorting initially
    try:
        recent_messages = db.execute("""
            SELECT * FROM messages 
            WHERE patient_name=? 
            LIMIT 3
        """, (session["patient"],)).fetchall()
    except sqlite3.OperationalError:
        # If timestamp column causes issues, use simpler query
        recent_messages = db.execute("""
            SELECT id, patient_name, msg_type, message FROM messages 
            WHERE patient_name=? 
            LIMIT 3
        """, (session["patient"],)).fetchall()
    
    db.close()
    
    return render_template("patient_dashboard.html", 
                          name=session["patient"], 
                          tasks_count=tasks_count,
                          recent_messages=recent_messages)

# =================================================
# EMOTION PAGE
# =================================================

EMERGENCY_EMOTIONS = ["Angry", "Fear", "Sad"]

@app.route("/emotion")
def emotion():
    if "patient" not in session:
        return redirect("/patient_login")
    return render_template("emotion.html")

def run_emotion_camera(seconds, patient_name):
    # -------------------------------------------------
    # SAFETY CHECKS
    # -------------------------------------------------
    if face_cascade is None or emotion_model is None:
        print("❌ Emotion detection components not available")

        db = get_db()
        db.execute("""
            INSERT INTO messages (patient_name, msg_type, message, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            patient_name,
            "emotion",
            "Emotion detection not available",
            get_current_timestamp()
        ))
        db.commit()
        db.close()
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    # -------------------------------------------------
    # INITIALIZE VARIABLES
    # -------------------------------------------------
    start_time = time.time()
    emotion_count = {e: 0 for e in EMOTIONS}
    total_frames = 0
    MIN_FRAMES_REQUIRED = 10   # avoid false emotion from few frames

    # -------------------------------------------------
    # CAMERA LOOP
    # -------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face.astype("float32") / 255.0
                face = face.reshape(1, 48, 48, 1)

                preds = emotion_model.predict(face, verbose=0)
                label = EMOTIONS[np.argmax(preds)]

                emotion_count[label] += 1
                total_frames += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            except Exception as e:
                print(f"⚠️ Face processing error: {e}")
                continue

        # Show counts on screen
        y_offset = 30
        for emo, cnt in emotion_count.items():
            cv2.putText(frame, f"{emo}: {cnt}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y_offset += 22

        cv2.imshow("Emotion Detection (Press Q to quit)", frame)

        # Time limit
        if seconds > 0 and (time.time() - start_time) >= seconds:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # -------------------------------------------------
    # FINAL EMOTION DECISION
    # -------------------------------------------------
    if total_frames < MIN_FRAMES_REQUIRED:
        final_emotion = "No face detected"
    else:
        final_emotion = max(emotion_count, key=emotion_count.get)

    emotion_messages = {
        "Angry": "Patient seems angry",
        "Fear": "Patient shows fear",
        "Sad": "Patient seems sad",
        "Happy": "Patient is happy",
        "Neutral": "Patient appears neutral",
        "No face detected": "No clear face detected"
    }

    message = emotion_messages.get(final_emotion, final_emotion)

    # -------------------------------------------------
    # SAVE TO DATABASE
    # -------------------------------------------------
    db = get_db()
    db.execute("""
        INSERT INTO messages (patient_name, msg_type, message, timestamp)
        VALUES (?, ?, ?, ?)
    """, (
        patient_name,
        "emotion",
        message,
        get_current_timestamp()
    ))
    db.commit()
    db.close()

    print(f"✅ Emotion saved for {patient_name}: {final_emotion}")

    # -------------------------------------------------
    # 🚨 EMERGENCY ALERT LOGIC
    # -------------------------------------------------
    # -------------------------------------------------
# 🚨 EMERGENCY EMAIL ALERT (ONLY NEGATIVE EMOTIONS)
# -------------------------------------------------
    if final_emotion in EMERGENCY_EMOTIONS:
        realtime_alert_flag[patient_name] = True

        email = get_caretaker_email(patient_name)
        if email:
            send_alert_email(
                email,
                "🚨 EMERGENCY ALERT - Patient Emotional Distress",
                f"""
    Emergency Emotion Alert!

    Patient Name : {patient_name}
    Detected Emotion : {final_emotion}
    Time : {get_current_timestamp()}

    ⚠ Immediate attention may be required.
    Please check the caretaker dashboard.
    """
            )
        else:
            print("❌ No caretaker email found — email skipped")
    else:
        print(f"ℹ️ Emotion '{final_emotion}' detected — no emergency email sent")


# =================================================
# EMOTION DETECTION
# =================================================
@app.route("/emotion_detect/<int:seconds>")
def emotion_detect(seconds):
    if "patient" not in session:
        return redirect("/patient_login")

    threading.Thread(
        target=run_emotion_camera,
        args=(seconds, session["patient"]),
        daemon=True
    ).start()

    return redirect("/patient_dashboard")

# =================================================
# VOICE MESSAGE
# =================================================
from flask import flash
@app.route("/voice", methods=["GET", "POST"])
def voice():
    if "patient" not in session:
        return redirect("/patient_login")

    if request.method == "POST":
        if "audio" not in request.files:
            flash("No audio received")
            return redirect("/voice")

        audio_file = request.files["audio"]
        timestamp = get_current_timestamp()
        text = "Voice not clear"

        # -------------------------------
        # SAVE AUDIO
        # -------------------------------
        webm_path = "temp_voice.webm"
        wav_path = "temp_voice.wav"
        audio_file.save(webm_path)

        # -------------------------------
        # CONVERT WEBM → WAV
        # -------------------------------
        try:
            from pydub import AudioSegment
            AudioSegment.from_file(webm_path).export(wav_path, format="wav")
        except Exception as e:
            flash("Audio conversion failed")
            print("❌ Audio conversion error:", e)
            return redirect("/voice")

        # -------------------------------
        # SPEECH RECOGNITION
        # -------------------------------
        r = sr.Recognizer()
        try:
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)
            print(f"🗣 Recognized: {text}")

        except sr.UnknownValueError:
            text = "Could not understand audio"
        except sr.RequestError as e:
            text = f"Speech recognition error: {e}"
        except Exception as e:
            text = f"Error: {str(e)}"

        # -------------------------------
        # SAVE MESSAGE TO DATABASE
        # -------------------------------
        db = get_db()
        db.execute("""
            INSERT INTO messages (patient_name, msg_type, message, timestamp)
            VALUES (?, ?, ?, ?)
        """, (session["patient"], "voice", text, timestamp))
        db.commit()
        db.close()

        flash(text)
        print(f"✅ Voice saved for {session['patient']}")

        # -------------------------------
        # 🚨 EMERGENCY DETECTION (FIXED)
        # -------------------------------
        import re

        # Clean text (remove punctuation)
        clean_text = re.sub(r"[^a-zA-Z ]", "", text.lower())

        EMERGENCY_VOICE_KEYWORDS = [
            "help",
            "save me",
            "danger",
            "emergency",
            "doctor",
            "hospital",
            "pain",
            "not safe"
        ]

        if any(keyword in clean_text for keyword in EMERGENCY_VOICE_KEYWORDS):
            realtime_alert_flag[session["patient"]] = True
            print(f"🚨 Emergency Voice Alert for {session['patient']}")

            email = get_caretaker_email(session["patient"])
            if email:
                send_alert_email(
                    email,
                    "🚨 EMERGENCY ALERT - Voice Detected",
                    f"""
Emergency Voice Alert!

Patient : {session['patient']}
Message : {text}
Time    : {timestamp}

Immediate attention required.
"""
                )
            else:
                print("❌ No caretaker email found — email skipped")
        else:
            print("ℹ️ Voice detected but NOT emergency")

        return redirect("/voice")

    return render_template("voice.html")


# =================================================
# CARETAKER REGISTER
# =================================================
@app.route("/caretaker_register", methods=["GET", "POST"])
def caretaker_register():
    if request.method == "POST":
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO caretaker (name, phone, password)
            VALUES (?, ?, ?)
        """, (
            request.form["name"],
            request.form["phone"],
            request.form["password"]
        ))
        db.commit()
        db.close()
        return redirect("/caretaker_login")

    return render_template("caretaker_register.html")

# =================================================
# CARETAKER LOGIN
# =================================================
@app.route("/caretaker_login", methods=["GET", "POST"])
def caretaker_login():
    if request.method == "POST":
        db = get_db()
        user = db.execute("""
            SELECT * FROM caretaker WHERE name=? AND password=?
        """, (
            request.form["name"],
            request.form["password"]
        )).fetchone()
        db.close()

        if user:
            session["caretaker"] = user["name"]
            session["caretaker_id"] = user["id"]
            return redirect("/caretaker_dashboard")
        else:
            return render_template("caretaker_login.html", error="Invalid credentials")

    return render_template("caretaker_login.html")

# =================================================
# CARETAKER DASHBOARD
# =================================================
@app.route("/caretaker_dashboard")
def caretaker_dashboard():
    if "caretaker" not in session:
        return redirect("/caretaker_login")

    # Get message type from query parameter
    msg_type = request.args.get('type', 'gesture')
    valid_types = ['gesture', 'voice', 'emotion', 'task', 'all']
    if msg_type not in valid_types:
        msg_type = 'gesture'

    db = get_db()
    
    # Get messages based on type
    try:
        if msg_type == 'all':
            messages = db.execute("""
                SELECT * FROM messages 
                ORDER BY timestamp DESC 
                LIMIT 20
            """).fetchall()
        else:
            messages = db.execute("""
                SELECT * FROM messages 
                WHERE msg_type=?
                ORDER BY timestamp DESC 
                LIMIT 20
            """, (msg_type,)).fetchall()
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        if msg_type == 'all':
            messages = db.execute("""
                SELECT id, patient_name, msg_type, message FROM messages 
                LIMIT 20
            """).fetchall()
        else:
            messages = db.execute("""
                SELECT id, patient_name, msg_type, message FROM messages 
                WHERE msg_type=?
                LIMIT 20
            """, (msg_type,)).fetchall()
    
    # Convert messages to list of dictionaries
    messages_list = []
    for msg in messages:
        try:
            msg_dict = dict(msg)
            if 'timestamp' not in msg_dict:
                msg_dict['timestamp'] = 'N/A'
            messages_list.append(msg_dict)
        except Exception as e:
            print(f"Error converting message: {e}")
            continue
    
    # Get counts for all message types
    try:
        gesture_count = db.execute("""
            SELECT COUNT(*) as count FROM messages 
            WHERE msg_type='gesture'
        """).fetchone()["count"]
        voice_count = db.execute("""
            SELECT COUNT(*) as count FROM messages 
            WHERE msg_type='voice'
        """).fetchone()["count"]
        emotion_count = db.execute("""
            SELECT COUNT(*) as count FROM messages 
            WHERE msg_type='emotion'
        """).fetchone()["count"]
        task_count = db.execute("""
            SELECT COUNT(*) as count FROM messages 
            WHERE msg_type='task'
        """).fetchone()["count"]
        all_count = db.execute("""
            SELECT COUNT(*) as count FROM messages 
        """).fetchone()["count"]
    except:
        gesture_count = 0
        voice_count = 0
        emotion_count = 0
        task_count = 0
        all_count = 0
    
    # Get all patients
    patients = db.execute("SELECT name FROM patient").fetchall()
    patients_list = []
    for p in patients:
        try:
            patients_list.append(dict(p))
        except:
            patients_list.append({'name': 'Unknown'})
    
    # Get tasks
    try:
        tasks = db.execute("""
            SELECT * FROM tasks 
            ORDER BY timestamp DESC
            LIMIT 10
        """).fetchall()
    except sqlite3.OperationalError:
        tasks = db.execute("""
            SELECT id, patient_name, tablet_name, task_time, status FROM tasks
            LIMIT 10
        """).fetchall()
    
    tasks_list = []
    for t in tasks:
        try:
            tasks_list.append(dict(t))
        except:
            continue
    
    # Get stats
    pending_tasks = db.execute("""
        SELECT COUNT(*) as count FROM tasks 
        WHERE status='Pending'
    """).fetchone()["count"]
    
    total_patients = db.execute("""
        SELECT COUNT(*) as count FROM patient
    """).fetchone()["count"]
    
    # Set recent alerts based on current message type
    if msg_type == 'gesture':
        recent_alerts = gesture_count
    elif msg_type == 'voice':
        recent_alerts = voice_count
    elif msg_type == 'emotion':
        recent_alerts = emotion_count
    elif msg_type == 'task':
        recent_alerts = task_count
    else:
        recent_alerts = all_count
    
    db.close()

    return render_template(
        "caretaker_dashboard.html",
        messages=messages_list,
        tasks=tasks_list,
        patients=patients_list,
        pending_tasks=pending_tasks,
        total_patients=total_patients,
        recent_alerts=recent_alerts,
        voice_count=voice_count,
        emotion_count=emotion_count,
        task_count=task_count,
        all_count=all_count,
        current_type=msg_type,
        caretaker_name=session.get('caretaker', 'Caretaker')
    )

# =================================================
# GET MESSAGES BY TYPE (JSON API)
# =================================================
@app.route("/messages/<msg_type>")
def get_messages_by_type(msg_type):   # ✅ COLON FIXED
    if "caretaker" not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    db = get_db()

    valid_types = ['gesture', 'voice', 'emotion', 'task', 'all']
    if msg_type not in valid_types:
        msg_type = 'gesture'

    try:
        if msg_type == 'all':
            messages = db.execute("""
                SELECT * FROM messages
                ORDER BY timestamp DESC
                LIMIT 20
            """).fetchall()
        else:
            messages = db.execute("""
                SELECT * FROM messages
                WHERE msg_type = ?
                ORDER BY timestamp DESC
                LIMIT 20
            """, (msg_type,)).fetchall()

        messages_list = []
        for msg in messages:
            msg_dict = dict(msg)
            msg_dict.setdefault('timestamp', 'N/A')
            messages_list.append(msg_dict)

        db.close()
        return jsonify(messages_list)

    except Exception as e:
        db.close()
        print(f"❌ Error fetching messages: {e}")
        return jsonify({'error': str(e)}), 500

    
# =================================================
# ADD TABLET TASK
# =================================================
@app.route("/add_task", methods=["GET", "POST"])
def add_task():
    if "caretaker" not in session:
        return redirect("/caretaker_login")

    if request.method == "POST":
        db = get_db()
        db.execute("""
            INSERT INTO tasks (patient_name, tablet_name, task_time, status, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            request.form["patient_name"],
            request.form["tablet"],
            request.form["time"],
            'Pending',
            get_current_timestamp()
        ))
        db.commit()
        db.close()
        return redirect("/caretaker_dashboard")

    # Get patient list for dropdown
    db = get_db()
    patients = db.execute("SELECT name FROM patient").fetchall()
    db.close()
    
    return render_template("add_task.html", patients=patients)

# =================================================
# PATIENT TASKS
# =================================================
@app.route("/patient_tasks")
def patient_tasks():
    if "patient" not in session:
        return redirect("/patient_login")

    db = get_db()
    try:
        tasks = db.execute("""
            SELECT * FROM tasks
            WHERE patient_name=? AND status='Pending'
            ORDER BY task_time
        """, (session["patient"],)).fetchall()
    except sqlite3.OperationalError:
        # Without timestamp
        tasks = db.execute("""
            SELECT id, patient_name, tablet_name, task_time, status FROM tasks
            WHERE patient_name=? AND status='Pending'
            ORDER BY task_time
        """, (session["patient"],)).fetchall()
    db.close()

    return render_template("patient_tasks.html", tasks=tasks)

# =================================================
# COMPLETE TASK
# =================================================
@app.route("/complete_task/<int:task_id>")
def complete_task(task_id):
    if "patient" not in session:
        return redirect("/patient_login")

    db = get_db()
    
    try:
        # Get task details before updating
        task = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    except sqlite3.OperationalError:
        # Without timestamp
        task = db.execute("SELECT id, patient_name, tablet_name, task_time, status FROM tasks WHERE id=?", 
                         (task_id,)).fetchone()
    
    if task and task["patient_name"] == session["patient"]:
        # Update task status
        db.execute("""
            UPDATE tasks SET status='Completed' WHERE id=?
        """, (task_id,))

        # Add completion message
        db.execute("""
            INSERT INTO messages (patient_name, msg_type, message, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            session["patient"],
            "task",
            f"Completed: {task['tablet_name']} at {task['task_time']}",
            get_current_timestamp()
        ))

        db.commit()
    
    db.close()

    return redirect("/patient_dashboard")

# =================================================
# DELETE MESSAGE (CARETAKER)
# =================================================
@app.route("/delete_message/<int:msg_id>")
def delete_message(msg_id):
    if "caretaker" not in session:
        return redirect("/caretaker_login")
    
    db = get_db()
    db.execute("DELETE FROM messages WHERE id=?", (msg_id,))
    db.commit()
    db.close()
    
    return redirect("/caretaker_dashboard")

# =================================================
# DELETE TASK (CARETAKER)
# =================================================
@app.route("/delete_task/<int:task_id>")
def delete_task(task_id):
    if "caretaker" not in session:
        return redirect("/caretaker_login")
    
    db = get_db()
    db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    db.commit()
    db.close()
    
    return redirect("/caretaker_dashboard")

# =================================================
# LOAD GESTURE MODEL
# =================================================
try:
    gesture_model = tf.keras.models.load_model("hand_landmark_model.h5")
    print("✅ Gesture model loaded successfully")
except:
    print("⚠️  Gesture model not found. Using dummy model.")
    gesture_model = None

GESTURE_CLASSES = {
    'a': "I need help",
    'b': "I need food",
    'c': "I need water",
    'd': "Emergency",
    'e': "I feel uncomfortable",
    'f': "I am feeling good"
}

CLASS_ORDER = ['a','b','c','d','e','f']

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

@app.route("/gesture")
def gesture():
    if "patient" not in session:
        return redirect("/patient_login")
    return render_template("gesture.html")

def run_gesture_camera(patient_name):
    global realtime_alert_flag

    print(f"🎥 Gesture camera started for patient: {patient_name}")

    # =================================================
    # MODEL CHECK
    # =================================================
    if gesture_model is None:
        print("❌ Gesture model not available")

        timestamp = get_current_timestamp()

        db = get_db()
        db.execute("""
            INSERT INTO messages (patient_name, msg_type, message, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            patient_name,
            "gesture",
            "Gesture detection not available",
            timestamp
        ))
        db.commit()
        db.close()

        realtime_alert_flag[patient_name] = True
        print(f"🚨 Realtime alert triggered for {patient_name}")

        # 📧 EMAIL ALERT (SAFE)
        email = get_caretaker_email(patient_name)
        print("👤 Caretaker email fetched:", repr(email))

        if email:
            send_alert_email(
                email,
                "🚨 EMERGENCY ALERT - Gesture System Error",
                f"""
Emergency Alert!

Patient: {patient_name}
Issue: Gesture detection system not available
Time: {timestamp}

Please check caretaker dashboard.
"""
            )
        else:
            print("❌ No caretaker email found — email skipped")

        return

    # =================================================
    # CAMERA INIT
    # =================================================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera for gesture detection")
        return

    prediction_queue = deque(maxlen=10)
    detected_message = None
    gesture_start_time = None
    gesture_hold_time = 2  # seconds

    print("📷 Camera opened successfully")

    # =================================================
    # CAMERA LOOP
    # =================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_message = None

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data = np.array(data).reshape(1, -1)

            probs = gesture_model.predict(data, verbose=0)[0]
            class_id = np.argmax(probs)
            confidence = float(np.max(probs))

            if confidence > 0.8:
                prediction_queue.append(class_id)

                if len(prediction_queue) >= 5:
                    final_class = max(set(prediction_queue), key=prediction_queue.count)
                    label = CLASS_ORDER[final_class]
                    current_message = GESTURE_CLASSES[label]

                    if detected_message != current_message:
                        detected_message = current_message
                        gesture_start_time = time.time()

                    elif gesture_start_time and (time.time() - gesture_start_time) >= gesture_hold_time:
                        timestamp = get_current_timestamp()

                        # =================================================
                        # SAVE GESTURE
                        # =================================================
                        db = get_db()
                        db.execute("""
                            INSERT INTO messages (patient_name, msg_type, message, timestamp)
                            VALUES (?, ?, ?, ?)
                        """, (
                            patient_name,
                            "gesture",
                            detected_message,
                            timestamp
                        ))
                        db.commit()
                        db.close()

                        print(f"✅ Gesture saved: {detected_message}")
                        realtime_alert_flag[patient_name] = True
                        print(f"🚨 Realtime alert triggered for {patient_name}")

                        # =================================================
                        # EMAIL ALERT (SAFE)
                        # =================================================
                        email = get_caretaker_email(patient_name)
                        print("👤 Caretaker email fetched:", repr(email))

                        if email:
                            send_alert_email(
                                email,
                                "🚨 EMERGENCY ALERT - Gesture Detected",
                                f"""
Emergency Gesture Alert!

Patient: {patient_name}
Gesture Detected: {detected_message}
Time: {timestamp}

Please check caretaker dashboard immediately.
"""
                            )
                        else:
                            print("❌ No caretaker email found — email skipped")

                        gesture_start_time = None

                        cv2.putText(
                            frame,
                            "Gesture Saved!",
                            (180, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0, 255, 0),
                            3
                        )

            if current_message:
                cv2.putText(frame, f"Gesture: {current_message}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gesture Detection - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # =================================================
    # CLEANUP
    # =================================================
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Gesture camera stopped")



@app.route("/gesture_detect")
def gesture_detect():
    if "patient" not in session:
        return redirect("/patient_login")

    threading.Thread(
        target=run_gesture_camera,
        args=(session["patient"],),
        daemon=True
    ).start()

    return redirect("/patient_dashboard")

# =================================================
# VIEW PATIENT PROFILE
# =================================================
@app.route("/view_patient/<name>")
def view_patient(name):
    if "caretaker" not in session:
        return redirect("/caretaker_login")
    
    db = get_db()
    patient = db.execute("SELECT * FROM patient WHERE name=?", (name,)).fetchone()
    
    if patient:
        try:
            # Get patient's messages with timestamp
            messages = db.execute("""
                SELECT * FROM messages 
                WHERE patient_name=? 
                ORDER BY timestamp DESC 
                LIMIT 10
            """, (name,)).fetchall()
        except sqlite3.OperationalError:
            # Without timestamp
            messages = db.execute("""
                SELECT id, patient_name, msg_type, message FROM messages 
                WHERE patient_name=? 
                LIMIT 10
            """, (name,)).fetchall()
        
        try:
            # Get patient's tasks with timestamp
            tasks = db.execute("""
                SELECT * FROM tasks 
                WHERE patient_name=? 
                ORDER BY timestamp DESC
            """, (name,)).fetchall()
        except sqlite3.OperationalError:
            # Without timestamp
            tasks = db.execute("""
                SELECT id, patient_name, tablet_name, task_time, status FROM tasks 
                WHERE patient_name=?
            """, (name,)).fetchall()
        
        db.close()
        
        return render_template("view_patient.html", 
                              patient=patient, 
                              messages=messages, 
                              tasks=tasks)
    
    db.close()
    return redirect("/caretaker_dashboard")

# =================================================
# RESET DATABASE (DEVELOPMENT ONLY)
# =================================================
@app.route("/reset_db")
def reset_db():
    """Development only - reset database completely"""
    import os
    if os.path.exists("database.db"):
        os.remove("database.db")
        print("🗑️  Database deleted")
    
    init_db()
    return "Database reset complete. <a href='/'>Go Home</a>"


@app.route("/realtime_alert")
def realtime_alert():
    if "patient" not in session:
        return jsonify({"new": False})

    patient = session["patient"]

    if realtime_alert_flag.get(patient):
        realtime_alert_flag[patient] = False
        return jsonify({"new": True})

    return jsonify({"new": False})

# =================================================
# LOGOUT
# =================================================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# =================================================
# RUN
# =================================================
if __name__ == "__main__":
    print("🚀 Starting Flask application...")
    print("📊 Database initialized")

    # Render / production safe run
    app.run(host="0.0.0.0", port=5000)
