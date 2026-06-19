from flask import Flask, render_template, Response, jsonify
import cv2
from picamera2 import Picamera2
import numpy as np
import sounddevice as sd
import os
import wave
import shutil
import subprocess
from datetime import datetime
from threading import Lock
import atexit
import time
import sqlite3

# Cascade dosyaları app.py ile aynı klasörde olmalı
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
frontal_path = os.path.join(LOCAL_DIR, 'haarcascade_frontalface_default.xml')
profile_path = os.path.join(LOCAL_DIR, 'haarcascade_profileface.xml')

face_cascade = cv2.CascadeClassifier(frontal_path)
profile_cascade = cv2.CascadeClassifier(profile_path)

if face_cascade.empty():
    print('UYARI: Frontal face cascade yüklenemedi:', frontal_path)
if profile_cascade.empty():
    print('UYARI: Profile face cascade yüklenemedi:', profile_path)

pozisyon = 'Analiz Bekleniyor'
pozisyon_lock = Lock()

# Picamera2 ile kamera başlat
camera = Picamera2()
camera.configure(camera.create_video_configuration(main={"size": (320, 240)}))
camera.start()
time.sleep(0.3)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

RECORDINGS_DIR = os.path.join(BASE_DIR, 'recordings')
os.makedirs(RECORDINGS_DIR, exist_ok=True)

DB_FILE = os.path.join(BASE_DIR, 'recordings.db')

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            timestamp TEXT,
            duration REAL,
            file_size INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

camera_lock = Lock()
recording_lock = Lock()
audio_lock = Lock()

is_recording = False
video_writer = None
current_recording_file = None
current_video_file = None
current_audio_file = None
current_final_file = None
recording_fps = 20.0
recording_start_time = None

AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 1
AUDIO_BLOCKSIZE = 1024

audio_stream = None
audio_stream_error = None
audio_wave_file = None
latest_audio_level = 0
latest_waveform = []


def get_camera_fps():
    return 20.0


def audio_callback(indata, frames, time_info, status):
    global latest_audio_level, latest_waveform
    try:
        audio_flat = indata[:, 0].copy()
        rms = np.sqrt(np.mean(audio_flat ** 2))
        level = int(rms * 12000)
        level = max(0, min(level, 100))
        sample_count = 160
        step = max(1, len(audio_flat) // sample_count)
        waveform = audio_flat[::step][:sample_count]
        waveform = np.clip(waveform * 80, -1, 1).tolist()
        audio_int16 = np.int16(np.clip(audio_flat, -1, 1) * 32767)
        with audio_lock:
            latest_audio_level = level
            latest_waveform = waveform
            if audio_wave_file is not None:
                audio_wave_file.writeframes(audio_int16.tobytes())
    except Exception:
        pass


def ensure_audio_stream():
    global audio_stream, audio_stream_error
    if audio_stream is not None:
        return True
    try:
        audio_stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype='float32',
            blocksize=AUDIO_BLOCKSIZE,
            device=2,
            callback=audio_callback
        )
        audio_stream.start()
        audio_stream_error = None
        return True
    except Exception as e:
        audio_stream = None
        audio_stream_error = str(e)
        return False


def merge_video_and_audio(video_file, audio_file, output_file):
    if not shutil.which('ffmpeg'):
        return False, 'ffmpeg bulunamadı.'
    command = [
        'ffmpeg', '-y',
        '-i', video_file,
        '-i', audio_file,
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-shortest',
        output_file
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return False, result.stderr[-700:]
        return True, 'Başarıyla birleştirildi.'
    except Exception as e:
        return False, str(e)


def generate_frames():
    global video_writer, pozisyon
    kare_sayaci = 0

    while True:
        with camera_lock:
            frame = camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with recording_lock:
            if is_recording and video_writer is not None:
                video_writer.write(frame)

        # Her 10 karede bir pozisyon tespiti yap (performans için)
        kare_sayaci += 1
        if kare_sayaci % 10 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_width = frame.shape[1]

            frontal = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

            if len(frontal) > 0:
                yeni_pozisyon = 'Sırtüstü'
                x, y, w, h = frontal[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Sirtustu', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                profile_sol = profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                if len(profile_sol) > 0:
                    yeni_pozisyon = 'Sağa Dönük'
                    x, y, w, h = profile_sol[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, 'Saga Donuk', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    gray_flipped = cv2.flip(gray, 1)
                    profile_sag = profile_cascade.detectMultiScale(gray_flipped, 1.1, 5, minSize=(60, 60))
                    if len(profile_sag) > 0:
                        yeni_pozisyon = 'Sola Dönük'
                        x_ters, y, w, h = profile_sag[0]
                        x_orijinal = frame_width - (x_ters + w)
                        cv2.rectangle(frame, (x_orijinal, y), (x_orijinal + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, 'Sola Donuk', (x_orijinal, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        yeni_pozisyon = 'Yüzüstü / Tespit Edilemiyor'

            with pozisyon_lock:
                pozisyon = yeni_pozisyon

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    ensure_audio_stream()
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, video_writer, current_recording_file
    global current_video_file, current_audio_file, current_final_file, recording_fps, audio_wave_file, recording_start_time

    with recording_lock:
        if is_recording:
            return jsonify({
                'status': 'already_recording',
                'message': 'Kayıt zaten devam ediyor.',
                'file': current_recording_file
            })

    if not ensure_audio_stream():
        return jsonify({
            'status': 'error',
            'message': 'Mikrofon başlatılamadı.',
            'error': audio_stream_error
        }), 500

    with camera_lock:
        frame = camera.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    recording_fps = get_camera_fps()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    recording_start_time = datetime.now()
    base_name = f'kayit_{timestamp}'

    video_filepath = os.path.join(RECORDINGS_DIR, f'{base_name}_video.avi')
    audio_filepath = os.path.join(RECORDINGS_DIR, f'{base_name}_ses.wav')
    final_filepath = os.path.join(RECORDINGS_DIR, f'{base_name}_sesli.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(video_filepath, fourcc, recording_fps, (width, height))

    if not writer.isOpened():
        return jsonify({
            'status': 'error',
            'message': 'Video dosyası oluşturulamadı.'
        }), 500

    try:
        wav_file = wave.open(audio_filepath, 'wb')
        wav_file.setnchannels(AUDIO_CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(AUDIO_SAMPLE_RATE)
    except Exception as e:
        writer.release()
        return jsonify({
            'status': 'error',
            'message': 'Ses dosyası oluşturulamadı.',
            'error': str(e)
        }), 500

    with audio_lock:
        audio_wave_file = wav_file

    with recording_lock:
        video_writer = writer
        current_video_file = video_filepath
        current_audio_file = audio_filepath
        current_final_file = final_filepath
        current_recording_file = final_filepath
        is_recording = True

    return jsonify({
        'status': 'recording_started',
        'message': 'Sesli kayıt başlatıldı.',
        'file': current_recording_file,
        'video_file': current_video_file,
        'audio_file': current_audio_file
    })


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording, video_writer, current_recording_file
    global current_video_file, current_audio_file, current_final_file, audio_wave_file

    with recording_lock:
        if not is_recording:
            return jsonify({'status': 'not_recording', 'message': 'Aktif kayıt yok.'})

        is_recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None

        saved_video_file = current_video_file
        saved_audio_file = current_audio_file
        saved_final_file = current_final_file
        current_recording_file = None
        current_video_file = None
        current_audio_file = None
        current_final_file = None

    with audio_lock:
        if audio_wave_file is not None:
            audio_wave_file.close()
            audio_wave_file = None

    merge_success, merge_message = merge_video_and_audio(saved_video_file, saved_audio_file, saved_final_file)

    saved_file = saved_final_file if merge_success else saved_video_file
    file_size = os.path.getsize(saved_file) if os.path.exists(saved_file) else 0
    duration = round((datetime.now() - recording_start_time).total_seconds(), 2) if recording_start_time else 0

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO recordings (file_name, timestamp, duration, file_size) VALUES (?, ?, ?, ?)',
        (saved_file, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), duration, file_size)
    )
    conn.commit()
    conn.close()

    return jsonify({
        'status': 'recording_stopped',
        'message': 'Kayıt bitirildi.',
        'file': saved_file,
        'final_file': saved_final_file if merge_success else None,
        'video_file': saved_video_file,
        'audio_file': saved_audio_file,
        'merge_status': 'merged' if merge_success else 'not_merged',
        'warning': None if merge_success else merge_message
    })


@app.route('/recording_status')
def recording_status():
    with recording_lock:
        return jsonify({
            'is_recording': is_recording,
            'file': current_recording_file,
            'video_file': current_video_file,
            'audio_file': current_audio_file
        })


@app.route('/sound_level')
def sound_level():
    if not ensure_audio_stream():
        return jsonify({'level': 0, 'waveform': [], 'error': audio_stream_error})
    with audio_lock:
        return jsonify({'level': latest_audio_level, 'waveform': latest_waveform})


@app.route('/pozisyon')
def get_pozisyon():
    with pozisyon_lock:
        return jsonify({'pozisyon': pozisyon})


@app.route('/database')
def database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM recordings ORDER BY id DESC")
    recordings = cursor.fetchall()
    conn.close()
    return render_template('database.html', recordings=recordings)


def cleanup():
    global video_writer, audio_wave_file, audio_stream
    with recording_lock:
        if video_writer is not None:
            video_writer.release()
    with audio_lock:
        if audio_wave_file is not None:
            audio_wave_file.close()
    if audio_stream is not None:
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception:
            pass
    camera.stop()


atexit.register(cleanup)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)
