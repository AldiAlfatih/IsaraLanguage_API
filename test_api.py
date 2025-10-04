import requests
import base64
import cv2
import json
import os
import time

TEST_VIDEO_FOLDER = r'data_test' 

FLASK_API_URL = 'http://127.0.0.1:5000/predict'
SEQUENCE_LENGTH = 30

def prepare_test_data(video_path):
    print(f"\nMempersiapkan data dari video: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Gagal membuka video.")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < SEQUENCE_LENGTH:
        print(f"Error: Video terlalu pendek, hanya {total_frames} frame.")
        return None
    base64_frames = []
    frame_interval = max(int(total_frames / SEQUENCE_LENGTH), 1)
    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Gagal membaca frame ke-{i}")
            return None
        _, buffer = cv2.imencode('.jpg', frame)
        b64_string = base64.b64encode(buffer).decode('utf-8')
        base64_frames.append(b64_string)
    cap.release()
    payload = {"frames": base64_frames}
    return json.dumps(payload)

def test_all_videos_in_folder():
    if not os.path.isdir(TEST_VIDEO_FOLDER):
        print(f"[ERROR] Folder tes tidak ditemukan di: {TEST_VIDEO_FOLDER}")
        return

    video_files = [f for f in os.listdir(TEST_VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"Tidak ada file video yang ditemukan di folder '{TEST_VIDEO_FOLDER}'.")
        return

    print(f"--- Memulai Pengujian API pada {len(video_files)} Video ---")
    headers = {'Content-Type': 'application/json'}

    for video_file in video_files:
        video_path = os.path.join(TEST_VIDEO_FOLDER, video_file)
        payload = prepare_test_data(video_path)
        
        if payload is None:
            continue

        print(f"Mengirim request untuk {video_file}...")
        try:
            response = requests.post(FLASK_API_URL, data=payload, headers=headers)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("--> Hasil Prediksi:", response.json())
            else:
                print("--> Error Response:", response.text)
        except requests.exceptions.ConnectionError:
            print(f"\n[ERROR] Tidak bisa terhubung ke server. Pastikan 'python app.py' sudah berjalan.")
            return
        
        time.sleep(1) 

    print("\n--- Pengujian Selesai ---")


if __name__ == '__main__':
    test_all_videos_in_folder()