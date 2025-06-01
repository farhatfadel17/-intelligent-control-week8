import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime
import os
import csv

# Path model dan gambar
MODEL_PATH = "best (3).pt"
IMAGE_PATH = "coba.jpeg"  # Ganti sesuai file Anda
CSV_FILE = "hasil_deteksi_gambar.csv"

# Inisialisasi file CSV jika belum ada
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Waktu", "Nama File", "Jumlah Intrusion", "Jumlah Trespasser",
            "Akurasi (%)", "Preprocess (ms)", "Inference (ms)", "Postprocess (ms)"
        ])

# Load model
try:
    model = YOLO(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# Deteksi dari gambar
def detect_from_image(image_path, low_threshold=50, high_threshold=150):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Gambar tidak ditemukan!")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame_rgb)
    predictions = results[0]
    class_names = model.names
    speed = predictions.speed  # waktu proses

    mask_model = np.zeros_like(edges)
    overlay = frame.copy()

    count_intrusion = 0
    count_trespasser = 0
    total_confidence = 0
    total_predictions = 0

    # Instance segmentation
    if predictions.masks is not None and predictions.masks.xy:
        for mask in predictions.masks.xy:
            mask = np.array(mask, np.int32)
            cv2.fillPoly(overlay, [mask], (255, 0, 0))
            cv2.polylines(frame, [mask], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(mask_model, [mask], 255)

    # Bounding box & label
    if predictions.boxes is not None and len(predictions.boxes.xyxy) > 0:
        for i, box in enumerate(predictions.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), thickness=cv2.FILLED)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(mask_model, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)

            # Akurasi (confidence)
            if hasattr(predictions.boxes, "conf"):
                conf = float(predictions.boxes.conf[i])
                total_confidence += conf
                total_predictions += 1

            # Label class
            if hasattr(predictions.boxes, "cls"):
                class_id = int(predictions.boxes.cls[i])
                class_name = class_names.get(class_id, f"ID {class_id}")
                if class_name.lower() == "intrusion":
                    count_intrusion += 1
                elif class_name.lower() == "trespasser":
                    count_trespasser += 1
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Gabungkan mask
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Tambahkan hasil Canny pada sisi kanan
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_bgr[np.where((mask_model == 255) & (edges == 255))] = [0, 0, 255]
    edges_bgr[np.where((mask_model == 0) & (edges == 255))] = [255, 255, 255]

    combined = np.hstack((frame, edges_bgr))

    # Simpan hasil deteksi sebagai gambar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"hasil_deteksi_{timestamp}.jpg"
    cv2.imwrite(output_file, combined)
    print(f"Hasil disimpan: {output_file}")

    # Hitung akurasi rata-rata
    if total_predictions > 0:
        avg_conf = total_confidence / total_predictions
        avg_conf_percent = round(avg_conf * 100, 2)
    else:
        avg_conf_percent = 0.0

    # Ambil timing
    preprocess = speed.get('preprocess', 0.0)
    inference = speed.get('inference', 0.0)
    postprocess = speed.get('postprocess', 0.0)

    # Tulis ke CSV
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp, os.path.basename(image_path), count_intrusion, count_trespasser,
            avg_conf_percent, f"{preprocess:.1f}", f"{inference:.1f}", f"{postprocess:.1f}"
        ])

    # Tampilkan hasil
    cv2.imshow("Hasil Deteksi + Canny", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Jalankan
detect_from_image(IMAGE_PATH)