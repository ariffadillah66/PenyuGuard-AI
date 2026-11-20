import math
import os
import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# Muat model hasil training
model = YOLO("best.pt")

# Parameter Von Bertalanffy (Ramirez et al., 2021 - Tabel 3 Model D)
SPECIES_INFO = {
    "green": ("Penyu Hijau (Chelonia mydas)", 110.27, 0.051, 5.0),
    "loggerhead": ("Penyu Tempayan (Caretta caretta)", 100.06, 0.066, 4.5),
    "hawksbill": ("Penyu Sisik (Eretmochelys imbricata)", 84.63, 0.113, 4.2),
    "OliveRidley": ("Penyu Lekang (Lepidochelys olivacea)", 81.52, 0.127, 4.2),
    "KempsRidley": ("Penyu Kemp's Ridley (Lepidochelys kempii)", 69.97, 0.198, 4.0),
    "leatherback": ("Penyu Belimbing (Dermochelys coriacea)", None, None, None),
}


def estimate_age_and_stage(species_key, length_cm):
    info = SPECIES_INFO.get(species_key)
    if not info:
        return "Spesies Tidak Dikenal", None, "-"

    nama_id, L_inf, K, L0 = info

    if L_inf is None:
        return (
            nama_id,
            "Data parameter jurnal tidak tersedia",
            "Tidak dapat diklasifikasi",
        )

    t0 = (1.0 / K) * math.log(1.0 - (L0 / L_inf))

    if length_cm >= L_inf:
        age_text = f"Dewasa (Mencapai batas asimtotik {L_inf} cm)"
    else:
        age_years = t0 - (1.0 / K) * math.log(1.0 - (length_cm / L_inf))
        age_text = f"± {max(0.1, round(age_years, 1))} tahun"

    if length_cm < 15:
        stage = "Tukik (Hatchling)"
    elif length_cm < 45:
        stage = "Juvenil (Muda)"
    elif length_cm < 75:
        stage = "Sub-Adult (Pra-Dewasa)"
    else:
        stage = "Dewasa (Adult)"

    return nama_id, age_text, stage


def detect_aruco_scale(image_bgr, marker_real_cm):
    aruco_dicts = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_4X4_100,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_5X5_100,
        cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_6X6_100,
        cv2.aruco.DICT_7X7_50,
        cv2.aruco.DICT_ARUCO_ORIGINAL,
    ]

    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10

    for dict_type in aruco_dicts:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(image_bgr)

        if ids is not None and len(corners) > 0:
            c = corners[0][0]
            side1 = np.linalg.norm(c[0] - c[1])
            side2 = np.linalg.norm(c[1] - c[2])
            side3 = np.linalg.norm(c[2] - c[3])
            side4 = np.linalg.norm(c[3] - c[0])
            avg_side_px = (side1 + side2 + side3 + side4) / 4.0

            cm_per_pixel = marker_real_cm / avg_side_px
            cv2.aruco.drawDetectedMarkers(image_bgr, corners, ids)
            return cm_per_pixel, True

    return None, False


def analyze_turtle(image_rgb, marker_size_cm):
    if image_rgb is None:
        return None, "Silakan unggah, ambil via webcam, atau paste foto penyu terlebih dahulu."

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w, _ = image_bgr.shape
    total_img_area = img_h * img_w

    cm_per_pixel, marker_found = detect_aruco_scale(image_bgr, marker_size_cm)
    results = model.predict(image_bgr, conf=0.75, verbose=False)[0]

    valid_boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_area = (x2 - x1) * (y2 - y1)
        if (box_area / total_img_area) >= 0.02:
            valid_boxes.append(box)

    annotated_bgr = image_bgr.copy()

    if len(valid_boxes) == 0:
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb, (
            "⚠️ **Objek Bukan Penyu atau Gambar Tidak Jelas.**\n\nTidak ditemukan"
            " penyu yang memenuhi standar keyakinan minimal (≥ 75%). Pastikan objek"
            " utama di dalam foto adalah penyu laut."
        )

    report_lines = []
    if marker_found:
        report_lines.append(
            f"✅ **Kalibrasi ArUco Berhasil** (Skala: {cm_per_pixel:.4f} cm/piksel)\n"
        )
    else:
        report_lines.append(
            "ℹ️ **Mode Identifikasi Spesies (Tanpa Marker ArUco)**\n*Sertakan"
            " marker ArUco di samping penyu jika ingin menghitung panjang karapas"
            " dan estimasi usia.*\n"
        )

    for i, box in enumerate(valid_boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0]) * 100

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width_px = x2 - x1
        height_px = y2 - y1

        cv2.rectangle(
            annotated_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3
        )
        cv2.putText(
            annotated_bgr,
            f"{cls_name} {conf:.1f}%",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        nama_id = SPECIES_INFO.get(cls_name, (cls_name, None, None, None))[0]
        report_lines.append(f"### Penyu #{i+1}")
        report_lines.append(f"* **Spesies:** {nama_id} *(Keyakinan: {conf:.1f}%)*")

        if marker_found:
            max_px = max(width_px, height_px)
            est_length_cm = round(max_px * cm_per_pixel, 2)
            _, est_age, stage = estimate_age_and_stage(cls_name, est_length_cm)

            report_lines.append(f"* **Estimasi Panjang Karapas:** {est_length_cm} cm")
            report_lines.append(f"* **Estimasi Usia (VBGF):** {est_age}")
            report_lines.append(f"* **Fase Hidup:** {stage}\n")
        else:
            report_lines.append("* **Estimasi Panjang Karapas:** *Memerlukan marker*")
            report_lines.append("* **Estimasi Usia (VBGF):** *Memerlukan marker*\n")

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, "\n".join(report_lines)


demo = gr.Interface(
    fn=analyze_turtle,
    inputs=[
        gr.Image(
            sources=["upload", "webcam", "clipboard"],
            type="numpy",
            label="Unggah, Ambil via Webcam, atau Paste Foto Penyu",
        ),
        gr.Number(value=10.0, label="Ukuran Fisik Sisi ArUco Marker (cm)"),
    ],
    outputs=[
        gr.Image(label="Hasil Deteksi & Kalibrasi"),
        gr.Markdown(label="Laporan Biometrik Penyu (Pulau Rinca - PKM-RE)"),
    ],
    title="Sistem Identifikasi & Morfometri Penyu Laut",
    description=(
        "Deteksi spesies penyu otomatis, serta estimasi panjang karapas dan"
        " usia pertumbuhan Von Bertalanffy (Ramirez et al., 2021) jika"
        " terkalibrasi ArUco Marker."
    ),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
    