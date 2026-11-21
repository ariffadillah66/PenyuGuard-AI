import base64
import math
import os
import cv2
import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# Muat model hasil training
model = YOLO("best.pt")

# Parameter Von Bertalanffy (Ramirez et al., 2021 - Tabel 3 Model D)
SPECIES_INFO = {
    "green": ("Penyu Hijau", "Chelonia mydas", 110.27, 0.051, 5.0),
    "loggerhead": ("Penyu Tempayan", "Caretta caretta", 100.06, 0.066, 4.5),
    "hawksbill": ("Penyu Sisik", "Eretmochelys imbricata", 84.63, 0.113, 4.2),
    "OliveRidley": ("Penyu Lekang", "Lepidochelys olivacea", 81.52, 0.127, 4.2),
    "KempsRidley": ("Penyu Kemp's Ridley", "Lepidochelys kempii", 69.97, 0.198, 4.0),
    "leatherback": ("Penyu Belimbing", "Dermochelys coriacea", None, None, None),
}


def estimate_age_and_stage(species_key, length_cm):
    info = SPECIES_INFO.get(species_key)
    if not info:
        return "Spesies Tidak Dikenal", None, None, "-"

    common_name, sci_name, L_inf, K, L0 = info
    if L_inf is None:
        return common_name, None, "Data parameter jurnal tidak tersedia", "Tidak dapat diklasifikasi"

    length_cm = float(length_cm)
    t0 = (1.0 / K) * math.log(1.0 - (L0 / L_inf))

    if length_cm >= L_inf:
        age_years = None
        age_text = f"Dewasa (Mencapai batas asimtotik {L_inf} cm)"
    else:
        age_years = float(max(0.1, round(t0 - (1.0 / K) * math.log(1.0 - (length_cm / L_inf)), 1)))
        age_text = f"± {age_years} tahun"

    if length_cm < 15:
        stage = "Tukik (Hatchling)"
    elif length_cm < 45:
        stage = "Juvenil (Muda)"
    elif length_cm < 75:
        stage = "Sub-Adult (Pra-Dewasa)"
    else:
        stage = "Dewasa (Adult)"

    return common_name, age_years, age_text, stage


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
            side1 = float(np.linalg.norm(c[0] - c[1]))
            side2 = float(np.linalg.norm(c[1] - c[2]))
            side3 = float(np.linalg.norm(c[2] - c[3]))
            side4 = float(np.linalg.norm(c[3] - c[0]))
            avg_side_px = (side1 + side2 + side3 + side4) / 4.0

            cm_per_pixel = float(marker_real_cm) / float(avg_side_px)
            cv2.aruco.drawDetectedMarkers(image_bgr, corners, ids)
            return float(cm_per_pixel), True

    return None, False


def run_inference_bgr(image_bgr, marker_size_cm=10.0):
    img_h, img_w, _ = image_bgr.shape
    total_img_area = img_h * img_w
    marker_size_cm = float(marker_size_cm) if marker_size_cm and marker_size_cm > 0 else 10.0

    annotated_bgr = image_bgr.copy()
    cm_per_pixel, marker_found = detect_aruco_scale(annotated_bgr, marker_size_cm)
    results = model.predict(image_bgr, conf=0.75, verbose=False)[0]

    valid_boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_area = (x2 - x1) * (y2 - y1)
        if (box_area / total_img_area) >= 0.02:
            valid_boxes.append(box)

    turtles = []

    for box in valid_boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width_px = float(x2 - x1)
        height_px = float(y2 - y1)

        cv2.rectangle(annotated_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (16, 185, 129), 3)
        cv2.putText(
            annotated_bgr,
            f"{cls_name} {conf * 100:.1f}%",
            (int(x1), max(24, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (16, 185, 129),
            2,
        )

        info = SPECIES_INFO.get(cls_name, (cls_name, cls_name, None, None, None))
        common_name, sci_name = info[0], info[1]
        est_length_cm = None
        age_years = None
        age_text = "Memerlukan marker ArUco"
        stage = "Memerlukan marker ArUco"

        if marker_found and cm_per_pixel is not None:
            max_px = max(width_px, height_px)
            est_length_cm = float(round(max_px * float(cm_per_pixel), 2))
            _, age_years, age_text, stage = estimate_age_and_stage(cls_name, est_length_cm)

        turtles.append({
            "species_key": cls_name,
            "common_name": common_name,
            "scientific_name": sci_name,
            "nama_id": f"{common_name} ({sci_name})",
            "confidence": round(float(conf), 4),
            "est_length_cm": est_length_cm,
            "age_years": age_years,
            "est_age": age_text,
            "stage": stage,
        })

    return annotated_bgr, marker_found, cm_per_pixel, turtles

def analyze_turtle_gradio(image_rgb, marker_size_cm):
    if image_rgb is None:
        return None, "Silakan unggah, ambil via webcam, atau paste foto penyu terlebih dahulu."

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    annotated_bgr, marker_found, cm_per_pixel, turtles = run_inference_bgr(image_bgr, marker_size_cm)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if len(turtles) == 0:
        return annotated_rgb, (
            "⚠️ **Objek Bukan Penyu atau Gambar Tidak Jelas.**\n\n"
            "Tidak ditemukan penyu dengan keyakinan minimal (≥ 75%)."
        )

    lines = []
    if marker_found:
        lines.append(f"✅ **Kalibrasi ArUco Berhasil** (Skala: {cm_per_pixel:.4f} cm/piksel)\n")
    else:
        lines.append("ℹ️ **Mode Identifikasi Spesies (Tanpa Marker ArUco)**\n")

    for i, t in enumerate(turtles):
        lines.append(f"### Penyu #{i+1}")
        lines.append(f"* **Spesies:** {t['nama_id']} *(Keyakinan: {t['confidence']*100:.1f}%)*")
        lines.append(f"* **Estimasi Panjang Karapas:** {t['est_length_cm']} cm" if marker_found else "* **Estimasi Panjang Karapas:** *Memerlukan marker*")
        lines.append(f"* **Estimasi Usia (VBGF):** {t['est_age']}")
        lines.append(f"* **Fase Hidup:** {t['stage']}\n")

    return annotated_rgb, "\n".join(lines)


# Antarmuka Gradio (tetap tersedia di root URL)
demo = gr.Interface(
    fn=analyze_turtle_gradio,
    inputs=[
        gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy", label="Foto Penyu"),
        gr.Number(value=10.0, label="Ukuran Fisik Sisi ArUco Marker (cm)"),
    ],
    outputs=[
        gr.Image(label="Hasil Deteksi & Kalibrasi"),
        gr.Markdown(label="Laporan Biometrik Penyu"),
    ],
    title="Sistem Identifikasi & Morfometri Penyu Laut",
)

# FastAPI + CORS agar bisa diakses langsung oleh website PenyuGuard
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    image: str
    marker_size_cm: float = 10.0


@app.get("/api/health")
def health_check():
    return {"status": "online", "model": "YOLOv8n + ArUco + VBGF"}


@app.post("/api/analyze")
def analyze_api(req: AnalyzeRequest):
    img_data = req.image.split(",", 1)[1] if "," in req.image else req.image
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return {"status": "error", "message": "Gambar tidak dapat dibaca."}

    annotated_bgr, marker_found, cm_per_pixel, turtles = run_inference_bgr(image_bgr, req.marker_size_cm)
    _, buffer = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return {
        "status": "ok",
        "detected": len(turtles) > 0,
        "marker_found": marker_found,
        "cm_per_pixel": round(cm_per_pixel, 4) if cm_per_pixel else None,
        "annotated_image": annotated_b64,
        "turtles": turtles,
    }


app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
