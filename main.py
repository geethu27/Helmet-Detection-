import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create app FIRST
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import glob
import traceback

# Attempt to locate and load a trained YOLO `best.pt`. If not found, defer loading
def _find_best_weight():
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "best.pt"),
        os.path.join(here, "weights", "best.pt"),
        os.path.join(here, "..", "helmet_dataset", "runs", "detect", "train8", "weights", "best.pt"),
    ]
    base = os.path.abspath(os.path.join(here, "..", "helmet_dataset"))
    pattern = os.path.join(base, "runs", "detect", "*", "weights", "best.pt")
    found = glob.glob(pattern)
    if found:
        found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return found[0]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None

# lazy-load model; set to None if not available at import time
helmet_model = None
try:
    weight = _find_best_weight()
    if weight:
        helmet_model = YOLO(weight)
    else:
        helmet_model = None
        print("Warning: no YOLO weights found at import; `/detect/` will attempt to load on first request.")
except Exception:
    helmet_model = None
    print("Warning: failed to load YOLO model at import:")
    traceback.print_exc()

# Load trained helmet model
@app.post("/detect-video/")
async def detect_video(file: UploadFile = File(...)):
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(temp_path)
    violations = []
    plate_images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = helmet_model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 1 and conf > 0.6:  # No Helmet
                violations.append("No Helmet Detected")
                # save plate screenshot logic here

    cap.release()
    return {"result": "Video Processed", "violations": violations}



def read_image(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


@app.get("/")
def home():
    return {"message": "Helmet Detection API Running"}


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        image = read_image(file.file)
        if image is None:
            return {"error": "Invalid image"}


        # ensure model is loaded (lazy load if import-time failed)
        global helmet_model
        if helmet_model is None:
            try:
                weight = _find_best_weight()
                if weight:
                    helmet_model = YOLO(weight)
                else:
                    return {"error": "no model weights found"}
            except Exception as e:
                return {"error": f"failed to load model: {str(e)}"}

        # run inference with model and keep raw results
        results = helmet_model(image)[0]

        # parameters
        CONF_THRESHOLD = 0.5
        IOU_RESOLVE = 0.3

        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "cls": cls,
                "conf": conf,
                "xy": (x1, y1, x2, y2)
            })

        # resolve overlaps: if helmet and no_helmet overlap, prefer higher confidence
        def iou(a, b):
            xA = max(a[0], b[0])
            yA = max(a[1], b[1])
            xB = min(a[2], b[2])
            yB = min(a[3], b[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            inter = interW * interH
            areaA = (a[2] - a[0]) * (a[3] - a[1])
            areaB = (b[2] - b[0]) * (b[3] - b[1])
            union = areaA + areaB - inter
            return inter / union if union > 0 else 0

        final_dets = []
        used = [False] * len(detections)
        for i, d in enumerate(detections):
            if used[i]:
                continue
            # compare with other detections
            keep = d
            for j, o in enumerate(detections):
                if i == j or used[j]:
                    continue
                if (d["cls"] in (0, 1)) and (o["cls"] in (0, 1)):
                    if iou(d["xy"], o["xy"]) > IOU_RESOLVE:
                        # prefer higher confidence
                        if o["conf"] > keep["conf"]:
                            keep = o
                            used[j] = True
                        else:
                            used[j] = True
            final_dets.append(keep)

        violation = False
        plate_text = ""
        plate_crop = None

        # helper to find plate-like region inside a box
        def find_plate(candidate_img):
            gray = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(gray, 30, 200)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            plate_box = None
            max_area = 0
            h, w = candidate_img.shape[:2]
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                if area < 500:
                    continue
                ratio = cw / float(ch) if ch > 0 else 0
                if 2.0 <= ratio <= 6.0 and area > max_area:
                    max_area = area
                    plate_box = (x, y, x + cw, y + ch)
            if plate_box:
                x1, y1, x2, y2 = plate_box
                return candidate_img[y1:y2, x1:x2]
            return None

        for d in final_dets:
            cls = d["cls"]
            x1, y1, x2, y2 = d["xy"]
            conf = d["conf"]

            if cls == 0:
                color = (0, 255, 0)
                label = f"Helmet {conf:.2f}"
                # green mark
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif cls == 1:
                color = (0, 0, 255)
                label = f"No Helmet {conf:.2f}"
                violation = True
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # attempt to find plate within lower part of the detected box
                bx_h = y2 - y1
                search_y1 = y1 + int(bx_h * 0.4)
                search_y2 = y2
                search_x1 = x1
                search_x2 = x2
                search_y2 = min(search_y2, image.shape[0])
                search_region = image[search_y1:search_y2, search_x1:search_x2]
                if search_region.size != 0:
                    candidate = find_plate(search_region)
                    if candidate is not None:
                        plate_crop = candidate
                        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                        text = pytesseract.image_to_string(gray, config="--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                        plate_text = ''.join(e for e in text if e.isalnum()).upper()
                        # draw plate box and fine text
                        ph, pw = plate_crop.shape[:2]
                        # approximate plate location on original image
                        # we only have relative pos inside search_region
                        # find contours again to get exact location
                        # fallback: just draw the search region as plate location
                        cv2.rectangle(image, (search_x1, search_y1), (search_x2, search_y2), (0, 0, 255), 2)
                        fine_label = f"FINE: {plate_text if plate_text else 'N/A'}"
                        cv2.putText(image, fine_label, (search_x1, search_y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                color = (255, 0, 0)
                label = f"Class {cls} {conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # prepare boxes metadata for frontend overlays
        boxes_out = []
        for d in final_dets:
            cls = int(d["cls"]) if isinstance(d.get("cls"), (int, float)) else int(d["cls"]) if "cls" in d else None
            x1, y1, x2, y2 = d["xy"]
            conf = float(d["conf"]) if "conf" in d else 0.0
            boxes_out.append({
                "cls": cls,
                "conf": conf,
                "xy": [int(x1), int(y1), int(x2), int(y2)]
            })

        _, buffer = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "violation": violation,
            "plate": plate_text,
            "image": img_base64,
            "boxes": boxes_out
        }

    except Exception as e:
        return {"error": str(e)}
