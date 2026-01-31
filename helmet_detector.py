from ultralytics import YOLO
from pathlib import Path

model_path = Path(__file__).parent / "best.pt"
model = YOLO(str(model_path))


def detect_helmet_from_image(image):
    results = model(image)[0]

    helmet_found = False
    no_helmet_found = False

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Ignore weak detections
        if conf < 0.5:
            continue

        # 0 = helmet
        if cls == 0:
            helmet_found = True

        # 1 = no_helmet
        elif cls == 1:
            no_helmet_found = True

    # Rule: No-helmet overrides helmet
    if no_helmet_found:
        return False, results

    if helmet_found:
        return True, results

    # Nothing detected
    return False, results
