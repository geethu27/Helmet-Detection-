import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# 🔹 Plate format checker
def validate_plate(plate):
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'
    return bool(re.match(pattern, plate))


# 🔹 Plate detection + OCR
def detect_plate_number(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None, False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plate = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h

        if 2 < ratio < 5:
            plate = img[y:y+h, x:x+w]
            break

    if plate is None:
        return None, False

    text = pytesseract.image_to_string(plate, config='--psm 8')
    text = re.sub(r'[^A-Z0-9]', '', text)

    valid = validate_plate(text)
    return text, valid
