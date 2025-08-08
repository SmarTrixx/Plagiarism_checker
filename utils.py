import cv2
import numpy as np
import pytesseract
from docx import Document
import fitz
from pdf2image import convert_from_path

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    processed = cv2.erode(processed, kernel, iterations=1)
    return processed

def extract_text_from_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        return ""
    processed = preprocess_image(image)
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(processed, config=custom_config)

def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    if not text.strip():
        images = convert_from_path(file_path)
        for image in images:
            text += pytesseract.image_to_string(image)
    return text

def extract_text(file_path):
    ext = file_path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'docx':
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif ext in ['png', 'jpg', 'jpeg']:
        return extract_text_from_image(file_path)
    return ""