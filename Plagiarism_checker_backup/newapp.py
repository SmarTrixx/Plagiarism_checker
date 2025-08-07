import os
import tempfile
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from pdf2image import convert_from_path
from docx import Document
import fitz  # PyMuPDF for extracting from PDFs
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from collections import defaultdict

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
model = SentenceTransformer('all-MiniLM-L6-v2')
SIMILARITY_THRESHOLD = 0.75  # Adjust this as needed

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Enhance image quality for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Apply dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    processed = cv2.erode(processed, kernel, iterations=1)
    
    return processed

def extract_text_from_image(file_path):
    try:
        image = cv2.imread(file_path)
        if image is None:
            return ""
        
        processed = preprocess_image(image)
        custom_config = r'--oem 3 --psm 6'  # OCR engine mode and page segmentation mode
        text = pytesseract.image_to_string(processed, config=custom_config)
        return text
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path):
    text = ""
    try:
        # First try to extract text directly
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        
        # If no text found, try OCR
        if not text.strip():
            images = convert_from_path(file_path)
            for image in images:
                text += pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
    return text

def extract_text(file_path):
    ext = file_path.rsplit('.', 1)[-1].lower()
    try:
        if ext == 'pdf':
            return extract_text_from_pdf(file_path)
        elif ext == 'docx':
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        elif ext in ['png', 'jpg', 'jpeg']:
            return extract_text_from_image(file_path)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    return ""

def calculate_document_similarity(text1, text2):
    """Calculate overall similarity between two documents"""
    if not text1.strip() or not text2.strip():
        return 0.0
    
    # Split into sentences or chunks
    chunks1 = [text1[i:i+500] for i in range(0, len(text1), 500)]  # Split into 500-character chunks
    chunks2 = [text2[i:i+500] for i in range(0, len(text2), 500)]
    
    # Encode chunks
    emb1 = model.encode(chunks1, convert_to_tensor=True)
    emb2 = model.encode(chunks2, convert_to_tensor=True)
    
    # Calculate similarity matrix
    sim_matrix = cosine_similarity(emb1, emb2)
    
    # Get average similarity
    if sim_matrix.size > 0:
        return sim_matrix.max(axis=1).mean()
    return 0.0

def generate_heatmap(similarity_matrix, filenames):
    """Generate a heatmap of document similarities"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=filenames, yticklabels=filenames)
    plt.title("Document Similarity Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save to a temporary buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    heatmap_data = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    return heatmap_data

def get_risk_level(score):
    """Determine risk level based on similarity score"""
    if score >= 0.8:
        return "High", "danger"
    elif score >= 0.6:
        return "Medium", "warning"
    elif score >= 0.4:
        return "Low", "info"
    else:
        return "Very Low", "success"

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)
        
        files = request.files.getlist('files')
        if len(files) < 2:
            return render_template('upload.html', error="Please upload at least 2 files for comparison")
        
        file_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                file_paths.append(file_path)
        
        if len(file_paths) < 2:
            return render_template('upload.html', error="Need at least 2 valid files for comparison")
        
        return redirect(url_for('results', files=','.join(file_paths)))
    
    return render_template('upload.html')

@app.route('/results')
def results():
    file_paths = request.args.get('files', '').split(',')
    file_paths = [fp for fp in file_paths if fp]  # Remove empty strings
    
    if len(file_paths) < 2:
        return redirect(url_for('upload'))
    
    # Extract filenames for display
    filenames = [os.path.basename(fp) for fp in file_paths]
    
    # Prepare similarity matrix
    num_files = len(file_paths)
    similarity_matrix = np.zeros((num_files, num_files))
    comparison_results = []
    
    # Compare all pairs of files
    for i in range(num_files):
        text1 = extract_text(file_paths[i])
        for j in range(i+1, num_files):
            text2 = extract_text(file_paths[j])
            similarity = calculate_document_similarity(text1, text2)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # Symmetric
            
            risk_level, risk_class = get_risk_level(similarity)
            comparison_results.append({
                'file1': filenames[i],
                'file2': filenames[j],
                'similarity': f"{similarity*100:.1f}%",
                'score': similarity,
                'risk_level': risk_level,
                'risk_class': risk_class
            })
    
    # Sort results by similarity score (descending)
    comparison_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Add serial numbers
    for i, result in enumerate(comparison_results, 1):
        result['sn'] = i
    
    # Generate heatmap
    heatmap_data = generate_heatmap(similarity_matrix, filenames)
    
    return render_template('results.html', 
                         comparisons=comparison_results,
                         heatmap=heatmap_data,
                         filenames=filenames)

if __name__ == '__main__':
    app.run(debug=True)