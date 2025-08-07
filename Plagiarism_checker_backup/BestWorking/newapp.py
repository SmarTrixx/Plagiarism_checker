import os
import tempfile
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
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
from io import BytesIO, StringIO
import base64
from collections import defaultdict
import csv

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.secret_key = 'your-secret-key-here'  # Needed for flash messages
model = SentenceTransformer('all-MiniLM-L6-v2')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Enhance image quality for better OCR results"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
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
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed, config=custom_config)
        return text
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
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
    if not text1.strip() or not text2.strip():
        return 0.0
    chunks1 = [text1[i:i+500] for i in range(0, len(text1), 500)]
    chunks2 = [text2[i:i+500] for i in range(0, len(text2), 500)]
    emb1 = model.encode(chunks1, convert_to_tensor=True)
    emb2 = model.encode(chunks2, convert_to_tensor=True)
    sim_matrix = cosine_similarity(emb1, emb2)
    return sim_matrix.max(axis=1).mean() if sim_matrix.size > 0 else 0.0

def generate_heatmap(similarity_matrix, filenames):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=filenames, yticklabels=filenames)
    plt.title("Document Similarity Heatmap")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    heatmap_data = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    return heatmap_data

def get_risk_level(score):
    if score >= 0.8:
        return "High", "danger"
    elif score >= 0.6:
        return "Medium", "warning"
    elif score >= 0.4:
        return "Low", "info"
    else:
        return "Very Low", "success"

def generate_report(comparisons, filenames):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["S/N", "File 1", "File 2", "Similarity Score", "Risk Level"])
    for i, comp in enumerate(comparisons, 1):
        writer.writerow([
            i,
            comp['file1'],
            comp['file2'],
            comp['similarity'],
            comp['risk_level']
        ])
    return output.getvalue()

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        threshold = float(request.form.get('threshold', 0.75))
        if 'files' not in request.files:
            flash('No files selected', 'danger')
            return redirect(request.url)
        
        files = request.files.getlist('files')
        if len(files) < 2:
            flash('Please upload at least 2 files for comparison', 'warning')
            return redirect(request.url)
        
        file_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                file_paths.append(file_path)
        
        if len(file_paths) < 2:
            flash('Need at least 2 valid files for comparison', 'warning')
            return redirect(request.url)
        
        return redirect(url_for('results', files=','.join(file_paths), threshold=threshold))
    
    return render_template('upload.html')

@app.route('/results')
def results():
    file_paths = request.args.get('files', '').split(',')
    file_paths = [fp for fp in file_paths if fp]
    threshold = float(request.args.get('threshold', 0.75))
    
    if len(file_paths) < 2:
        return redirect(url_for('upload'))
    
    filenames = [os.path.basename(fp) for fp in file_paths]
    num_files = len(file_paths)
    similarity_matrix = np.zeros((num_files, num_files))
    comparison_results = []
    text_contents = {}
    
    for i in range(num_files):
        text_contents[filenames[i]] = extract_text(file_paths[i])
    
    for i in range(num_files):
        for j in range(i+1, num_files):
            similarity = calculate_document_similarity(
                text_contents[filenames[i]],
                text_contents[filenames[j]]
            )
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
            
            if similarity >= threshold:
                risk_level, risk_class = get_risk_level(similarity)
                comparison_results.append({
                    'file1': filenames[i],
                    'file2': filenames[j],
                    'similarity': f"{similarity*100:.1f}%",
                    'score': similarity,
                    'risk_level': risk_level,
                    'risk_class': risk_class,
                    'text1': text_contents[filenames[i]],
                    'text2': text_contents[filenames[j]]
                })
    
    comparison_results.sort(key=lambda x: x['score'], reverse=True)
    for i, result in enumerate(comparison_results, 1):
        result['sn'] = i
    
    heatmap_data = generate_heatmap(similarity_matrix, filenames)
    
    return render_template('results.html', 
                         comparisons=comparison_results,
                         heatmap=heatmap_data,
                         filenames=filenames,
                         threshold=threshold)

@app.route('/download-report')
def download_report():
    file_paths = request.args.get('files', '').split(',')
    file_paths = [fp for fp in file_paths if fp]
    threshold = float(request.args.get('threshold', 0.75))
    
    filenames = [os.path.basename(fp) for fp in file_paths]
    comparison_results = []
    
    for i in range(len(file_paths)):
        for j in range(i+1, len(file_paths)):
            text1 = extract_text(file_paths[i])
            text2 = extract_text(file_paths[j])
            similarity = calculate_document_similarity(text1, text2)
            if similarity >= threshold:
                risk_level, _ = get_risk_level(similarity)
                comparison_results.append({
                    'file1': filenames[i],
                    'file2': filenames[j],
                    'similarity': f"{similarity*100:.1f}%",
                    'risk_level': risk_level
                })
    
    report = generate_report(comparison_results, filenames)
    mem = BytesIO()
    mem.write(report.encode('utf-8'))
    mem.seek(0)
    return send_file(
        mem,
        as_attachment=True,
        download_name='similarity_report.csv',
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(debug=True)