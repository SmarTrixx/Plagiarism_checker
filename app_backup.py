# import os
# import tempfile
# from flask import Flask, request, render_template, redirect, url_for, flash, send_file, session
# from werkzeug.utils import secure_filename
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import pytesseract
# from pdf2image import convert_from_path
# from docx import Document
# import fitz  # PyMuPDF for extracting from PDFs
# import cv2
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from io import BytesIO, StringIO
# import base64
# from collections import defaultdict
# import csv
# import json
# from utils import allowed_file, preprocess_image, extract_text, extract_text_from_image, extract_text_from_pdf

# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'pdf', 'docx', 'png', 'jpg', 'jpeg'}

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
# app.secret_key = 'your-secret-key-here'  # Needed for flash messages
# model = SentenceTransformer('all-MiniLM-L6-v2')

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# def calculate_document_similarity(text1, text2):
#     if not text1.strip() or not text2.strip():
#         return 0.0
#     chunks1 = [text1[i:i+500] for i in range(0, len(text1), 500)]
#     chunks2 = [text2[i:i+500] for i in range(0, len(text2), 500)]
#     emb1 = model.encode(chunks1, convert_to_tensor=True)
#     emb2 = model.encode(chunks2, convert_to_tensor=True)
#     sim_matrix = cosine_similarity(emb1, emb2)
#     return sim_matrix.max(axis=1).mean() if sim_matrix.size > 0 else 0.0

# def generate_heatmap(similarity_matrix, filenames):
#     """Generate a heatmap of document similarities"""
#     try:
#         plt.switch_backend('Agg')  # Set the backend to Agg before creating figure
#         plt.figure(figsize=(10, 8))
        
#         # Create a mask for the upper triangle to avoid duplicate values
#         mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
        
#         # Create heatmap with masked upper triangle
#         sns.heatmap(similarity_matrix, mask=mask, annot=True, fmt=".2f", cmap="YlOrRd",
#                     xticklabels=filenames, yticklabels=filenames, vmin=0, vmax=1)
        
#         plt.title("Document Similarity Heatmap")
#         plt.xticks(rotation=45, ha='right')
#         plt.yticks(rotation=0)
#         plt.tight_layout()
        
#         # Save to a temporary buffer
#         buf = BytesIO()
#         plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
#         buf.seek(0)
#         heatmap_data = base64.b64encode(buf.read()).decode('ascii')
#         plt.close()  # Explicitly close the figure to free memory
#         return heatmap_data
#     except Exception as e:
#         print(f"Error generating heatmap: {e}")
#         return None

# def get_risk_level(score):
#     if score >= 0.8:
#         return "High", "danger"
#     elif score >= 0.6:
#         return "Medium", "warning"
#     elif score >= 0.4:
#         return "Low", "info"
#     else:
#         return "Very Low", "success"

# def generate_report(comparisons, filenames):
#     output = StringIO()
#     writer = csv.writer(output)
#     writer.writerow(["S/N", "File 1", "File 2", "Similarity Score", "Risk Level"])
#     for i, comp in enumerate(comparisons, 1):
#         writer.writerow([
#             i,
#             comp['file1'],
#             comp['file2'],
#             comp['similarity'],
#             comp['risk_level']
#         ])
#     return output.getvalue()

# @app.route('/', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         threshold = float(request.form.get('threshold', 0.75))
#         if 'files' not in request.files:
#             flash('No files selected', 'danger')
#             return redirect(request.url)
        
#         files = request.files.getlist('files')
#         if len(files) < 2:
#             flash('Please upload at least 2 files for comparison', 'warning')
#             return redirect(request.url)
        
#         file_paths = []
#         for file in files:
#             if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
#                 filename = secure_filename(file.filename)
#                 file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(file_path)
#                 file_paths.append(file_path)
        
#         if len(file_paths) < 2:
#             flash('Need at least 2 valid files for comparison', 'warning')
#             return redirect(request.url)
        
#         return redirect(url_for('results', files=','.join(file_paths), threshold=threshold))
    
#     return render_template('upload.html')

# @app.route('/results')
# def results():
#     file_paths = request.args.get('files', '').split(',')
#     file_paths = [fp for fp in file_paths if fp]
#     threshold = float(request.args.get('threshold', 0.75))
    
#     if len(file_paths) < 2:
#         return redirect(url_for('upload'))
    
#     filenames = [os.path.basename(fp) for fp in file_paths]
#     num_files = len(file_paths)
#     similarity_matrix = np.zeros((num_files, num_files))
#     comparison_results = []
#     text_contents = {}
    
#     for i in range(num_files):
#         text_contents[filenames[i]] = extract_text(file_paths[i])
    
#     for i in range(num_files):
#         for j in range(i+1, num_files):
#             similarity = calculate_document_similarity(
#                 text_contents[filenames[i]],
#                 text_contents[filenames[j]]
#             )
#             similarity_matrix[i][j] = similarity
#             similarity_matrix[j][i] = similarity
            
#             if similarity >= threshold:
#                 risk_level, risk_class = get_risk_level(float(similarity))  # ensure Python float
#                 comparison_results.append({
#                     'file1': filenames[i],
#                     'file2': filenames[j],
#                     'similarity': f"{float(similarity)*100:.1f}%",  # ensure Python float
#                     'score': float(similarity),                     # ensure Python float
#                     'risk_level': risk_level,
#                     'risk_class': risk_class,
#                     'text1': text_contents[filenames[i]],
#                     'text2': text_contents[filenames[j]]
#                 })
    
#     comparison_results.sort(key=lambda x: x['score'], reverse=True)
#     for i, result in enumerate(comparison_results, 1):
#         result['sn'] = i
    
#     heatmap_data = generate_heatmap(similarity_matrix, filenames)
    
#     try:
#         heatmap_data = generate_heatmap(similarity_matrix, filenames)
#     except Exception as e:
#         print(f"Failed to generate heatmap: {e}")
#         heatmap_data = None
    
#     # Store results in session
#     session['comparison_results'] = json.dumps(comparison_results)
#     session['filenames'] = json.dumps(filenames)
    
#     return render_template('results.html', 
#                          comparisons=comparison_results,
#                          heatmap=heatmap_data,
#                          filenames=filenames,
#                          threshold=threshold)

# from flask import session
# import json

# @app.route('/download-report')
# def download_report():
#     comparison_results = json.loads(session.get('comparison_results', '[]'))
#     filenames = json.loads(session.get('filenames', '[]'))
#     report = generate_report(comparison_results, filenames)
#     mem = BytesIO()
#     mem.write(report.encode('utf-8'))
#     mem.seek(0)
#     return send_file(
#         mem,
#         as_attachment=True,
#         download_name='similarity_report.csv',
#         mimetype='text/csv'
#     )

# if __name__ == '__main__':
#     app.run(debug=True)