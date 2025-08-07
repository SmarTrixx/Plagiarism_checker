# utils.py
import os
import shutil
import difflib
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

UPLOAD_FOLDER = "uploads"
ARCHIVE_FOLDER = os.path.join(UPLOAD_FOLDER, "archive")
TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, "temp")

def ensure_dirs():
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join([page.extract_text() or '' for page in reader.pages])
    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

def move_to_archive():
    for filename in os.listdir(TEMP_FOLDER):
        src = os.path.join(TEMP_FOLDER, filename)
        dst = os.path.join(ARCHIVE_FOLDER, filename)
        shutil.move(src, dst)

def get_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(('.txt', '.pdf', '.docx'))]

def compare_texts(file1, file2):
    text1 = extract_text(file1)
    text2 = extract_text(file2)

    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    return round(similarity * 100, 2)

def batch_compare(temp_files, archive_files, threshold=30.0):
    results = []
    seen_pairs = set()

    temp_basenames = {os.path.basename(f): f for f in temp_files}
    archive_basenames = {os.path.basename(f): f for f in archive_files}

    all_files = {**temp_basenames, **archive_basenames}

    # Compare all unique pairs
    file_list = list(all_files.keys())
    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            f1_name, f2_name = file_list[i], file_list[j]
            pair_key = tuple(sorted([f1_name, f2_name]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            f1_path = all_files[f1_name]
            f2_path = all_files[f2_name]

            sim = compare_texts(f1_path, f2_path)
            if sim >= threshold:
                # Ensure uploaded file is always on the left
                if f1_name in temp_basenames:
                    results.append((f1_name, f2_name, sim))
                elif f2_name in temp_basenames:
                    results.append((f2_name, f1_name, sim))

    # Sort by similarity descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results

    results = []
    seen_pairs = set()

    # Compare new uploads with themselves
    for i in range(len(temp_files)):
        for j in range(i + 1, len(temp_files)):
            f1, f2 = temp_files[i], temp_files[j]
            pair_key = tuple(sorted([os.path.basename(f1), os.path.basename(f2)]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            sim = compare_texts(f1, f2)
            if sim >= threshold:
                results.append((pair_key[0], pair_key[1], sim))

    # Compare new uploads with archive
    for f1 in temp_files:
        for f2 in archive_files:
            if os.path.basename(f1) == os.path.basename(f2):
                continue

            pair_key = tuple(sorted([os.path.basename(f1), os.path.basename(f2)]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            sim = compare_texts(f1, f2)
            if sim >= threshold:
                results.append((pair_key[0], pair_key[1], sim))

    # Sort results by similarity score descending
    results.sort(key=lambda x: x[2], reverse=True)

    return results

    results = []
    seen_pairs = set()

    # Compare new uploads with themselves
    for i in range(len(temp_files)):
        for j in range(i + 1, len(temp_files)):
            f1, f2 = temp_files[i], temp_files[j]
            pair_key = tuple(sorted([os.path.basename(f1), os.path.basename(f2)]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            sim = compare_texts(f1, f2)
            if sim >= threshold:
                results.append((pair_key[0], pair_key[1], sim))

    # Compare new uploads with archive
    for f1 in temp_files:
        for f2 in archive_files:
            if os.path.basename(f1) == os.path.basename(f2):
                continue

            pair_key = tuple(sorted([os.path.basename(f1), os.path.basename(f2)]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            sim = compare_texts(f1, f2)
            if sim >= threshold:
                results.append((pair_key[0], pair_key[1], sim))

    return results

    results = []

    # Compare new uploads with themselves
    for i in range(len(temp_files)):
        for j in range(i + 1, len(temp_files)):
            f1, f2 = temp_files[i], temp_files[j]
            sim = compare_texts(f1, f2)
            if sim >= threshold:
                results.append((os.path.basename(f1), os.path.basename(f2), sim))

    # Compare new uploads with archive
    for f1 in temp_files:
        for f2 in archive_files:
            if os.path.basename(f1) == os.path.basename(f2):  # skip if same file
                continue
            sim = compare_texts(f1, f2)
            if sim >= threshold:
                results.append((os.path.basename(f1), os.path.basename(f2), sim))

    return results

def generate_heatmap(results, output_path="static/heatmap.png"):
    if not results:
        return None

    # Only include files that actually participated in at least one result
    involved_files = set()
    for f1, f2, _ in results:
        involved_files.add(f1)
        involved_files.add(f2)
    involved_files = sorted(involved_files)

    # Create a filtered similarity matrix (empty)
    matrix = pd.DataFrame(index=involved_files, columns=involved_files, dtype=float)

    # Fill only actual pairs from filtered results (no symmetry unless present in results)
    for f1, f2, sim in results:
        matrix.at[f1, f2] = sim

    # Drop any rows/cols with all NaN (shouldn't happen now, but just in case)
    matrix.dropna(axis=0, how='all', inplace=True)
    matrix.dropna(axis=1, how='all', inplace=True)

    # Plot only this trimmed matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Reds", square=True, cbar=True, mask=matrix.isnull())
    plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels to prevent overlap
    plt.title("Filtered Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


    return output_path

    if not results:
        return None

    # Collect all unique files from the filtered result list
    unique_files = sorted(set([r[0] for r in results] + [r[1] for r in results]))

    # Create an empty similarity matrix (only for the files that appear in results)
    matrix = pd.DataFrame(index=unique_files, columns=unique_files, dtype=float)

    # Fill only the entries shown in results (no symmetry)
    for f1, f2, sim in results:
        matrix.at[f1, f2] = sim  # only fill (f1, f2), not (f2, f1)

    # Plot the heatmap (non-symmetric)
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Reds", square=True, cbar=True)
    plt.title("Similarity Heatmap (Filtered)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path

    if not results:
        return None

    # Only include files that appear in the result table
    files_left = set(r[0] for r in results)
    files_right = set(r[1] for r in results)
    all_files = sorted(files_left.union(files_right))

    # Initialize an empty square matrix
    matrix = pd.DataFrame(0, index=all_files, columns=all_files, dtype=float)

    for f1, f2, sim in results:
        matrix.at[f1, f2] = sim
        matrix.at[f2, f1] = sim  # Optional: make it symmetric visually

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Reds", square=True, cbar=True)
    plt.title("Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Free up memory
    return output_path

    if not results:
        return None

    # Only include files that appear in the result table
    files_left = set(r[0] for r in results)
    files_right = set(r[1] for r in results)
    all_files = sorted(files_left.union(files_right))

    # Initialize an empty square matrix
    matrix = pd.DataFrame(0, index=all_files, columns=all_files, dtype=float)

    for f1, f2, sim in results:
        matrix.at[f1, f2] = sim
        matrix.at[f2, f1] = sim  # Optional: make it symmetric visually

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Reds", square=True, cbar=True)
    plt.title("Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Free up memory
    return output_path

    if not results:
        return None

    files = list(set([item for pair in results for item in pair[:2]]))
    matrix = pd.DataFrame(0, index=files, columns=files, dtype=float)

    for f1, f2, sim in results:
        matrix.at[f1, f2] = sim
        matrix.at[f2, f1] = sim  # symmetric

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="Reds", square=True)
    plt.title("Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    return output_path
