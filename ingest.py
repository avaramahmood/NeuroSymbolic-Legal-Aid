import os
import weaviate
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import get_client, COLLECTION_NAME, TAG_CASE, EMBED_MODEL_NAME

# --- SETTINGS ---
DATA_DIR = "data"
CHUNK_SIZE = 500
OVERLAP = 50
START_YEAR = 2000
END_YEAR = 2025

def load_embedding_model():
    print(f"\nLoading Local Embedding Model ({EMBED_MODEL_NAME})...")
    return SentenceTransformer(EMBED_MODEL_NAME)

def read_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
        return text
    except Exception as e:
        return None

def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def ingest_batch(client, model, file_path, category, year):
    text = read_pdf(file_path)
    if not text: return

    chunks = chunk_text(text)
    if not chunks: return

    filename = os.path.basename(file_path)
    vectors = model.encode(chunks)
    collection = client.collections.get(COLLECTION_NAME)
    
    # Batch upload
    try:
        with collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                batch.add_object(
                    properties={
                        "content": chunk,
                        "source": filename,
                        "year": year,
                        "category": category
                    },
                    vector=vectors[i]
                )
    except Exception as e:
        print(f"Upload Error: {e}")

def verify_db_count(client):
    """Checks the database to see if data actually arrived."""
    try:
        collection = client.collections.get(COLLECTION_NAME)
        count = collection.aggregate.over_all(total_count=True).total_count
        print(f"\n📊 VERIFICATION: Database now contains {count} chunks of text.")
    except Exception as e:
        print(f"\n⚠️ Could not verify count (this is minor): {e}")

def run_ingestion():
    # 1. Locate the folder
    # We look for 'supreme_court_judgments' OR 'supreme_court_judgements'
    target_dir = None
    for name in ["supreme_court_judgments", "supreme_court_judgements"]:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            target_dir = path
            break
    
    if not target_dir:
        print(f"Error: Could not find 'supreme_court_judgments' folder in '{DATA_DIR}'")
        return

    print(f"Found Data Folder: {target_dir}")
    model = load_embedding_model()
    client = get_client()

    print(f"   Scanning years {START_YEAR} to {END_YEAR}...")
    
    files_found_total = 0

    # 2. Walk through folders
    for root, dirs, files in os.walk(target_dir):
        folder_name = os.path.basename(root)
        
        # Check if the folder name is a number (e.g. "2000")
        if folder_name.isdigit():
            year = int(folder_name)
            
            if START_YEAR <= year <= END_YEAR:
                # Find PDFs (Case Insensitive! .pdf, .PDF, .Pdf)
                pdf_files = [f for f in files if f.lower().endswith(".pdf")]
                
                if pdf_files:
                    print(f"   Year {year}: Found {len(pdf_files)} files. Processing...")
                    files_found_total += len(pdf_files)
                    
                    for f in tqdm(pdf_files, leave=False, desc=f"Year {year}"):
                        ingest_batch(client, model, os.path.join(root, f), TAG_CASE, year)
                else:
                    # Optional: warning if folder is empty
                    pass 

    if files_found_total == 0:
        print("\n CRITICAL: No PDF files were found!")
        print("   Please check: Are the files inside the year folders? Do they end in .pdf?")
    else:
        print(f"\n Processing finished for {files_found_total} files.")
        verify_db_count(client)

    client.close()

if __name__ == "__main__":
    run_ingestion()