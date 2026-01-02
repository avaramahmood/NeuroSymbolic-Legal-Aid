import os
import json
import requests
from pypdf import PdfReader
from tqdm import tqdm
from config import GEN_MODEL

# --- SETTINGS ---
DATA_DIR = "data"
LAWS_FOLDER = "latest_law"
OUTPUT_FOLDER = "extracted_rules"  # Where we save the JSONs
OLLAMA_URL = "http://localhost:11434/api/generate"

def ensure_output_dir():
    """Creates the output folder if it doesn't exist."""
    path = os.path.join(DATA_DIR, OUTPUT_FOLDER)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def extract_structured_rules(text_chunk, filename):
    """
    Sends text to DeepSeek and gets back a specific JSON structure.
    """
    prompt = f"""
    You are a Legal Compiler.
    Read the text from '{filename}' and convert the legal rules into a STRICT JSON dataset.
    
    Output a JSON LIST of objects. Each object must have this schema:
    {{
      "section": "The section number (e.g., 'Section 45')",
      "title": "The name of the crime/rule",
      "type": "Definition" or "Penalty" or "Procedure",
      "prerequisites": ["Condition 1", "Condition 2"],
      "consequence": "The punishment or legal result",
      "exceptions": ["Exception 1"]
    }}

    Return ONLY the JSON list. If no rules are found, return [].
    
    TEXT:
    {text_chunk}
    """
    
    payload = {
        "model": GEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            json_str = json.loads(response.text)["response"]
            return json.loads(json_str)
    except Exception as e:
        # print(f"Skipping chunk due to error: {e}")
        pass
    return []

def run_rule_extraction():
    # 1. Setup
    rules_dir = os.path.join(DATA_DIR, LAWS_FOLDER)
    output_dir = ensure_output_dir()
    
    if not os.path.exists(rules_dir):
        print(f"Error: Source folder '{rules_dir}' not found.")
        return

    files = [f for f in os.listdir(rules_dir) if f.lower().endswith(".pdf")]
    
    print(f"Found {len(files)} PDFs. Extracting Rules to JSON...")

    # 2. Process Each PDF
    for filename in files:
        file_path = os.path.join(rules_dir, filename)
        pdf_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{pdf_name}.json")
        
        reader = PdfReader(file_path)
        all_rules = []

        print(f"\nProcessing '{filename}' ({len(reader.pages)} pages)...")
        
        # 3. Extract Page by Page
        for page in tqdm(reader.pages, desc="   Extracting Logic"):
            text = page.extract_text()
            if not text or len(text) < 100: continue

            # AI Logic Extraction
            rules = extract_structured_rules(text, filename)
            if rules:
                all_rules.extend(rules)

        # 4. Save to Local JSON
        if all_rules:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_rules, f, indent=2, ensure_ascii=False)
            print(f"   Saved {len(all_rules)} rules to '{output_file}'")
        else:
            print(f"   No rules found in {filename}.")

    print(f"\nExtraction Complete! Check the '{DATA_DIR}/{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    run_rule_extraction()