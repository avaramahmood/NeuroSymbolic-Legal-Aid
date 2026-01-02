import os
import json
import time
import re
import numpy as np
import faiss
import weaviate
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION (Direct Paths) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RULES_DIR = os.path.join(DATA_DIR, "extracted_rules")
CACHE_FILE = os.path.join(DATA_DIR, "rule_vectors.npy")

# Import settings from your config.py
from config import (
    COLLECTION_NAME, 
    EMBED_MODEL_NAME, 
    GEN_MODEL, 
    OLLAMA_BASE_URL,
    get_client
)


class RuleSearchEngine:
    """
    Responsible for finding exact Acts and Sections (Statutes).
    Uses FAISS for fast semantic search.
    """
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.rules = []
        self.index = None
        self._load_memory()

    def _load_memory(self):
        print(f"   Loading Statutory Rules from: {RULES_DIR}")
        if not os.path.exists(RULES_DIR):
            print(f"   Error: '{RULES_DIR}' folder not found.")
            return

        # 1. Load Raw JSONs
        raw_data = []
        for f in os.listdir(RULES_DIR):
            if f.endswith(".json"):
                theme = f.replace(".json", "") 
                path = os.path.join(RULES_DIR, f)
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        if isinstance(data, list):
                            for r in data:
                                r['source_theme'] = theme
                                raw_data.append(r)
                except Exception as e:
                    print(f"   Error reading {f}: {e}")

        self.rules = raw_data

        # 2. Build or Load FAISS Index
        if os.path.exists(CACHE_FILE):
            # print("    Loading cached Vector Index...")
            vectors = np.load(CACHE_FILE)
            if len(vectors) != len(self.rules):
                print("    Data mismatch. Rebuilding index...")
                self._build_new_index()
            else:
                self._build_faiss_index(vectors)
        else:
            self._build_new_index()
        
        print(f"    Indexed {len(self.rules)} statutory rules.")

    def _build_new_index(self):
        print("   Building Index (This takes time once)...")
        if not self.rules: return
            
        texts = [
            f"{r['source_theme']} {r.get('section','')} {r.get('title','')} {r.get('consequence','')}"
            for r in self.rules
        ]
        vectors = self.embed_model.encode(texts, show_progress_bar=True)
        np.save(CACHE_FILE, vectors)
        self._build_faiss_index(vectors)

    def _build_faiss_index(self, vectors):
        vectors = vectors.astype('float32')
        faiss.normalize_L2(vectors)
        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

    def search(self, query, top_k=5):
        if self.index is None or not self.rules: return []

        # A. Semantic Search (FAISS)
        query_vec = self.embed_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, k=25)
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.rules):
                rule = self.rules[idx].copy()
                rule['score'] = float(distances[0][i])
                candidates.append(rule)

        # B. Keyword Boosting (Re-ranking)
        query_lower = query.lower()
        results = []
        for rule in candidates:
            score = rule['score']
            sec = str(rule.get('section', '')).lower()
            theme = str(rule.get('source_theme', '')).lower()
            
            # Boost matches if the query explicitly mentions section numbers
            if sec and sec in query_lower: score += 0.8
            if theme in query_lower: score += 0.2
            
            rule['final_score'] = score
            results.append(rule)

        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]


class LegalBrain:
    def __init__(self):
        print("\n Initializing Recursive Legal Brain (Local)...")
        
        # 1. Initialize Reasoning Model (Ollama)
        self.llm_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        
        # 2. Initialize Embedding Model
        print(f"   Loading Embeddings ({EMBED_MODEL_NAME})...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        
        # 3. Connect to Case Law Database (Weaviate)
        print("   🔌 Connecting to Weaviate (Case Law)...")
        self.db_client = None
        try:
            self.db_client = get_client()
            self.collection = self.db_client.collections.get(COLLECTION_NAME)
        except Exception as e:
            print(f"    DB Connection Failed: {e}")
            self.collection = None
        
        # 4. Initialize Rule Engine (FAISS)
        self.rule_engine = RuleSearchEngine(self.embed_model)

    def search_cases(self, query, limit=3):
        """Fetches relevant precedents from Weaviate."""
        if not self.collection: return []
        try:
            vec = self.embed_model.encode(query)
            response = self.collection.query.near_vector(near_vector=vec, limit=limit)
            return response.objects
        except Exception:
            return []

    def think(self, user_query):
        print(f"\n Processing Query: '{user_query}'")
        
        # --- PHASE 1: PRECEDENT RETRIEVAL (Weaviate) ---
        print("   1. Searching Case Law (Precedents)...")
        cases = self.search_cases(user_query)
        print(f"      Found {len(cases)} relevant cases.")
        
        # --- PHASE 2: RECURSIVE DISCOVERY ---
        # Scan the found cases for mentions of specific Sections (e.g., "Section 79")
        print("   2. Analyzing cases for hidden statutory references...")
        
        case_text_blob = " ".join([c.properties.get('content', '') for c in cases]) if cases else ""
        
        # Regex to find "Section 123", "Article 21", "Order 39", etc.
        found_sections = list(set(re.findall(r'((?:Section|Article|Order|Rule)\s+\d+[A-Za-z]?)', case_text_blob, re.IGNORECASE)))
        
        smart_query = user_query
        if found_sections:
            print(f"      Detected implied sections: {found_sections}")
            # We append these sections to the query to force the Rule Engine to find them
            smart_query = f"{user_query} {' '.join(found_sections)}"

        # --- PHASE 3: STATUTORY SEARCH (FAISS) ---
        print(f"   3. Searching Statutes (FAISS) using Smart Query...")
        rules = self.rule_engine.search(smart_query, top_k=5)
        
        # --- PHASE 4: CONTEXT ASSEMBLY ---
        context_str = ""
        
        if cases:
            context_str += "--- RELEVANT CASE PRECEDENTS ---\n"
            for i, c in enumerate(cases):
                # We limit text length to avoid overflowing context window
                content = c.properties.get('content', '')[:500].replace('\n', ' ')
                source = c.properties.get('source', 'Unknown Case')
                context_str += f"[Case {i+1}] {source}: \"{content}...\"\n\n"
            
        if rules:
            context_str += "--- APPLICABLE STATUTES & ACTS ---\n"
            for r in rules:
                context_str += f"[{r['source_theme']}] {r.get('section')}: {r.get('title')}\n"
                context_str += f"   -> Legal Consequence: {r.get('consequence')}\n\n"
        
        if not context_str:
            context_str = "No specific legal documents found. Rely on general legal principles."

        # --- PHASE 5: PROMPT ENGINEERING (Conflict Aware) ---
        prompt = f"""
        [ROLE]
        You are a Senior Legal Strategist for the Supreme Court.
        
        [INPUT DATA]
        USER QUERY: "{user_query}"
        
        LEGAL CONTEXT (Sources of Law):
        {context_str}
        
        [CRITICAL INSTRUCTIONS]
        You must structure your response EXACTLY as follows:
        
        PART 1: INTERNAL REASONING
        Start this section with "### DEEP THOUGHTS".
        - First, analyze the Case Law. What did the courts decide?
        - Second, analyze the Statutes. What do the written rules say?
        - **CONFLICT CHECK:** Does the User's contract conflict with a Statute (e.g., Section 27 vs Non-Compete)? If yes, STATUTES usually override private contracts.
        - **DEFENSE CHECK:** Does the company claim a defense (e.g., Safe Harbor) that is invalidated by their actions (e.g., Active Logistics)?
        
        PART 2: FINAL OPINION
        Start this section with "### LEGAL OPINION".
        - Provide the final professional advice to the user.
        - Cite the specific Sections and Case Names found in the context.
        """

        # --- PHASE 6: EXECUTION ---
        print("   4. Synthesizing Opinion (DeepSeek-R1)...")
        try:
            response = self.llm_client.chat.completions.create(
                model=GEN_MODEL,
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.6,
                max_tokens=4000
            )
            
            raw = response.choices[0].message.content
            
            # --- PHASE 7: ROBUST PARSING (Handle Missing Tags) ---
            # 1. Try Standard XML Tags
            match_xml = re.search(r'<think>(.*?)</think>', raw, re.DOTALL | re.IGNORECASE)
            # 2. Try Our Custom Headers
            match_custom = re.search(r'### DEEP THOUGHTS(.*?)(?=### |### LEGAL)', raw, re.DOTALL | re.IGNORECASE)
            
            thought_content = ""
            final_answer = raw

            if match_xml:
                thought_content = match_xml.group(1).strip()
                final_answer = raw.replace(match_xml.group(0), "").strip()
            elif match_custom:
                thought_content = match_custom.group(1).strip()
                # Clean up the output by removing the thought section from the final string
                final_answer = raw.replace(match_custom.group(0), "").replace("### DEEP THOUGHTS", "").strip()
                final_answer = re.sub(r'###  LEGAL OPINION', '', final_answer, flags=re.IGNORECASE).strip()

            # Format the display
            output = ""
            if thought_content:
                output += "\n" + "="*20 + " INTERNAL REASONING " + "="*20 + "\n"
                output += thought_content + "\n"
                output += "="*60 + "\n"
            
            output += "\n### LEGAL OPINION\n" + final_answer
            return output

        except Exception as e:
            return f"Error: {e}"

    def close(self):
        """Gracefully closes database connections."""
        if self.db_client:
            print("\n   🔌 Closing Weaviate Connection...")
            self.db_client.close()


if __name__ == "__main__":
    brain = None
    try:
        brain = LegalBrain()
        print("\nSystem Ready. Type 'exit' to quit.")
        
        while True:
            q = input("\n Ask Legal Brain: ")
            if q.lower() in ["exit", "quit"]: break
            if not q.strip(): continue
            
            start_t = time.time()
            print("\n" + "-"*60)
            print(brain.think(q))
            print("-"*60)
            print(f"Total Time: {time.time()-start_t:.2f}s")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if brain:
            brain.close()