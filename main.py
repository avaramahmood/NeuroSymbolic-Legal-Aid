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

# Import settings
from config import (
    COLLECTION_NAME, 
    EMBED_MODEL_NAME, 
    GEN_MODEL, 
    OLLAMA_BASE_URL,
    get_client
)

# --- PART 1: THE FAISS SEARCH ENGINE ---
class RuleSearchEngine:
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
            print("    Loading cached Vector Index...")
            vectors = np.load(CACHE_FILE)
            if len(vectors) != len(self.rules):
                print("    Data mismatch. Rebuilding index...")
                self._build_new_index()
            else:
                self._build_faiss_index(vectors)
        else:
            self._build_new_index()
        
        print(f"    Indexed {len(self.rules)} rules in memory.")

    def _build_new_index(self):
        print("    Building Index (This takes time once)...")
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
            
            # Boost specific matches
            if sec and sec in query_lower: score += 0.8
            if theme in query_lower: score += 0.2
            
            rule['final_score'] = score
            results.append(rule)

        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]

# --- PART 2: THE RECURSIVE BRAIN (Logic Core) ---
class LegalBrain:
    def __init__(self):
        print("\n Initializing Recursive Legal Brain (Local)...")
        self.llm_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        
        print(f"   Loading AI Model ({EMBED_MODEL_NAME})...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        
        print("   Connecting to Embedded Weaviate...")
        try:
            self.db_client = get_client()
            self.collection = self.db_client.collections.get(COLLECTION_NAME)
        except Exception as e:
            print(f"    DB Connection Failed: {e}")
            self.collection = None
        
        self.rule_engine = RuleSearchEngine(self.embed_model)

    def search_cases(self, query, limit=3):
        if not self.collection: return []
        try:
            vec = self.embed_model.encode(query)
            response = self.collection.query.near_vector(near_vector=vec, limit=limit)
            return response.objects
        except Exception:
            return []

    def think(self, user_query):
        print(f"\n🔍 Query: '{user_query}'")
        
        # --- STEP 1: ROBUST RETRIEVAL ---
        print("   1. Searching Case Law & Statutes...")
        try:
            cases = self.search_cases(user_query)
            # Safe extraction of text for the search query
            case_text = " ".join([c.properties.get('content', '') for c in cases]) if cases else ""
            
            # Extract section numbers (e.g., "Section 79") to refine the rule search
            found_sections = list(set(re.findall(r'((?:Section|Article|Order|Rule)\s+\d+[A-Za-z]?)', case_text, re.IGNORECASE)))
            
            smart_query = f"{user_query} {' '.join(found_sections)}" if found_sections else user_query
            rules = self.rule_engine.search(smart_query, top_k=5)
        except Exception as e:
            print(f"      [!] Retrieval Warning: {e}")
            cases, rules = [], []

        # --- STEP 2: CONTEXT PREPARATION ---
        context_str = ""
        if cases:
            context_str += "--- RELEVANT PRECEDENTS ---\n"
            for c in cases:
                # Truncate to prevent context overflow
                snippet = c.properties.get('content', '')[:400].replace('\n', ' ')
                context_str += f"- {snippet}...\n"
        
        if rules:
            context_str += "\n--- STATUTORY RULES ---\n"
            for r in rules:
                context_str += f"- {r.get('section')}: {r.get('title')} -> {r.get('consequence')}\n"
        
        if not context_str:
            context_str = "No specific case law or statutes found in database. Rely on general legal principles."

        # --- STEP 3: PROMPT ENGINEERING (The "Jailbreak") ---
        # We enforce a specific structure to ensure we can parse the output even if <think> tags fail.
        prompt = f"""
        [ROLE]
        You are a Senior Legal Strategist for the Indian Supreme Court.
        
        [INPUT DATA]
        USER QUERY: "{user_query}"
        LEGAL CONTEXT:
        {context_str}
        
        [INSTRUCTIONS]
        You must strictly follow this format. Do not output anything else.
        
        PART 1: INTERNAL REASONING
        Start with the exact header: "###  DEEP THOUGHTS"
        - Analyze the conflict between the User's situation and the Statutes/Precedents.
        - If laws conflict (e.g., Contract Act vs Consumer Act), debate which one prevails.
        - Be skeptical of the company's claims.
        
        PART 2: FINAL VERDICT
        Start with the exact header: "###  LEGAL OPINION"
        - Provide clear, actionable advice to the user.
        - Cite specific sections from the context provided.
        """

        # --- STEP 4: EXECUTION & PARSING ---
        print("   4. Synthesizing Opinion (DeepSeek-R1)...")
        try:
            response = self.llm_client.chat.completions.create(
                model=GEN_MODEL,
                messages=[{"role": "user", "content": prompt}], # 'User' role is better for R1 models
                temperature=0.6, # 0.6 is the sweet spot for reasoning
                max_tokens=4000  # Ensure it doesn't cut off mid-thought
            )
            
            raw_output = response.choices[0].message.content
            
            # --- ROBUST PARSING LOGIC ---
            # 1. Try to find standard <think> tags (native DeepSeek behavior)
            think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL | re.IGNORECASE)
            
            # 2. If no tags, look for our custom headers
            custom_think_match = re.search(r'###  DEEP THOUGHTS(.*?)(?=### |### LEGAL)', raw_output, re.DOTALL | re.IGNORECASE)
            
            thought_content = ""
            final_answer = raw_output

            if think_match:
                thought_content = think_match.group(1).strip()
                final_answer = raw_output.replace(think_match.group(0), "").strip()
            elif custom_think_match:
                thought_content = custom_think_match.group(1).strip()
                # Remove the thought part from the final answer
                final_answer = raw_output.replace(custom_think_match.group(0), "").replace("### DEEP THOUGHTS", "").strip()
                # Clean up the second header if it exists
                final_answer = re.sub(r'###  LEGAL OPINION', '', final_answer, flags=re.IGNORECASE).strip()

            # --- FORMATTING THE OUTPUT ---
            formatted_output = ""
            
            if thought_content:
                formatted_output += f"\n###  DEEP THOUGHTS\n{thought_content}\n"
                formatted_output += "\n" + "-"*40 + "\n"
            
            formatted_output += f"\n###  LEGAL OPINION\n{final_answer}"
            
            return formatted_output

        except Exception as e:
            return f" **Error generating legal opinion:** {str(e)}\n\n*Check if your Ollama server is running 'deepseek-r1:7b'*"
    def close(self):
        """Gracefully closes the Weaviate client."""
        if self.db_client:
            print("\n    Closing Weaviate Connection...")
            self.db_client.close()

if __name__ == "__main__":
    try:
        brain = LegalBrain()
        print("\n System Ready. Type 'exit' to quit.")
        
        while True:
            q = input("\n  Ask Legal Brain: ")
            if q.lower() in ["exit", "quit"]: break
            if not q.strip(): continue
            
            start_t = time.time()
            print("\n" + "="*60)
            print(brain.think(q))
            print("="*60)
            print(f" Total Time: {time.time()-start_t:.2f}s")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    