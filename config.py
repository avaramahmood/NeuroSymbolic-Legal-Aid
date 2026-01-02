import os
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv

load_dotenv()

# --- CONSTANTS ---
W_URL = os.getenv("WEAVIATE_URL")
W_KEY = os.getenv("WEAVIATE_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

# --- CATEGORY TAGS ---
TAG_RULE = "statutory_rule"      
TAG_TREND = "judiciary_trend"    
TAG_CASE = "case_law" 

DATA_DIR = "data"
LAWS_FOLDER = "latest_law"

def get_client():
    """
    Connects to your Weaviate Cloud Sandbox.
    """
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=W_URL,
        auth_credentials=Auth.api_key(W_KEY),
        # We add this header just in case, though we are doing vectorization locally
        headers={"X-Ollama-BaseURL": "http://localhost:11434"} 
    )