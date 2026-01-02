import weaviate.classes.config as wvc
from config import get_client, COLLECTION_NAME

def setup_schema():
    client = get_client()
    
    print(f"  Connecting to Weaviate Cloud...")

    # 1. Reset Collection (Delete old data)
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
        print(f"  Deleted existing '{COLLECTION_NAME}' in Cloud.")

    # 2. Create New Structure
    client.collections.create(
        name=COLLECTION_NAME,
        
        # KEY SETTING: We bring our own vectors!
        vectorizer_config=wvc.Configure.Vectorizer.none(),
        
        # We enable Generative Search (RAG) just in case we want the Cloud to try it later,
        # though we will mostly do this locally too.
        generative_config=wvc.Configure.Generative.ollama(
            api_endpoint="http://host.docker.internal:11434", # Placeholder for cloud
            model="deepseek-r1:7b"
        ),
        
        properties=[
            wvc.Property(name="content", data_type=wvc.DataType.TEXT),
            wvc.Property(name="source", data_type=wvc.DataType.TEXT),
            wvc.Property(name="year", data_type=wvc.DataType.INT),
            wvc.Property(name="category", data_type=wvc.DataType.TEXT),
        ]
    )
    
    print(f" Schema created successfully on Weaviate Cloud!")
    client.close()

if __name__ == "__main__":
    setup_schema()