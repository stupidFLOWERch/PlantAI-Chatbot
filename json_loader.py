import chromadb
import json

client = chromadb.PersistentClient(path="./chroma_db")

# Reset DB
try:
    client.delete_collection("uniform_manual")
except:
    pass

collection = client.get_or_create_collection("uniform_manual")

def build_embedding(chunk):
    # use natural language format to make semantic search more precise

    topic = chunk.get("topic", "")
    content = chunk.get("content", "")
    applies_to = chunk.get("applies_to", [])
    
    #  extend applies_to to natural language
    if applies_to:
        # If it's a list, join the items with "and"
        applies_str = " and ".join(applies_to)
    else:
        applies_str = "all members"
    
    # Natural language format
    parts = [
        f"This document is about {topic}.",
        f"It applies to {applies_str}.",
        f"Content: {content}"
    ]
    
    return " ".join(parts)

# Load JSON
with open("uniform_manual.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Insert into ChromaDB
for idx, chunk in enumerate(data):
    embedding_text = build_embedding(chunk)
    
    metadata = {
        "topic": chunk.get("topic", ""),
        "applies_to": chunk.get("applies_to", [])
    }
    
    collection.add(
        documents=[embedding_text],
        metadatas=metadata,
        ids=[str(idx)]
    )

print(f"DB rebuilt successfully! Added {len(data)} chunks.")