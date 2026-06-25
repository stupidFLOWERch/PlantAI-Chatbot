import chromadb
import json

client = chromadb.PersistentClient(path="./chroma_db")

# -------------------------
# Reset DB
# -------------------------
try:
    client.delete_collection("plant_knowledge")
except:
    pass

collection = client.get_or_create_collection("plant_knowledge")

# -------------------------
# Build embedding text (NEW FORMAT)
# -------------------------
def build_embedding(topic, chunk):

    parts = [
        f"Topic: {topic}",
        f"Type: {chunk.get('type', '')}",
        f"Text: {chunk.get('text', '')}"
    ]

    return "\n".join(parts)

# -------------------------
# Load JSON
# -------------------------
with open("plants.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------
# Insert into ChromaDB (CHUNK-BASED)
# -------------------------
doc_id = 0

for item in data:

    topic = item.get("topic", "unknown")
    level = item.get("level", "unknown")

    chunks = item.get("chunks", [])

    for chunk in chunks:

        embedding_text = build_embedding(topic, chunk)

        collection.add(
            documents=[embedding_text],
            metadatas={
                "topic": topic,
                "level": level,
                "type": chunk.get("type", "unknown")
            },
            ids=[str(doc_id)]
        )

        doc_id += 1

print("DB rebuilt successfully!")