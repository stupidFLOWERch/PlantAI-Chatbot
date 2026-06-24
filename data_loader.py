import chromadb
import re

client = chromadb.PersistentClient(path="./chroma_db")

def parse(chunk):
    topic = re.search(r"TOPIC:\s*(.*)", chunk)
    type_ = re.search(r"TYPE:\s*(.*)", chunk)
    factor = re.search(r"FACTOR:\s*(.*)", chunk)
    content = re.search(r"CONTENT:\s*(.*)", chunk, re.S)

    topic = topic.group(1).strip() if topic else "unknown"
    type_ = type_.group(1).strip() if type_ else "unknown"
    factor = factor.group(1).strip() if factor else None
    content = content.group(1).strip() if content else chunk

    # 🔥 build better embedding text
    if factor:
        embedding_text = f"{factor}. {content}"
    else:
        embedding_text = content

    return topic, type_, factor, embedding_text

# 🔥 delete old collection if exists
try:
    client.delete_collection("plant_knowledge")
except:
    pass

# recreate fresh collection
collection = client.get_or_create_collection(
    name="plant_knowledge"
)

with open("plants.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = text.split("\n\n")
chunks = [c.strip() for c in chunks if c.strip()]

for i, chunk in enumerate(chunks):
    topic, type_, factor, embedding_text = parse(chunk)

    collection.add(
    documents=[embedding_text],
    metadatas={
        "topic": topic,
        "type": type_,
        "factor": factor if factor is not None else ""
    },
    ids=[str(i)]
)

print("DB rebuilt successfully!")