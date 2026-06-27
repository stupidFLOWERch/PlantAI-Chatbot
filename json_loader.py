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
    """
    构建用于 embedding 的文本。
    用自然语言格式，让向量检索更精准。
    """
    topic = chunk.get("topic", "")
    content = chunk.get("content", "")
    applies_to = chunk.get("applies_to", [])
    
    # 把 applies_to 展开成自然语言
    if applies_to:
        # 如果是列表，用 "and" 连接
        applies_str = " and ".join(applies_to)
    else:
        applies_str = "all members"
    
    # 自然语言格式
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