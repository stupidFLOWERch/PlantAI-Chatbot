import chromadb
import json

client = chromadb.PersistentClient(path="./chroma_db")

# -------------------------
# Reset DB
# -------------------------
try:
    client.delete_collection("uniform_manual")
except:
    pass

collection = client.get_or_create_collection("uniform_manual")

# -------------------------
# Build embedding text
# -------------------------
def build_embedding(chunk):
    """
    构建用于 embedding 的文本。
    把关键信息都放进去，让向量检索更精准。
    """
    topic = chunk.get("topic", "")
    content = chunk.get("content", "")
    applies_to = chunk.get("applies_to", [])
    
    # 把 applies_to 变成可读字符串，辅助语义
    applies_str = ", ".join(applies_to) if applies_to else ""
    
    parts = [
        f"Topic: {topic}",
        f"Applies to: {applies_str}",
        f"Content: {content}"
    ]
    
    return "\n".join(parts)

# -------------------------
# Load JSON
# -------------------------
with open("uniform_manual.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------
# Insert into ChromaDB
# -------------------------
for idx, chunk in enumerate(data):
    
    # 构建 embedding 文本
    embedding_text = build_embedding(chunk)
    
    # 提取 metadata
    metadata = {
        "topic": chunk.get("topic", ""),
        "applies_to": chunk.get("applies_to", []),  # ChromaDB 支持 array
    }
    
    # 可选：如果有 dress_code，也加进去方便过滤
    # 但你的当前 chunk 里已经把 dress_code 合并到 applies_to 了
    # 例如 ["officers", "ceremonial"]，所以不需要额外字段
    
    collection.add(
        documents=[embedding_text],
        metadatas=metadata,
        ids=[str(idx)]
    )

print(f"DB rebuilt successfully! Added {len(data)} chunks.")