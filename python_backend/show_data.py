import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("uniform_manual")

data = collection.get()

print(f"Total chunks: {len(data['documents'])}\n")
print("Showing first 5 chunks:\n")

for idx in range(min(5, len(data["documents"]))):
    print(f"===== Chunk {idx} =====")
    print(f"Document:\n{data['documents'][idx]}")
    print(f"Metadata: {data['metadatas'][idx]}")
    print()