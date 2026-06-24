import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("plant_knowledge")

results = collection.get()

print(results)
