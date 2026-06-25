import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("plant_knowledge")

data = collection.get()

for doc in data["documents"]:
    print("-----")
    print(doc)
