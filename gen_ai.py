import chromadb
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# Gemini Client
client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("plant_knowledge")

history = []

# -------------------------
# Memory
# -------------------------
def update_memory(user, ai):
    history.append({
        "user": user,
        "ai": ai
    })

    if len(history) > 5:
        history.pop(0)


# -------------------------
# Intent Detection
# -------------------------
def detect_intent(text):
    text = text.lower()

    if any(w in text for w in ["hi", "hello", "hey"]):
        return "greeting"

    return "question"


# -------------------------
# Router
# -------------------------
def route(intent):

    if intent == "greeting":
        return "direct"

    if intent == "question":
        return "rag"

    return "fallback"


# -------------------------
# Retrieve Knowledge
# -------------------------
def retrieve(prompt):
    results = collection.query(
        query_texts=[prompt],
        n_results=3,
        include=["documents", "distances"]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    # 没有相关知识
    valid_docs = []

    for doc, dist in zip(documents, distances):
        if dist < 0.8:
            valid_docs.append(doc)

    if not valid_docs:
        return None

    return "\n\n".join(valid_docs)




# -------------------------
# Chat
# -------------------------
def chat(prompt):

    intent = detect_intent(prompt)
    mode = route(intent)

    if mode == "direct":
        return "Hello! I am your plant biology assistant."
    
    context = retrieve(prompt)

    if context is None:
        return "Sorry, I don't have enough information in my knowledge base."

    memory = ""

    for h in history:
        memory += f"User: {h['user']}\nAI: {h['ai']}\n"


    full_prompt = f"""
You are a plant biology teacher.

Previous Conversation:
{memory}

Knowledge:
{context}

Question:
{prompt}

Previous Conversation is ONLY for understanding pronouns like "it", "this".
Do NOT treat it as factual knowledge.
Answer based on the knowledge.

"""

    response = client_ai.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=full_prompt
    )

    answer = response.text

    update_memory(prompt, answer)

    return answer


# -------------------------
# Main Loop
# -------------------------
while True:

    msg = input("You: ")

    if msg.lower() == "exit":
        break

    answer = chat(msg)

    print("AI:", answer)