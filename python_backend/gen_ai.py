import chromadb
from google import genai
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()

# Gemini Client
client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("uniform_manual")

history = []


# -------------------------
# Conversation Memory
# Stores the last 3 user-assistant exchanges.
# -------------------------
def update_memory(user, ai):
    history.append({
        "user": user,
        "ai": ai
    })
    if len(history) > 3:
        history.pop(0)


# -------------------------
# Intent Detection
# -------------------------
def detect_intent(text):
    text = text.lower().strip()

    question_words = ["what", "why", "how", "when", "where", "can", "is", "are", "do", "does"]
    greeting_words = ["hi", "hello", "hey", "good morning", "good afternoon"]

    is_greeting = any(re.search(rf"\b{word}\b", text) for word in greeting_words)
    is_question = (
        "?" in text or
        any(word in text for word in question_words)
    )

    if is_question:
        return "question"
    if is_greeting:
        return "greeting"
    return "question"


# -------------------------
# Detect Uniform Context
# -------------------------
def detect_uniform_context(text):

    # Extract uniform-related labels from the user's query.

    text = text.lower().strip()
    
    role_keywords = {
        "officers": ["officer", "officers"],
        "seniors": ["senior", "seniors"],
        "juniors": ["junior", "juniors"],
        "pre-juniors": ["pre-junior", "pre-juniors"],
        "primers": ["primer", "primers"],
        "teacher_advisors": ["teacher advisor", "teacher advisors"],
        "warrant_officers": ["warrant officer", "warrant officers"],
        "captains": ["captain", "captains"],
        "chaplains": ["chaplain", "chaplains"],
        "boys": ["boys"],
        "girls": ["girls"]
    }
    
    dress_keywords = {
        "ceremonial": ["ceremonial", "ceremony"],
        "formal": ["formal"],
        "day": ["day dress"],
        "drill": ["drill kit"],
        "pt": ["pt kit"]
    }
    
    part_keywords = {
        "badge": ["badge", "badges", "award", "awards", "medal", "medals", "chevron"],
        "cap": ["glengarry", "field service cap"],
        "shirt": ["shirt", "shirts", "blazer"],
        "trousers": ["trouser", "trousers", "pants"],
        "skirt": ["skirt"],
        "belt": ["belt", "belts"],
        "haversack": ["haversack", "pouch"],
        "footwear": ["shoe", "shoes", "boot", "boots", "sock", "socks"],
        "tie": ["tie", "necktie"],
        "lanyard": ["lanyard", "whistle"],
        "name tag": ["name tag", "nametag"]
    }
    
    detected = []
    
    # detect role
    for role, keywords in role_keywords.items():
        for kw in keywords:
            if kw in text:
                detected.append(role)
                break
    
    # detect dress type
    for dress, keywords in dress_keywords.items():
        for kw in keywords:
            if kw in text:
                detected.append(dress)
                break
    
    # detect uniform part
    for part, keywords in part_keywords.items():
        for kw in keywords:
            if kw in text:
                detected.append(part)
                break
    
    return detected


# Route the request based on detected intent.
def route(intent):
    if intent == "greeting":
        return "direct"
    if intent == "question":
        return "rag"
    return "fallback"


# -------------------------
# Retrieve Knowledge
# -------------------------
def retrieve(prompt, labels):

    # 1. Perform semantic search using ChromaDB.
    results = collection.query(
        query_texts=[prompt],
        n_results=5,
        include=["documents", "metadatas"]
    )
    
    docs = results["documents"][0]
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    
    print(f"=== RETRIEVED {len(docs)} DOCS (VECTOR) ===")
    for i, doc in enumerate(docs):
        meta = metadatas[i] if i < len(metadatas) else {}
        print(f"  [{i}] {meta.get('topic', 'Unknown')}")
    
    # 2. If too few relevant documents are retrieved,
    #    perform keyword-based retrieval using detected labels.
    if len(docs) < 3 and labels:
        print(f"=== Vector only got {len(docs)}, trying keyword fallback with labels: {labels} ===")
        
        # Retrieve all indexed documents for keyword matching.
        all_data = collection.get(include=["documents", "metadatas"])
        all_docs = all_data["documents"]
        all_metas = all_data["metadatas"]
        
        # Match metadata using extracted labels.
        for i, meta in enumerate(all_metas):
            if all_docs[i] in docs:
                continue  # 跳过已有的
            
            topic = meta.get("topic", "").lower()
            applies = " ".join(meta.get("applies_to", [])).lower()
            combined = f"{topic} {applies}"
            
            for label in labels:
                if label in combined:
                    docs.append(all_docs[i])
                    metadatas.append(meta)
                    print(f"  [+] Keyword match: {meta.get('topic', 'Unknown')} (label: {label})")
                    break
        
        docs = docs[:5]
        metadatas = metadatas[:5]
    
    # 3. Return the top 5 semantic search results if fall back.
    if len(docs) < 3:
        print(f"=== Still only {len(docs)} docs, using top 5 from vector search ===")
        # 重新取 5 条向量结果
        results = collection.query(
            query_texts=[prompt],
            n_results=5,
            include=["documents", "metadatas"]
        )
        docs = results["documents"][0]
        metadatas = results["metadatas"][0] if results["metadatas"] else []
    
    if not docs:
        return ""
    
    # 4. format results
    formatted_docs = []
    for i, doc in enumerate(docs):
        meta = metadatas[i] if i < len(metadatas) else {}
        topic = meta.get("topic", "Unknown")
        formatted_docs.append(f"[Topic: {topic}]\n{doc}")
    
    return "\n\n---\n\n".join(formatted_docs)


# -------------------------
# Chat
# -------------------------
def chat(prompt):
    intent = detect_intent(prompt)
    mode = route(intent)

    if mode == "direct":
        return "Hello! What can I help you with? You can ask me questions about the Boys' Brigade uniform."

    # Extract context labels from the user's query.
    labels = detect_uniform_context(prompt)
    
    # retrieve knowledge from db
    context = retrieve(prompt, labels)

    print("=== CONTEXT START ===")
    print(f"Labels detected: {labels}")
    print(context)
    print("=== CONTEXT END ===")

    if not context:
        return "I cannot find this in my uniform knowledge base. Please check the Uniform Manual or ask a different question."

    memory = ""
    if history:
        memory = "\n".join([
            f"User: {h['user']}\nAI: {h['ai']}"
            for h in history
        ])

    full_prompt = f"""
You are a strict Boys' Brigade uniform advisor.

RULES:
1. Use ONLY the provided KNOWLEDGE.
2. Do NOT add extra information not in KNOWLEDGE.
3. Do NOT expand beyond the question.
4. Keep answer short and concise (max 3-5 sentences).
5. If knowledge is insufficient, say "I cannot find this in my uniform knowledge base."
6. Be precise about uniform regulations - include specific colours, placements, and dress codes when mentioned.

KNOWLEDGE (use this only):
-------------------------
{context}
-------------------------

QUESTION:
{prompt}

Previous Conversation:
{memory}
"""
    # Generate the final response using Gemini with the retrieved context.
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