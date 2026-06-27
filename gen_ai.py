import chromadb
from google import genai
from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
import os
import json
import re

load_dotenv()

# Gemini Client
client_ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("uniform_manual")

# reranker
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
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
# def detect_uniform_context(text):

#     # Extract uniform-related labels from the user's query.

#     text = text.lower().strip()
    
#     role_keywords = {
#         "officers": ["officer", "officers"],
#         "seniors": ["senior", "seniors"],
#         "juniors": ["junior", "juniors"],
#         "pre-juniors": ["pre-junior", "pre-juniors"],
#         "primers": ["primer", "primers"],
#         "teacher_advisors": ["teacher advisor", "teacher advisors"],
#         "warrant_officers": ["warrant officer", "warrant officers"],
#         "captains": ["captain", "captains"],
#         "chaplains": ["chaplain", "chaplains"],
#         "boys": ["boys"],
#         "girls": ["girls"],
#         "senior_ncos": ["senior nco", "senior ncos", "nco", "ncos"],
#         "staff_sergeants": ["staff sergeant", "staff sergeants"],
#         "sergeants": ["sergeant", "sergeants"],
#     }
    
#     dress_keywords = {
#         "ceremonial": ["ceremonial", "ceremony"],
#         "formal": ["formal"],
#         "day": ["day dress"],
#         "drill": ["drill kit"],
#         "pt": ["pt kit"]
#     }
    
#     part_keywords = {
#         "badge": ["badge", "badges", "award", "awards", "medal", "medals", "chevron"],
#         "cap": ["glengarry", "field service cap"],
#         "shirt": ["shirt", "shirts", "blazer"],
#         "trousers": ["trouser", "trousers", "pants"],
#         "skirt": ["skirt"],
#         "belt": ["belt", "belts"],
#         "haversack": ["haversack", "pouch"],
#         "footwear": ["shoe", "shoes", "boot", "boots", "sock", "socks"],
#         "tie": ["tie", "necktie"],
#         "lanyard": ["lanyard", "whistle"],
#         "name tag": ["name tag", "nametag"]
#     }
    
#     detected = []
    
#     # detect role
#     for role, keywords in role_keywords.items():
#         for kw in keywords:
#             if kw in text:
#                 detected.append(role)
#                 break
    
#     # detect dress type
#     for dress, keywords in dress_keywords.items():
#         for kw in keywords:
#             if kw in text:
#                 detected.append(dress)
#                 break
    
#     # detect uniform part
#     for part, keywords in part_keywords.items():
#         for kw in keywords:
#             if kw in text:
#                 detected.append(part)
#                 break
    
#     return detected


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
def retrieve(prompt):

    # 1. semantic search for 20 results
    results = collection.query(
        query_texts=[prompt],
        n_results=20,
        include=["documents", "metadatas"]
    )
    
    docs = results["documents"][0]
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    
    print(f"=== RETRIEVED {len(docs)} DOCS (VECTOR) ===")
    for i, doc in enumerate(docs):
        meta = metadatas[i] if i < len(metadatas) else {}
        print(f"  [{i}] {meta.get('topic', 'Unknown')}")
    
    # 2. skip reranker if results too few
    if len(docs) <= 2:
        print("=== Too few docs, skipping rerank ===")
        return format_docs(docs, metadatas)
    
    # 3. Use reranker 
    print(f"=== RERANKING {len(docs)} DOCS ===")
    pairs = [[prompt, doc] for doc in docs]
    scores = reranker.compute_score(pairs, normalize=True)
    
    # 4. Get the top 3 results
    sorted_pairs = sorted(zip(docs, metadatas, scores), key=lambda x: x[2], reverse=True)
    
    print(f"=== RERANKED TOP 3 ===")
    for i, (doc, meta, score) in enumerate(sorted_pairs[:3]):
        print(f"  [{i}] {meta.get('topic', 'Unknown')} (score: {score:.4f})")
    
    top_docs = [item[0] for item in sorted_pairs[:3]]
    top_metas = [item[1] for item in sorted_pairs[:3]]
    
    return format_docs(top_docs, top_metas)


# format results
def format_docs(docs, metadatas):
    if not docs:
        return ""
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

    # labels = detect_uniform_context(prompt)
    
    if history:
        last_user = history[-1].get("user", "")
        enhanced_prompt = f"{last_user} {prompt}"
        print(f"=== ENHANCED PROMPT: {enhanced_prompt} ===")
    else:
        last_user = ""
        enhanced_prompt = prompt
    
    context = retrieve(enhanced_prompt)

    print("=== CONTEXT START ===")
    # print(f"Labels detected: {labels}")
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
7. **If the current question is a follow-up (e.g., "how about senior", "what about officers"), treat it as a continuation of the previous topic.**

KNOWLEDGE (use this only):
-------------------------
{context}
-------------------------

CURRENT QUESTION:
{prompt}

Previous Conversation:
{memory}
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