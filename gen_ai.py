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

    if len(history) > 3:
        history.pop(0)


# -------------------------
# Intent Detection
# -------------------------
def detect_intent(text):
    text = text.lower().strip()

    question_words = ["what", "why", "how", "when", "where"]
    greeting_words = ["hi", "hello", "hey", "good morning", "good afternoon"]

    is_greeting = any(re.search(rf"\b{word}\b", text) for word in greeting_words)

    is_question = (
        "?" in text or
        any(word in text for word in question_words)
    )

    if is_question:
        return "question"

    # greeting only
    if is_greeting:
        return "greeting"

    # fallback
    return "question"

def detect_topic(text):

    prompt = f"""
You are a biology classifier.

Classify the question into ONE topic only:

Topics:
- Plant Cell
- Photosynthesis
- Transpiration
- Germination
- Plant Transport System
- Plant Nutrition
- Plant Reproduction
- Seed Dispersal
- Plant Adaptation
- Plant Tropism

If not related, return "unknown".

Return ONLY the topic name.

Question:
{text}
"""

    response = client_ai.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text.strip()

def detect_answer_type(text):

    text = text.lower()

    if "what is" in text:
        return "definition"
        
    if "why" in text:
        return "purpose"

    if "explain" in text:
        return "explanation"

    if "mistake" in text or "common" in text:
        return "misconception"

    return "general"

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
def retrieve(prompt, topic=None):

    if topic:
        results = collection.query(
            query_texts=[prompt],
            n_results=3,
            where={"topic": topic},
            include=["documents", "metadatas"]
        )
    else:
        results = collection.query(
            query_texts=[prompt],
            n_results=3,
            include=["documents", "metadatas"]
        )

    docs = results["documents"][0]

    if not docs:
        return ""

    return "\n\n".join(docs)


# -------------------------
# Chat
# -------------------------
def chat(prompt):

    intent = detect_intent(prompt)
    mode = route(intent)

    if mode == "direct":
        return "Hello! What can I help you with ? You can ask me question about plants."
    
    topic = detect_topic(prompt)
    if topic == "unknown":
        context = retrieve(prompt)
    else:
        context = retrieve(prompt, topic)

    print("=== CONTEXT START ===")
    print(topic)
    print(context)
    print("=== CONTEXT END ===")

    if not context:
        return "Don't have enough information in knowledge base."

    memory = ""

    memory = "\n".join([
        f"User: {h['user']}\nAI: {h['ai']}"
        for h in history
    ])


    full_prompt = f"""
You are a strict secondary school plant biology teacher.

RULES:
1. Use ONLY the provided KNOWLEDGE.
2. Do NOT add extra topics not in KNOWLEDGE.
3. Do NOT expand beyond the question.
4. Keep answer short (max 3-5 lines).
5. If knowledge is insufficient, say "I cannot find this in my plant knowledge base."

KNOWLEDGE (use this only):
-------------------------
{context}
-------------------------

QUESTION:
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