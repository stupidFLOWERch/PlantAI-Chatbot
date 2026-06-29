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
    greeting_words = ["hi", "hello", "hey", "good morning", "good afternoon", "good night"]
    thanks_words = ["thank you", "thanks", "tq", "thx", "ty", "appreciate it", "much appreciated"]
    is_greeting = any(re.search(rf"\b{word}\b", text) for word in greeting_words)
    is_thanks = any(re.search(rf"\b{word}\b", text) for word in thanks_words)
    is_question = (
        "?" in text or
        any(word in text for word in question_words)
    )

    if is_question:
        return "question"
    if is_greeting:
        return "greeting"
    if is_thanks:
        return "thanks"

    return "question"


# -------------------------
# Follow-up Detection
# -------------------------
def is_follow_up(prompt, history):
    # if no history question, no way is follow up
    if not history:
        return False
    
    prompt_lower = prompt.lower().strip()
    
    # 1. detect words that always appear in follow up question
    follow_up_patterns = [
        r"how about",
        r"what about",
        r"^and",
        r"^also",
        r"^then",
        r"^so",
        r"for .+",
    ]
    
    for pattern in follow_up_patterns:
        if re.search(pattern, prompt_lower, re.IGNORECASE):
            return True
    
    # 2. if question less than 5 words, assume as follow up question
    if len(prompt_lower.split()) < 5:
        return True
    
    # get all word with length > 3 in last and current question
    last_user = history[-1].get("user", "").lower()
    last_words = set(re.findall(r'\b[a-z]{3,}\b', last_user))
    current_words = set(re.findall(r'\b[a-z]{3,}\b', prompt_lower))
    # 3. if current question got same repeated word as last question, assume as follow up
    if last_words & current_words:
        return True
    
    return False


# Route the request based on detected intent.
def route(intent):
    if intent == "greeting":
        return "direct"
    if intent == "question":
        return "rag"
    if intent == "thanks":
        return "thanks"
    return "fallback"


# -------------------------
# Retrieve Knowledge (Dual-Query)
# -------------------------
def retrieve_with_context(current_prompt, history_prompt=None):
    """
    双查询检索：
    1. 用当前问题检索 15 条
    2. 如果有历史上下文，用历史上下文检索 10 条
    3. 合并去重后，用 Reranker 精排取 Top 3
    """
    all_docs = []
    all_metadatas = []
    
    # 1. semantic search for current user prompt
    print(f"=== SEARCHING WITH CURRENT: {current_prompt[:50]}... ===")
    results_current = collection.query(
        query_texts=[current_prompt],
        n_results=15,
        include=["documents", "metadatas"]
    )
    
    docs_current = results_current["documents"][0]
    metas_current = results_current["metadatas"][0] if results_current["metadatas"] else []
    
    print(f"  Current search returned {len(docs_current)} docs")
    
    # add searched result to all_docs and all_metadatas
    for i, doc in enumerate(docs_current):
        if doc not in all_docs:
            all_docs.append(doc)
            all_metadatas.append(metas_current[i] if i < len(metas_current) else {})
    
    # 2. if history_prompt != none, do semantic search with history prompt
    if history_prompt:
        print(f"=== SEARCHING WITH HISTORY: {history_prompt[:50]}... ===")
        results_history = collection.query(
            query_texts=[history_prompt],
            n_results=10,
            include=["documents", "metadatas"]
        )
        
        docs_history = results_history["documents"][0]
        metas_history = results_history["metadatas"][0] if results_history["metadatas"] else []
        
        print(f"  History search returned {len(docs_history)} docs")
        
        # add searched result to all_docs and all_metadatas, avoid saving repeated results
        for i, doc in enumerate(docs_history):
            if doc not in all_docs:
                all_docs.append(doc)
                all_metadatas.append(metas_history[i] if i < len(metas_history) else {})
    
    print(f"=== COMBINED: {len(all_docs)} unique docs ===")
    
    # 3. if results too few, skip rerank
    if len(all_docs) < 3:
        print("=== Too few docs, skipping rerank ===")
        return format_docs(all_docs, all_metadatas)
    
    # 4. Rerank context
    print(f"=== RERANKING {len(all_docs)} DOCS ===")
    pairs = [[current_prompt, doc] for doc in all_docs]
    scores = reranker.compute_score(pairs, normalize=True)
    
    # 5. get the top 3 result
    sorted_pairs = sorted(zip(all_docs, all_metadatas, scores), key=lambda x: x[2], reverse=True)
    
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

    if mode == "thanks":
        return "You're welcome! Feel free to ask if you have more questions."

    # detect if is_follow_up, if yes, get last user prompt
    if history and is_follow_up(prompt, history):
        history_prompt = history[-1].get("user", "")
        print(f"=== FOLLOW-UP DETECTED, using history for retrieval ===")
    else:
        history_prompt = None
        print(f"=== NEW TOPIC, no history for retrieval ===")
    
    # Dual-Query Retrieval
    context = retrieve_with_context(prompt, history_prompt)

    print("=== CONTEXT START ===")
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
8. **If the user does not specify a section, provide the general answer.**
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