from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import os

import dateparser
import dateparser.search

from episodic.database import get_db
from episodic.models import EpisodicMemory
from episodic.schema import EpisodicMemoryCreate, EpisodicMemoryResponse
from episodic.helpers import check_duplicate_episodic

from classifier import classify_text, client
from embeddings import generate_embedding

from semantic.extractor import extract_triplet
from semantic.storage import store_triplet
from semantic.retrieval import search_semantic_memory

from response_formatter import format_chat_reply, format_episodic_result, format_semantic_result

app = FastAPI(title="Memory System")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ==================== SHARED STORAGE HELPERS (used by /ingest and /chat) ====================

def _store_episodic_unit(user_id: str, unit: dict, source: Optional[str], db: Session) -> Optional[str]:
    duplicate = check_duplicate_episodic(user_id, unit["content"], db)
    if duplicate:
        return None  # noop, nothing stored

    try:
        embedding_vector = generate_embedding(unit["content"])
    except Exception:
        return None

    new_memory = EpisodicMemory(
        user_id=user_id,
        content=unit["content"],
        event_timestamp=datetime.now(timezone.utc),
        source=source,
        type_confidence=unit["confidence"],
        importance_category=unit["importance_category"],
        importance_score=unit["importance_score"],
        embedding=embedding_vector
    )
    db.add(new_memory)
    return unit["content"]


def _store_semantic_unit(user_id: str, unit: dict, source: Optional[str]) -> Optional[dict]:
    try:
        triplet = extract_triplet(unit["content"])
        result = store_triplet(
            user_id=user_id,
            entity1=triplet["entity1"],
            relationship=triplet["relationship"],
            entity2=triplet["entity2"],
            confidence=triplet["confidence"],
            importance_category=triplet["importance_category"],
            importance_score=triplet["importance_score"],
            is_single_valued=triplet.get("is_single_valued", True),
            source=source
        )
        return {
            "fact": f"{triplet['entity1']} {triplet['relationship']} {triplet['entity2']}",
            "noop": result["noop"],
            "invalidated": result["invalidated_conflicts"]
        }
    except Exception:
        return None


# ==================== EPISODIC search internals (unchanged logic) ====================

def _has_time_reference(query: str):
    result = dateparser.search.search_dates(query)
    if not result:
        return False, None, None, None
    if len(result) == 1:
        phrase, date = result[0]
        return True, phrase, date, datetime.now(timezone.utc)
    start_phrase, start_date = result[0]
    end_phrase, end_date = result[-1]
    full_phrase = f"{start_phrase} {end_phrase}"
    return True, full_phrase, start_date, end_date


def _has_meaningful_topic(remaining_text: str) -> bool:
    if not remaining_text.strip():
        return False
    prompt = f"""Does the following text contain a specific topic/subject to search for, 
or is it just a generic request to show/list everything with no specific topic?

Text: "{remaining_text}"

Respond with ONLY one word: "SPECIFIC" or "GENERIC" """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip().upper()
    return "SPECIFIC" in answer


def _update_access_tracking(memories: list, db: Session):
    for memory in memories:
        memory.last_accessed_at = datetime.now(timezone.utc)
        memory.access_count = (memory.access_count or 0) + 1
    db.commit()
    return memories


def _pure_similarity_search(user_id: str, query_embedding, limit: int, db: Session, max_distance: float = 0.6):
    results_with_distance = (
        db.query(EpisodicMemory, EpisodicMemory.embedding.cosine_distance(query_embedding).label("distance"))
        .filter(EpisodicMemory.user_id == user_id)
        .order_by("distance")
        .limit(limit)
        .all()
    )
    filtered = [r[0] for r in results_with_distance if r[1] < max_distance]
    return _update_access_tracking(filtered, db)


def _time_filtered_similarity_search(user_id: str, query_embedding, start, end, limit: int, db: Session):
    results = (
        db.query(EpisodicMemory)
        .filter(EpisodicMemory.user_id == user_id, EpisodicMemory.event_timestamp >= start, EpisodicMemory.event_timestamp <= end)
        .order_by(EpisodicMemory.embedding.cosine_distance(query_embedding))
        .limit(limit)
        .all()
    )
    return _update_access_tracking(results, db)


def _pure_time_search(user_id: str, start, end, limit: int, db: Session):
    results = (
        db.query(EpisodicMemory)
        .filter(EpisodicMemory.user_id == user_id, EpisodicMemory.event_timestamp >= start, EpisodicMemory.event_timestamp <= end)
        .order_by(EpisodicMemory.event_timestamp.desc())
        .limit(limit)
        .all()
    )
    return _update_access_tracking(results, db)


def _search_episodic_core(user_id: str, query: str, db: Session, limit: int = 5):
    has_time, time_phrase, start, end = _has_time_reference(query)

    if has_time:
        search_content = query
        for phrase_part in time_phrase.split():
            search_content = search_content.replace(phrase_part, "")
        search_content = search_content.strip()

        start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end

        if _has_meaningful_topic(search_content):
            query_embedding = generate_embedding(search_content)
            return _time_filtered_similarity_search(user_id, query_embedding, start, end, limit, db)
        else:
            return _pure_time_search(user_id, start, end, limit, db)

    query_embedding = generate_embedding(query)
    return _pure_similarity_search(user_id, query_embedding, limit, db)


# ==================== UNIFIED CHAT ENDPOINT ====================

@app.post("/memories/chat")
def chat_with_memory(user_id: str, message: str, source: str = None, db: Session = Depends(get_db)):
    try:
        units = classify_text(message)
    except Exception as e:
        return {"reply": f"Sorry, I couldn't process that: {str(e)}"}

    query_units = [u for u in units if u["type"] == "query"]

    if len(query_units) > 1:
        return {"reply": "Please ask one question at a time — I noticed a few questions in there!"}

    stored_episodic = []
    stored_semantic = []

    for unit in units:
        if unit["type"] == "episodic":
            result = _store_episodic_unit(user_id, unit, source, db)
            if result:
                stored_episodic.append(result)
        elif unit["type"] == "semantic":
            result = _store_semantic_unit(user_id, unit, source)
            if result:
                stored_semantic.append(result)

    db.commit()

    retrieved_text = None
    if query_units:
        q = query_units[0]
        needs = q.get("needs", "both")
        query_text = q["content"]

        if needs == "episodic":
            results = _search_episodic_core(user_id, query_text, db)
            retrieved_text = format_episodic_result(results)

        elif needs == "semantic":
            results = search_semantic_memory(user_id, query_text)
            retrieved_text = format_semantic_result(results)

        else:  # both — run in parallel using threads (both are I/O-bound calls)
            with ThreadPoolExecutor() as executor:
                episodic_future = executor.submit(_search_episodic_core, user_id, query_text, db)
                semantic_future = executor.submit(search_semantic_memory, user_id, query_text)

                episodic_results = episodic_future.result()
                semantic_results = semantic_future.result()

            episodic_text = format_episodic_result(episodic_results)
            semantic_text = format_semantic_result(semantic_results)
            retrieved_text = episodic_text + "\n" + semantic_text

    reply = format_chat_reply(stored_episodic, stored_semantic, retrieved_text)
    return {"reply": reply}