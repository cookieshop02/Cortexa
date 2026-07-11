from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timezone

import dateparser
import dateparser.search

from episodic.database import get_db
from episodic.models import EpisodicMemory
from episodic.schema import EpisodicMemoryCreate, EpisodicMemoryResponse
from classifier import classify_text, client  # reusing the same Groq client
from embeddings import generate_embedding

app = FastAPI(title="Memory System")


# ---------- Basic CRUD (used for manual testing) ----------

@app.post("/memories/episodic", response_model=EpisodicMemoryResponse)
def create_episodic_memory(memory: EpisodicMemoryCreate, db: Session = Depends(get_db)):
    new_memory = EpisodicMemory(
        user_id=memory.user_id,
        content=memory.content,
        event_timestamp=memory.event_timestamp,
        source=memory.source
    )
    db.add(new_memory)
    db.commit()
    db.refresh(new_memory)
    return new_memory


@app.get("/memories/episodic", response_model=List[EpisodicMemoryResponse])
def get_episodic_memories(user_id: str, db: Session = Depends(get_db)):
    memories = db.query(EpisodicMemory).filter(EpisodicMemory.user_id == user_id).all()
    return memories


# ---------- Ingestion pipeline ----------

@app.post("/memories/ingest")
def ingest_memory(user_id: str, text: str, source: str = None, db: Session = Depends(get_db)):
    try:
        units = classify_text(text)
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}

    stored = []
    skipped = []

    for unit in units:
        if unit["type"] == "episodic":
            try:
                embedding_vector = generate_embedding(unit["content"])
            except Exception as e:
                skipped.append({"content": unit["content"], "reason": f"embedding failed: {str(e)}"})
                continue

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
            stored.append(unit["content"])
        else:
            skipped.append(unit)

    db.commit()

    return {"stored_episodic": stored, "skipped": skipped}


# ---------- Internal helper functions (NOT exposed as endpoints) ----------

def _has_time_reference(query: str) -> tuple[bool, Optional[str], Optional[datetime], Optional[datetime]]:
    result = dateparser.search.search_dates(query)
    if not result:
        return False, None, None, None

    if len(result) == 1:
        # Only one date mentioned — treat it as start, end = now
        phrase, date = result[0]
        return True, phrase, date, datetime.now(timezone.utc)

    # Two or more dates mentioned — treat first as start, last as end
    start_phrase, start_date = result[0]
    end_phrase, end_date = result[-1]
    full_phrase = f"{start_phrase} {end_phrase}"  # rough combined phrase to strip from query
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


def _pure_similarity_search(user_id: str, query_embedding, limit: int, db: Session):
    results = (
        db.query(EpisodicMemory)
        .filter(EpisodicMemory.user_id == user_id)
        .order_by(EpisodicMemory.embedding.cosine_distance(query_embedding))
        .limit(limit)
        .all()
    )
    return _update_access_tracking(results, db)


def _time_filtered_similarity_search(user_id: str, query_embedding, start, end, limit: int, db: Session):
    results = (
        db.query(EpisodicMemory)
        .filter(
            EpisodicMemory.user_id == user_id,
            EpisodicMemory.event_timestamp >= start,
            EpisodicMemory.event_timestamp <= end,
        )
        .order_by(EpisodicMemory.embedding.cosine_distance(query_embedding))
        .limit(limit)
        .all()
    )
    return _update_access_tracking(results, db)


def _pure_time_search(user_id: str, start, end, limit: int, db: Session):
    results = (
        db.query(EpisodicMemory)
        .filter(
            EpisodicMemory.user_id == user_id,
            EpisodicMemory.event_timestamp >= start,
            EpisodicMemory.event_timestamp <= end,
        )
        .order_by(EpisodicMemory.event_timestamp.desc())
        .limit(limit)
        .all()
    )
    return _update_access_tracking(results, db)


# ---------- Single public search endpoint ----------

@app.get("/memories/episodic/search")
def search_memories(user_id: str, query: str, limit: int = 5, db: Session = Depends(get_db)):
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