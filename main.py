from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from models import EpisodicMemory
from schema import EpisodicMemoryCreate, EpisodicMemoryResponse
from classifier import classify_text
from datetime import datetime, timezone

app = FastAPI(title = "Memory System")

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

@app.post("/memories/ingest")
def ingest_memory(user_id: str, text: str, source: str = None, db: Session = Depends(get_db)):
    units = classify_text(text)

    stored=[]
    skipped=[]

    for unit in units:
        if unit["type"] == "episodic":
            new_memory = EpisodicMemory(
                user_id=user_id,
                content=unit["content"],
                event_timestamp=datetime.now(timezone.utc),
                source=source,
                type_confidence = unit["confidence"],
                importance_category = unit["importance_category"],
                importance_score = unit["importance_score"]
            )
            db.add(new_memory)
            stored.append(new_memory)
        else:
            skipped.append(unit)

    db.commit()

    return {"stored_episodic": stored, "skipped": skipped}