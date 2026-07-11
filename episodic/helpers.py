from datetime import datetime, timezone, timedelta
from episodic.models import EpisodicMemory
from datetime import timedelta
from sqlalchemy.orm import Session

def check_duplicate_episodic(user_id: str, content: str, db: Session, window_minutes: int = 5):
    cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    
    existing = (
        db.query(EpisodicMemory)
        .filter(
            EpisodicMemory.user_id == user_id,
            EpisodicMemory.content.ilike(content),
            EpisodicMemory.created_at >= cutoff_time
        )
        .first()
    )
    return existing