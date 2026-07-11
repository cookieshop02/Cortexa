from sqlalchemy import Column, String, Text, DateTime, Float, Integer
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid
from datetime import datetime, timezone
from episodic.database import Base

class EpisodicMemory(Base):
    __tablename__="episodic_memories"

    id = Column(UUID(as_uuid = True), primary_key = True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index = True)
    content = Column(Text, nullable=False) #text for long content
    event_timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda:datetime.now(timezone.utc)) #lambda:Toh ye time file load hote hi ek baar calculate ho jaata (jab Python file padhta hai), aur hamesha wahi purana time use hota 
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    access_count = Column(Integer, default = 0)
    source = Column(String, nullable=True) #string for short names/texts
    type_confidence = Column(Float, nullable=True)
    importance_category = Column(String, nullable=True)
    importance_score = Column(Float, nullable=True)
    status = Column(String, default="active")
    embedding = Column(Vector(dim=384), nullable=False)
