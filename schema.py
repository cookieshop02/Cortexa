from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from uuid import UUID

#what client sends when creating a new episodic memory
class EpisodicMemoryCreate(BaseModel):
    user_id: str
    content: str
    event_timestamp: datetime
    source: Optional[str] = None

#what we send back to the client (includes db generated fields)
class EpisodicMemoryResponse(BaseModel):
    id: UUID
    user_id: str
    content: str
    event_timestamp: datetime
    created_at: datetime
    importance_category: Optional[str] = None
    importance_score: Optional[float] = None
    status: str

    class Config:
        from_attributes = True #allows Pydantic to read data from SQLAlchemy objects
