from database import Base, engine
from models import EpisodicMemory

# This looks at all models that inherit from Base,
# and creates their tables in the actual Postgres database
Base.metadata.create_all(bind=engine)

print("Tables created successfully.")