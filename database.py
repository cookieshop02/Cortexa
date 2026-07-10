import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    This function will give a db session to each api request 
    and will make sure it is closed properly afterwards.
    """
    db = SessionLocal()
    try:
        yield db #give db but it will be stopped here until request not over
    finally:
        db.close() #after request over then will be closed finally here

    #this function will be used by fastapi for db use and it is called dependency inject as fastapi dont make it itself but use it from here
    

