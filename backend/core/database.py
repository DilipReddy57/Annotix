from sqlmodel import SQLModel, create_engine, Session
from backend.core.config import settings
import os

# Use absolute path to database in storage directory
database_path = os.path.join(settings.STORAGE_DIR, "database.db")
sqlite_url = f"sqlite:///{database_path}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=True, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
