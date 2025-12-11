import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.core.config import settings
from backend.api.routes import tasks, projects, qa, export, auth, feedback
from backend.core.database import create_db_and_tables, engine
from sqlmodel import Session, select
from backend.core.models import User
from backend.core.security import get_password_hash

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Autonomous Annotation Agent API"
)

def create_initial_data():
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == "admin@cortex.ai")).first()
        if not user:
            user = User(
                email="admin@cortex.ai",
                hashed_password=get_password_hash("admin"),
                full_name="Admin User",
                is_superuser=True,
            )
            session.add(user)
            session.commit()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    create_initial_data()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
app.mount("/data", StaticFiles(directory=settings.DATA_DIR), name="data")

# Include Routes
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(tasks.router, prefix="/api", tags=["tasks"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(qa.router, prefix="/api/qa", tags=["qa"])
app.include_router(export.router, prefix="/api", tags=["export"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["feedback", "learning"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
