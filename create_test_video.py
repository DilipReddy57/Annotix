from sqlmodel import Session, select
from backend.core.database import engine
from backend.core.models import Project, Video
import os
import cv2
import numpy as np

def create_test_video():
    with Session(engine) as session:
        # Get the first project or create one
        project = session.exec(select(Project)).first()
        if not project:
            print("No project found. Creating 'Demo Project'...")
            project = Project(name="Demo Project", description="Auto-generated for testing")
            session.add(project)
            session.commit()
            session.refresh(project)

        # Create a dummy video file
        project_dir = os.path.join("d:/project/data annotaion using sam3/backend/data/uploads", project.id) # Use abs path to be safe
        os.makedirs(project_dir, exist_ok=True)
        video_path = os.path.join(project_dir, "test_video.mp4")
        
        # Initialize Video Writer
        width, height = 640, 480
        fps = 30
        seconds = 3
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"Failed to open video writer for {video_path}")
            return
            
        print(f"Generating synthetic video at {video_path}...")
        
        # Bouncing Ball Logic
        x, y = width // 2, height // 2
        dx, dy = 5, 4
        radius = 30
        
        for _ in range(fps * seconds):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw Ball (Red)
            cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)
            
            # Simple text prompt target
            cv2.putText(frame, "Target", (x - 20, y - radius - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out.write(frame)
            
            # Update position
            x += dx
            y += dy
            
            if x - radius < 0 or x + radius > width: dx = -dx
            if y - radius < 0 or y + radius > height: dy = -dy

        out.release()
        
        # Check if video already exists in DB
        existing_video = session.exec(select(Video).where(Video.filename == "test_video.mp4", Video.project_id == project.id)).first()
        if not existing_video:
            video = Video(
                project_id=project.id,
                filename="test_video.mp4",
                status="completed"
            )
            session.add(video)
            session.commit()
            print(f"Created DB entry for test video in project {project.name}")
        else:
            print("Video already in DB.")

if __name__ == "__main__":
    create_test_video()
