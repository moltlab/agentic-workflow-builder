from db.base import Base
from db.config import engine
from db import models  # Ensures all models are registered

# Create all tables in the database
if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully.")