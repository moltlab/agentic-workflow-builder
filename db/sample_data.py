from db.config import SessionLocal
from db.crud.transaction import create_transaction
from db.crud.agent import create_agent
import uuid
from datetime import datetime

# Example: populate some sample transaction and agent data
def populate_sample_data():
    db = SessionLocal()
    try:
        txn_data = {
            "id": uuid.uuid4(),
            "user_id": "test_user",
            "task_description": "Generate a summary",
            "status": "completed",
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "total_duration_ms": 1500,
            "input_data": {"prompt": "Summarize this document"},
            "final_output": {"summary": "It is about project roadmap."},
            "session_id": None
        }

        agent_data = {
            "id": uuid.uuid4(),
            "name": "summary_agent",
            "type": "generator",
            "description": "Generates text summaries",
            "llm_used": "gpt-4",
            "prompt_template": "Summarize the following: {{input}}",
            "config": {"temperature": 0.7}
        }

        create_transaction(db, txn_data)
        create_agent(db, agent_data)

        print("✅ Sample data populated successfully.")
    finally:
        db.close()

if __name__ == "__main__":
    populate_sample_data()
