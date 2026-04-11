import sqlite3
import json
import os
from datetime import datetime

class FeedbackLogger:
    def __init__(self, db_path="data/feedback_loop.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_query TEXT,
                    system_prompt TEXT,
                    retrieved_context TEXT,
                    llm_response TEXT,
                    feedback_score INTEGER,  -- 1 for thumbs up, -1 for down
                    corrected_response TEXT, -- Optional user-provided correction
                    processed FOR BOOLEAN DEFAULT 0
                )
            """)
            conn.commit()

    def log_interaction(self, user_query, system_prompt, retrieved_context, llm_response):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO interactions (timestamp, user_query, system_prompt, retrieved_context, llm_response, feedback_score)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (datetime.utcnow().isoformat(), user_query, system_prompt, json.dumps(retrieved_context), llm_response))
            return cursor.lastrowid

    def update_feedback(self, interaction_id, feedback_score, corrected_response=None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE interactions 
                SET feedback_score = ?, corrected_response = ?
                WHERE id = ?
            """, (feedback_score, corrected_response, interaction_id))
            conn.commit()

    def export_finetuning_dataset(self, output_jsonl="data/finetuning_dataset.jsonl"):
        """Exports positive feedback and user corrections to a JSONL file in conversational SFT format."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Select good interactions or ones that the user corrected
            cursor.execute("""
                SELECT id, user_query, system_prompt, retrieved_context, llm_response, corrected_response 
                FROM interactions 
                WHERE (feedback_score = 1 OR corrected_response IS NOT NULL) 
                AND processed = 0
            """)
            
            rows = cursor.fetchall()
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                for row in rows:
                    interaction_id, query, sys_prompt, context, response, correction = row
                    
                    target_response = correction if correction else response
                    full_prompt = f"{sys_prompt}\n\nContext:\n{context}\n\nUser:\n{query}"
                    
                    # Store in typical ChatML/SFT format
                    example = {
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": f"Context: {context}\n\n{query}"},
                            {"role": "assistant", "content": target_response}
                        ]
                    }
                    f.write(json.dumps(example) + "\n")
                    
                    # Mark as processed
                    cursor.execute("UPDATE interactions SET processed = 1 WHERE id = ?", (interaction_id,))
            
            conn.commit()
            return len(rows)
