# database.py
import sqlite3

def initialize_database(db_path="processed_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_samples (
            id INTEGER PRIMARY KEY,
            text TEXT,
            classification TEXT,
            user_feedback TEXT,
            user_classification TEXT,
            mode TEXT
        )
    ''')
    conn.commit()
    conn.close()
