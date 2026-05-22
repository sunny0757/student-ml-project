import sqlite3

conn = sqlite3.connect("prediction.db", check_same_thread=False)

cursor = conn.cursor()

cursor.execute("""

    CREATE TABLE IF NOT EXISTS predictions (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        gender TEXT,

        math REAL,

        science REAL,

        prediction TEXT

    )

    """)

conn.commit()
