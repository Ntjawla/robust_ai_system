import sqlite3

# Create SQLite DB and table if it doesn't exist
conn = sqlite3.connect("predictions.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS logs
             (id INTEGER PRIMARY KEY AUTOINCREMENT, review TEXT, sentiment TEXT, intent TEXT, summary TEXT)''')
conn.commit()

# Logging function
def log_prediction(review, sentiment, intent, summary):
    c.execute("INSERT INTO logs (review, sentiment, intent, summary) VALUES (?, ?, ?, ?)",
              (review, sentiment, intent, summary))
    conn.commit()
