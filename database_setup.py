import sqlite3
import pandas as pd
import os
import numpy as np

def setup_database():
    # Ensure the data folder exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Load processed data
    file_path = "data/processed_data.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError("❌ processed_data.csv not found! Run preprocess.py first.")

    df = pd.read_csv(file_path)

    # Compute key analytics
    total_revenue = df["revenue"].sum()
    cancellation_rate = df["is_canceled"].mean() * 100  # Convert to percentage
    top_countries = df["country"].value_counts().head(10).to_dict()
    lead_time_distribution = df["lead_time"].describe().to_dict() if "lead_time" in df.columns else {}

    # Connect to SQLite database
    db_path = "hotel_analytics.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create analytics table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analytics (
        metric TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    # Insert computed values into the table
    analytics_data = [
        ("total_revenue", str(total_revenue)),
        ("cancellation_rate", str(cancellation_rate)),
        ("top_countries", str(top_countries)),
        ("lead_time_distribution", str(lead_time_distribution))
    ]

    cursor.executemany("INSERT OR REPLACE INTO analytics (metric, value) VALUES (?, ?)", analytics_data)

    # Close connection
    conn.close()

    print("🎉✅ Database setup complete! Analytics stored in 'hotel_analytics.db'.")

if __name__ == "__main__":
    setup_database()