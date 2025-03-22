{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b395f29c-fa34-494c-9125-26e1885304bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🎉✅ Database setup complete! Analytics stored in 'hotel_analytics.db'.\n"
     ]
    }
   ],
   "source": [
    "import sqlite3\n",
    "import pandas as pd\n",
    "import os\n",
    "import numpy as np\n",
    "\n",
    "# ✅ Ensure the data folder exists\n",
    "if not os.path.exists(\"data\"):\n",
    "    os.makedirs(\"data\")\n",
    "\n",
    "# ✅ Load processed data\n",
    "file_path = \"data/processed_data.csv\"\n",
    "\n",
    "if not os.path.exists(file_path):\n",
    "    raise FileNotFoundError(\"❌ processed_data.csv not found! Run preprocess.py first.\")\n",
    "\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "# ✅ Compute key analytics\n",
    "total_revenue = df[\"revenue\"].sum()\n",
    "cancellation_rate = df[\"is_canceled\"].mean() * 100  # Convert to percentage\n",
    "top_countries = df[\"country\"].value_counts().head(10).to_dict()\n",
    "lead_time_distribution = df[\"lead_time\"].describe().to_dict() if \"lead_time\" in df.columns else {}\n",
    "\n",
    "# ✅ Connect to SQLite database\n",
    "db_path = \"hotel_analytics.db\"\n",
    "conn = sqlite3.connect(db_path)\n",
    "cursor = conn.cursor()\n",
    "\n",
    "# ✅ Create analytics table\n",
    "cursor.execute(\"\"\"\n",
    "CREATE TABLE IF NOT EXISTS analytics (\n",
    "    metric TEXT PRIMARY KEY,\n",
    "    value TEXT\n",
    ")\n",
    "\"\"\")\n",
    "\n",
    "# ✅ Insert computed values into the table\n",
    "analytics_data = [\n",
    "    (\"total_revenue\", str(total_revenue)),\n",
    "    (\"cancellation_rate\", str(cancellation_rate)),\n",
    "    (\"top_countries\", str(top_countries)),\n",
    "    (\"lead_time_distribution\", str(lead_time_distribution))\n",
    "]\n",
    "\n",
    "cursor.executemany(\"INSERT OR REPLACE INTO analytics (metric, value) VALUES (?, ?)\", analytics_data)\n",
    "\n",
    "# ✅ Close connection\n",
    "conn.close()\n",
    "\n",
    "print(\"🎉✅ Database setup complete! Analytics stored in 'hotel_analytics.db'.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a7a8dba-5235-4e07-b48a-6f1980efd939",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
