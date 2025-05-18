import csv
import random
from pathlib import Path
from datetime import datetime, timedelta

# הגדרות
NUM_SCRIPTS = 15
RUNS_PER_SCRIPT = 10
OUTPUT_FILE = "csv_reports/all_runs.csv"

TRIGGERED_BY_OPTIONS = ["user", "scheduler", "pipeline"]
BUILD_IDS = list(range(1001, 1006))

# יצירת תיקיית הפלט
Path("csv_reports").mkdir(parents=True, exist_ok=True)

# זמן ריצה בסיסי לכל תסריט (ממוצע "בריא")
script_base_times = {
    f"test_script_{i:02d}": random.uniform(5.0, 20.0)
    for i in range(1, NUM_SCRIPTS + 1)
}

# התחלת זמן דמי – נעשה בו אינקרמנט לכל ריצה
base_time = datetime.now() - timedelta(days=1)

# הכנת הדאטה
all_rows = []
run_counter = 1

for script_name, base_time_script in script_base_times.items():
    for i in range(RUNS_PER_SCRIPT):
        # סימולציה של זמן ריצה
        exec_time = round(max(random.normalvariate(base_time_script, 2.5), 0.1), 2)

        # לוגיקה לחיזוי סטטוס לפי חריגה
        if exec_time > base_time_script + 3.0:
            # חריגה ניכרת → סבירות גבוהה לכישלון
            status_weights = [0.3, 0.6, 0.1]
        elif exec_time < base_time_script - 2.0:
            # קצר מהרגיל → ייתכן שדולג
            status_weights = [0.3, 0.2, 0.5]
        else:
            # סביב הממוצע
            status_weights = [0.7, 0.2, 0.1]

        status = random.choices(["PASSED", "FAILED", "SKIPPED"], weights=status_weights, k=1)[0]

        # אם דולג – זמן ריצה 0
        if status == "SKIPPED":
            exec_time = 0.0

        row = {
            "run_id": f"run_{run_counter:04d}",
            "script_name": script_name,
            "execution_time": exec_time,
            "status": status,
            "timestamp": (base_time + timedelta(minutes=run_counter)).isoformat(),
            "build_id": random.choice(BUILD_IDS),
            "triggered_by": random.choice(TRIGGERED_BY_OPTIONS)
        }

        all_rows.append(row)
        run_counter += 1

# כתיבת קובץ CSV
with open(OUTPUT_FILE, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)

print(f"✅ נוצר קובץ CSV עם {len(all_rows)} רשומות ב: {OUTPUT_FILE}")
