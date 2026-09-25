import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks.json"
EXPECTED = "dcdbc196f7dbf394fbf043690c764c470dbbbf9ee333e344c1f62f320a72a164"

actual = hashlib.sha256(TASKS.read_bytes()).hexdigest()
tasks = json.loads(TASKS.read_text(encoding="utf-8"))
families = {t["family"] for t in tasks}
trajectory_count = sum(len(t["nl"]) + len(t["python"]) for t in tasks)

assert actual == EXPECTED, (EXPECTED, actual)
assert len(tasks) == 100
assert len(families) == 50
assert all(len(t["nl"]) == 3 and len(t["python"]) == 3 for t in tasks)
assert trajectory_count == 600

print("FROZEN STIMULI OK")
print("tasks:", len(tasks))
print("families:", len(families))
print("trajectories:", trajectory_count)
print("sha256:", actual)
