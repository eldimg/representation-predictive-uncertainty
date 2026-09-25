import ast
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
tasks = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))

assert len(tasks) == 100, f"Expected 100 tasks, got {len(tasks)}"
counts = Counter(t["family"] for t in tasks)
assert len(counts) == 50, f"Expected 50 families, got {len(counts)}"
assert set(counts.values()) == {2}, f"Every family must contain exactly two tasks: {counts}"

ids = [t["id"] for t in tasks]
assert len(ids) == len(set(ids)), "Duplicate task IDs"

python_count = 0
nl_count = 0

for task in tasks:
    assert len(task["nl"]) == 3
    assert len(task["python"]) == 3
    assert len(set(task["nl"])) == 3, f"Duplicate NL variants: {task['id']}"
    assert len(set(task["python"])) == 3, f"Duplicate Python variants: {task['id']}"
    for code in task["python"]:
        ast.parse(code)
        python_count += 1
    nl_count += len(task["nl"])

print("VALIDATION PASSED")
print(f"Tasks:               {len(tasks)}")
print(f"Families:            {len(counts)}")
print(f"NL references:       {nl_count}")
print(f"Python references:   {python_count}")
print(f"Total trajectories:  {nl_count + python_count}")
