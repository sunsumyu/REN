import json
from pathlib import Path

failures_path = Path(r"d:\REN\qa\logs\purification_failures.jsonl")
if not failures_path.exists():
    print("Failure file not found!")
    exit(1)

with open(failures_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            ln = event.get("original_line_number")
            reason = event.get("reason")
            data = event.get("data", {})
            q = data.get("Q")
            planners = data.get("planners", [])
            print(f"=== Line {ln} ===")
            print(f"Q: {q}")
            print(f"Reason: {reason}")
            for idx, p in enumerate(planners):
                planner_name = p.get("planner")
                # Check if this planner has audit info or if we can extract scores
                # Note: in purification_failures.jsonl, the planners list might contain the original or the failed attempt.
                # Let's see what is inside the planner object
                print(f"  Planner {idx+1}: {planner_name}")
                # We can also check if we have audit logs for this line
        except Exception as e:
            print(f"Error parsing line: {e}")
