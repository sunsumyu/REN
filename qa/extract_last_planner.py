import json

file_path = r"d:\REN\qa\medical_qa_dataset.jsonl"
out_path = r"d:\REN\qa\last_planner_output.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    line = f.readline()
    data = json.loads(line)

# Let's find the '长期预后影响' planner
last_planner = None
for p in data["planners"]:
    if p["planner"] == "长期预后影响":
        last_planner = p
        break

if last_planner:
    with open(out_path, 'w', encoding='utf-8') as out_f:
        out_f.write(f"Planner Name: {last_planner['planner']}\n")
        out_f.write("Answer:\n")
        out_f.write(last_planner['answer'])
    print("Successfully wrote output to last_planner_output.txt")
else:
    print("Could not find the '长期预后影响' planner in row 1")
