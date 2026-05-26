import json
import os

input_file = r"d:\REN\qa\medical_qa_dataset.jsonl"
output_file = r"d:\REN\qa\think_blocks_summary.txt"

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit(1)

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(output_file, 'w', encoding='utf-8') as out:
    out.write(f"Total rows in dataset: {len(lines)}\n\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            q = data.get("Q", "")
            out.write(f"==================================================\n")
            out.write(f"ROW {i+1}: {q}\n")
            out.write(f"==================================================\n\n")
            planners = data.get("planners", [])
            for j, p in enumerate(planners):
                facet = p.get("planner", "")
                raw_answer = p.get("answer", "")
                import re
                think_match = re.match(r"^<think>([\s\S]*?)</think>", raw_answer)
                out.write(f"--- Planner {j+1}: {facet} ---\n")
                if think_match:
                    think_content = think_match.group(1).strip()
                    # Just print first 800 characters to keep it readable, or print whole if needed
                    out.write(think_content)
                else:
                    out.write("[No <think> block found]")
                out.write("\n\n")
        except Exception as e:
            out.write(f"Error parsing row {i+1}: {e}\n\n")

print(f"Done! Saved summary of think blocks to {output_file}")
