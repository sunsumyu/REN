import json

file_path = r"d:\REN\qa\medical_qa_dataset.jsonl"
out_path = r"d:\REN\qa\huanyou_raw.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

found_data = None
for i, line in enumerate(lines):
    if not line.strip():
        continue
    try:
        data = json.loads(line)
        if "獾油" in data.get("Q", ""):
            found_data = (i + 1, data)
            break
    except Exception as e:
        print(f"Error parsing line {i+1}: {e}")

if found_data:
    line_num, data = found_data
    with open(out_path, 'w', encoding='utf-8') as out_f:
        out_f.write(f"Line Number in JSONL: {line_num}\n")
        out_f.write(f"Question: {data['Q']}\n")
        out_f.write(f"Number of Planners: {len(data.get('planners', []))}\n\n")
        for idx, p in enumerate(data.get("planners", [])):
            out_f.write(f"=== Planner {idx+1}: {p['planner']} ===\n")
            out_f.write(p['answer'])
            out_f.write("\n\n")
    print(f"Successfully extracted Huanyou Chaji row (Line {line_num}) to huanyou_raw.txt")
else:
    # Just write out all keys of all lines to see what is in there
    with open(out_path, 'w', encoding='utf-8') as out_f:
        out_f.write(f"Could not find Huanyou in any question. Total lines parsed: {len(lines)}\n")
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                out_f.write(f"Line {idx+1} Question: {data.get('Q', '')}\n")
            except Exception as e:
                out_f.write(f"Line {idx+1} Error: {e}\n")
