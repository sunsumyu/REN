import json

def check_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"File: {filename}")
    print(f"Total lines: {len(lines)}")
    if len(lines) >= 175:
        print("Lines 165 to 175:")
        for idx in range(164, min(175, len(lines))):
            try:
                data = json.loads(lines[idx])
                q = data.get("Q", "No Q key")
                print(f"  Line {idx+1}: {q[:60]}")
            except Exception as e:
                print(f"  Line {idx+1}: [JSON DECODE ERROR] {lines[idx][:60]} - {e}")
    else:
        print("File has fewer than 175 lines.")
    print("-" * 50)

check_file("medical_qa_dataset.jsonl")
check_file("medical_qa_dataset_raw.jsonl")
