import json

def get_line_data(file_path, line_number):
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if idx == line_number:
                return json.loads(line)
    return None

file_before = "d:/REN/qa/medical_qa_dataset_first_purified.jsonl"
file_after = "d:/REN/qa/medical_qa_dataset.jsonl"

for line_num in [218, 239]:
    print(f"\n=================== Line {line_num} ===================")
    data_before = get_line_data(file_before, line_num)
    data_after = get_line_data(file_after, line_num)
    
    if not data_before or not data_after:
        print(f"Error: Could not retrieve data for line {line_num}")
        continue
        
    print(f"Q: {data_before.get('Q')}")
    
    # Check planners
    for p_idx, (p_bef, p_aft) in enumerate(zip(data_before.get("planners", []), data_after.get("planners", []))):
        ans_bef = p_bef.get("answer", "")
        ans_aft = p_aft.get("answer", "")
        if ans_bef != ans_aft:
            print(f"\n--- Planner {p_idx} Answer Changed ---")
            print(f"BEFORE:\n{ans_bef[:500]}...\n")
            print(f"AFTER:\n{ans_aft[:500]}...\n")
            
    # Check summary
    sum_bef = data_before.get("summary", "")
    sum_aft = data_after.get("summary", "")
    if sum_bef != sum_aft:
        print(f"\n--- Summary Changed ---")
        print(f"BEFORE:\n{sum_bef}\n")
        print(f"AFTER:\n{sum_aft}\n")
