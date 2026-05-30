import json

def inspect_line_59():
    print("=== RAW LINE 59 ===")
    with open('d:/REN/qa/medical_qa_dataset_raw.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) >= 59:
        line_data = json.loads(lines[58]) # 59th line (0-indexed 58)
        print("Question:", line_data.get("Q"))
        planners = line_data.get("planners", [])
        print(f"Number of planners: {len(planners)}")
        for idx, p in enumerate(planners):
            planner_name = p.get("planner")
            answer = p.get("answer", "")
            # Find <think> and </think>
            think_content = ""
            if "<think>" in answer and "</think>" in answer:
                think_content = answer.split("<think>")[1].split("</think>")[0]
            print(f"Planner {idx} ({planner_name}):")
            print(f"  Think length: {len(think_content)}")
            # Check for structural words
            for word in ["首先", "其次", "第三", "第四", "一、", "二、", "三、", "四、"]:
                count = think_content.count(word)
                if count > 0:
                    print(f"    Contains '{word}': {count} times")
    else:
        print("Raw dataset has fewer than 59 lines")

    print("\n=== PURIFIED LINE 59 ===")
    with open('d:/REN/qa/medical_qa_dataset.jsonl', 'r', encoding='utf-8') as f:
        p_lines = f.readlines()
    if len(p_lines) >= 59:
        p_line_data = json.loads(p_lines[58])
        print("Question:", p_line_data.get("Q"))
        # Check keys
        print("Keys in purified data:", list(p_line_data.keys()))
        if "purified_cot" in p_line_data:
            cot = p_line_data["purified_cot"]
            print("Purified CoT Length:", len(cot))
            for word in ["首先", "其次", "第三", "第四", "一、", "二、", "三、", "四、"]:
                count = cot.count(word)
                if count > 0:
                    print(f"    Contains '{word}': {count} times")
            # Print a snippet of purified_cot to see the structural words
            print("\n--- Purified CoT Snippet ---")
            print(cot[:2000])
        elif "think" in p_line_data:
            think = p_line_data["think"]
            print("Think Length:", len(think))
            for word in ["首先", "其次", "第三", "第四", "一、", "二、", "三、", "四、"]:
                count = think.count(word)
                if count > 0:
                    print(f"    Contains '{word}': {count} times")
            print("\n--- Think Snippet ---")
            print(think[:2000])
    else:
        print("Purified dataset has fewer than 59 lines")

if __name__ == "__main__":
    inspect_line_59()
