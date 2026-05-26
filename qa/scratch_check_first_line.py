import json

file_path = r"d:\REN\qa\medical_qa_dataset.jsonl"
with open(file_path, 'r', encoding='utf-8') as f:
    line = f.readline()
    data = json.loads(line)
    
print("Question:", data["Q"])
print("Number of planners:", len(data["planners"]))
for i, p in enumerate(data["planners"]):
    print(f"\n--- Planner {i+1}: {p['planner']} ---")
    ans = p['answer']
    # split into think and answer body
    if "<think>" in ans and "</think>" in ans:
        think_part = ans[ans.index("<think>")+7:ans.index("</think>")]
        body_part = ans[ans.index("</think>")+8:]
        print("Think snippet:")
        print("\n".join(think_part.strip().split("\n")[:10]))
        print("...")
        print("Answer snippet:")
        print("\n".join(body_part.strip().split("\n")[:10]))
        print("...")
    else:
        print("No think block found.")
        print(ans[:200])
