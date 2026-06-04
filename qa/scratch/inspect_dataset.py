import json

file_path = "d:/REN/qa/medical_qa_dataset.jsonl"
search_terms = ["替米沙坦", "大黄酚", "去甲替林", "特瑞普利单抗"]

with open(file_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        for term in search_terms:
            if term in line:
                print(f"Line {idx}: found '{term}'")
                try:
                    data = json.loads(line)
                    print(f"  Q: {data.get('Q')}")
                except Exception as e:
                    print(f"  Error parsing JSON: {e}")
