import json
import collections

categories = collections.Counter()
total = 0
has_common_drug = 0
has_recommand_drug = 0

with open("medical.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            if line.startswith("["): line = line[1:]
            if line.endswith("],") or line.endswith("]"): line = line[:-2] if line.endswith("],") else line[:-1]
            if line.endswith(","): line = line[:-1]
            
            data = json.loads(line)
            total += 1
            cat = data.get("category", "")
            if isinstance(cat, list):
                # 取大类，比如 "疾病百科"
                categories[cat[0] if cat else "Unknown"] += 1
            else:
                categories[str(cat)] += 1
                
            if data.get("common_drug"):
                has_common_drug += 1
            if data.get("recommand_drug"):
                has_recommand_drug += 1
        except Exception as e:
            pass

print(f"Total entities: {total}")
print("Categories:")
for k, v in categories.items():
    print(f"  - {k}: {v} ({v/total*100:.2f}%)")
print(f"Entities with common_drug: {has_common_drug} ({has_common_drug/total*100:.2f}%)")
print(f"Entities with recommand_drug: {has_recommand_drug} ({has_recommand_drug/total*100:.2f}%)")
