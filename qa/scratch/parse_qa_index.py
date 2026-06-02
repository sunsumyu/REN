import re
import json

file_path = r"d:\REN\qa\logs\purification_run_[123-164]_20260602_171235.md"

qa_list = []
current_qa = None

with open(file_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        # Match something like: ## 📌 [QA-1] (数据集第 123 行) | 主问题: `健肝乐颗粒的用法用量是什么？`
        match = re.match(r"^## 📌 \[QA-(\d+)\] \(([^)]+)\) \| 主问题: `(.*?)`", line)
        if match:
            if current_qa:
                current_qa["end_line"] = idx - 1
                qa_list.append(current_qa)
            
            current_qa = {
                "qa_index": int(match.group(1)),
                "dataset_info": match.group(2),
                "question": match.group(3),
                "start_line": idx,
                "end_line": None,
                "facets": []
            }
        
        # Match facet header: ### 🔍 视角 [1]: 临床视角: **疗程与疗效**
        if current_qa:
            facet_match = re.search(r"### 🔍 视角 \[\d+\]: (.*?视角): \*\*(.*?)\*\*", line)
            if facet_match:
                current_qa["facets"].append({
                    "facet_type": facet_match.group(1),
                    "facet_name": facet_match.group(2),
                    "line": idx
                })

if current_qa:
    current_qa["end_line"] = idx
    qa_list.append(current_qa)

# Save the index to a json file in scratch
output_path = r"d:\REN\qa\scratch\qa_index.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(qa_list, f, ensure_ascii=False, indent=2)

print(f"Successfully indexed {len(qa_list)} QAs. Index saved to {output_path}.")
for qa in qa_list[:5]:
    print(f"QA-{qa['qa_index']}: {qa['question']} (Lines {qa['start_line']}-{qa['end_line']}), Facets: {[f['facet_name'] for f in qa['facets']]}")
