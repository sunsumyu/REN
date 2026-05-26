import json
import re
import os

from clean_dataset import clean_think_text

dataset_path = r"d:\REN\qa\medical_qa_dataset.jsonl"
output_artifact = r"C:\Users\cf\.gemini\antigravity-ide\brain\fc4ba524-383b-402d-b883-f283f268f581\removed_content.md"

if not os.path.exists(dataset_path):
    print("Dataset not found!")
    exit(1)

removed_details = []

with open(dataset_path, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            q = data.get("Q", "")
            planners = data.get("planners", [])
            
            for p_idx, p in enumerate(planners):
                facet = p.get("planner", "")
                raw_answer = p.get("answer", "")
                
                think_match = re.match(r"^\s*<think>([\s\S]*?)</think>", raw_answer)
                if think_match:
                    think_content = think_match.group(1).strip()
                    cleaned_think = clean_think_text(think_content)
                    
                    # 找出被剔除的行
                    before_lines = [l.strip() for l in think_content.splitlines()]
                    after_lines = [l.strip() for l in cleaned_think.splitlines()]
                    
                    removed_lines = []
                    for bl in before_lines:
                        if bl and bl not in after_lines:
                            removed_lines.append(bl)
                            
                    if removed_lines:
                        removed_details.append({
                            "row": line_idx + 1,
                            "question": q,
                            "facet": facet,
                            "removed": removed_lines
                        })
        except Exception as e:
            print(f"Error: {e}")

# 写入 markdown 报告
with open(output_artifact, 'w', encoding='utf-8') as out:
    out.write("# 🧹 废话清洗与元指令剔除明细报告\n\n")
    out.write("本报告展示了在 `clean_dataset.py` 中对思维链进行净化时，**实际被过滤掉的全部工程噪音和大模型元谈话（元指令）**。\n\n")
    
    for item in removed_details:
        out.write(f"## 📌 第 {item['row']} 行: {item['question']} (切面: **{item['facet']}**)\n\n")
        out.write("### ❌ 过滤掉的废话明细：\n")
        for line in item['removed']:
            out.write(f"- `{line}`\n")
        out.write("\n---\n\n")

print(f"Successfully generated removed content report to {output_artifact}")
