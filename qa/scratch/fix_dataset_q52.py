import json
from pathlib import Path

def fix_dataset():
    dataset_path = Path("d:/REN/qa/medical_qa_dataset.jsonl")
    if not dataset_path.exists():
        print("Dataset not found!")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 第 52 行对应 Python 列表下标 51
    line_idx = 51
    line_str = lines[line_idx]
    
    try:
        data = json.loads(line_str)
        print(f"Original QA: {data.get('Q')}")
        print(f"Original Facets: {[p.get('planner') for p in data.get('planners', [])]}")
        
        # 1. 过滤掉有严重化学与逆转药理幻觉的 "分子机制" 视角
        filtered_planners = []
        for p in data.get("planners", []):
            if p.get("planner") == "分子机制":
                print("-> Found and removed hallucinated '分子机制' facet.")
                continue
            
            # 2. 找到 "临床指南" 视角并物理切除草稿尾巴 '，见？'
            if p.get("planner") == "临床指南":
                old_answer = p["answer"]
                if "，见？" in old_answer:
                    p["answer"] = old_answer.replace("，见？", "，且血液透析对其清除极其有限")
                    print("-> Successfully patched draft placeholder '，见？' in '临床指南' facet.")
                elif "见？" in old_answer:
                    p["answer"] = old_answer.replace("见？", "，且血液透析对其清除极其有限")
                    print("-> Successfully patched draft placeholder '见？' in '临床指南' facet.")
            
            filtered_planners.append(p)
            
        data["planners"] = filtered_planners
        
        # 将修改后的行重新打包为 JSON 字符串
        lines[line_idx] = json.dumps(data, ensure_ascii=False) + "\n"
        
        # 写回数据集文件
        with open(dataset_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        print("🎉 Successfully executed high-precision surgical dataset repair!")
        
    except Exception as e:
        print(f"Repair failed: {e}")

if __name__ == "__main__":
    fix_dataset()
