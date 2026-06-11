import json
import os
import re

def extract_entity_name(source: str, context: str) -> str:
    if "实体库:" in source:
        m = re.search(r'实体库:([^》\s]+)', source)
        if m:
            return m.group(1)
    elif "图谱关系:" in source:
        m = re.search(r'图谱关系:([^》\s]+)', source)
        if m:
            parts = m.group(1).split("-")
            return parts[0] if parts else m.group(1)
    elif "常见临床疾病与合理用药诊疗路径-" in source:
        m = re.search(r'常见临床疾病与合理用药诊疗路径-([^》\s]+)', source)
        if m:
            return m.group(1)
            
    if context.startswith("概念定义:"):
        m = re.search(r'概念定义:\s*([^\s(（]+)', context)
        if m:
            return m.group(1)
    elif context.startswith("知识关联:"):
        m = re.search(r'【([^】]+)】', context)
        if m:
            return m.group(1)
            
    return ""

def main():
    dataset_path = "medical_qa_dataset.jsonl"
    if not os.path.exists(dataset_path):
        print("Dataset not found")
        return
        
    print("Analyzing expected facts from medical_qa_dataset.jsonl...")
    all_expected = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx > 322:
                break
            try:
                data = json.loads(line.strip())
                evidence_contract = data.get("evidence_contract", {})
                facts = evidence_contract.get("facts", []) if isinstance(evidence_contract, dict) else []
                
                for fact in facts:
                    source = fact.get("source", "")
                    context_preview = fact.get("context_preview", "")
                    if source and not any(x in source for x in ["PubMed", "在线公开", "抓取异常"]):
                        entity_name = extract_entity_name(source, context_preview)
                        all_expected.append({
                            "line": idx,
                            "query": data.get("Q", ""),
                            "entity_name": entity_name,
                            "source": source,
                            "context_preview": context_preview
                        })
            except Exception as e:
                pass
                
    # Group by source/entity
    unique_expected = {}
    for item in all_expected:
        key = (item["entity_name"], item["source"])
        if key not in unique_expected:
            unique_expected[key] = item
            
    print(f"Total unique expected facts: {len(unique_expected)}")
    print("\n--- ALL EXPECTED FACTS LIST ---")
    for i, (key, val) in enumerate(unique_expected.items(), 1):
        print(f"[{i}] Entity: '{val['entity_name']}' | Source: '{val['source']}'")
        print(f"    Preview: {val['context_preview'][:150]}")

if __name__ == "__main__":
    main()
