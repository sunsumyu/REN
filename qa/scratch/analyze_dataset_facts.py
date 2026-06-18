# -*- coding: utf-8 -*-
import json
import os
import re

def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(workspace_dir, "medical_qa_dataset.jsonl")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    facts_count = 0
    local_facts_count = 0
    facts_with_entity_name = 0
    source_patterns = {}
    unique_sources = set()
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            try:
                data = json.loads(line.strip())
                evidence_contract = data.get("evidence_contract", {})
                facts = evidence_contract.get("facts", []) if isinstance(evidence_contract, dict) else []
                
                for fact in facts:
                    facts_count += 1
                    source = fact.get("source", "")
                    metadata = fact.get("metadata", {})
                    entity_name = metadata.get("entity_name", "")
                    
                    if entity_name:
                        facts_with_entity_name += 1
                        
                    is_local = source and not any(x in source for x in ["PubMed", "在线公开", "抓取异常"])
                    if is_local:
                        local_facts_count += 1
                        unique_sources.add(source)
                        
                        # Categorize source pattern
                        pattern = "Other"
                        if "实体库:" in source:
                            pattern = "实体库"
                        elif "图谱关系:" in source:
                            pattern = "图谱关系"
                        elif "常见临床疾病与合理用药诊疗路径" in source:
                            pattern = "临床路径 (medical.json)"
                        elif "临床路径" in source:
                            pattern = "临床路径 (PDF)"
                        elif "说明书" in source:
                            pattern = "说明书"
                        
                        source_patterns[pattern] = source_patterns.get(pattern, 0) + 1
            except Exception as e:
                print(f"Error at line {idx}: {e}")
                
    print(f"Total facts: {facts_count}")
    print(f"Facts with entity_name in metadata: {facts_with_entity_name}")
    print(f"Local facts expected to retrieve: {local_facts_count}")
    print(f"Unique local sources expected to retrieve: {len(unique_sources)}")
    print("\nSource Pattern Distribution among Local Facts:")
    for pat, cnt in source_patterns.items():
        print(f"  - {pat}: {cnt}")
        
    print("\nExamples of unique local sources (first 20):")
    for src in sorted(list(unique_sources))[:20]:
        print(f"  {src}")

if __name__ == "__main__":
    main()
