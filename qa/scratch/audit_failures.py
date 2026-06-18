# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

def main():
    workspace = Path("d:/REN/qa")
    failures_jsonl = workspace / "logs/purification_failures.jsonl"
    dataset_path = workspace / "medical_qa_dataset.jsonl"
    
    if not failures_jsonl.exists():
        print("failures.jsonl not found")
        return
        
    failed_queries = set()
    with open(failures_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                q = data.get("data", {}).get("Q", "")
                if q:
                    failed_queries.add(q.strip())
            except Exception as e:
                print(f"Error parsing failure line: {e}")
                
    print(f"Loaded {len(failed_queries)} unique failed queries from purification_failures.jsonl")
    
    if not dataset_path.exists():
        print("dataset not found")
        return
        
    dataset_lines = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset_lines = [line.strip() for line in f if line.strip()]
        
    found_failures = []
    for idx, line_str in enumerate(dataset_lines):
        try:
            data = json.loads(line_str)
            q = data.get("Q", "").strip()
            # Also check if it has "failed" status or fallback signatures
            has_fallback = False
            for p in data.get("planners", []):
                answer = p.get("answer", "")
                if "触发高可用防拒答" in answer or "质量不达标" in answer or "failed" in answer:
                    has_fallback = True
            
            if q in failed_queries or has_fallback:
                found_failures.append({
                    "line_number": idx + 1,
                    "query": q,
                    "reason": "in_failed_queries" if q in failed_queries else "has_fallback_signature"
                })
        except Exception as e:
            print(f"Error parsing dataset line {idx+1}: {e}")
            
    print(f"Found {len(found_failures)} failed/rollback samples still present in medical_qa_dataset.jsonl:")
    for item in found_failures:
        print(f"Line {item['line_number']}: Q: '{item['query']}' | Reason: {item['reason']}")

if __name__ == '__main__':
    main()
