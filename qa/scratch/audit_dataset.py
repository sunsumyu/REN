import json
import re
import os

dataset_path = r"d:\REN\qa\medical_qa_dataset.jsonl"

def audit_dataset():
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    print(f"Auditing first 165 lines of {dataset_path}...\n")
    
    issues_found = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_num = idx + 1
            if line_num > 165:
                break
                
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                q = data.get("Q", "")
                planners = data.get("planners", [])
                summary = data.get("summary", "")
                refs = data.get("refs", [])
                
                # Check 1: Planner name corruption
                corrupted_planners = []
                for p_idx, p in enumerate(planners):
                    planner_name = p.get("planner", "")
                    if "rigorous data" in planner_name or "Schema" in planner_name:
                        corrupted_planners.append(p_idx)
                
                # Check 2: Engineering noise / conversational scaffolding in Think blocks
                noisy_planners = []
                for p_idx, p in enumerate(planners):
                    ans = p.get("answer", "")
                    think_match = re.match(r"^\s*<think>([\s\S]*?)</think>", ans)
                    if think_match:
                        think_content = think_match.group(1)
                        # Look for common developer / homework traces
                        traces = []
                        if "被要求" in think_content:
                            traces.append("被要求")
                        if "JSON" in think_content or "Schema" in think_content or "schema" in think_content:
                            traces.append("JSON/Schema")
                        if "refs" in think_content or "参考依据" in think_content:
                            traces.append("refs")
                        if re.search(r'\b[P|p]\d+\b', think_content):
                            traces.append("P1/P2/P3步骤")
                        if "由于" in think_content and "要求输出" in think_content:
                            traces.append("格式自我讨论")
                        if traces:
                            noisy_planners.append((p_idx, p.get("planner", "")[:15], traces))
                
                # Check 3: Mock / Fallback database hallucinations
                mock_hallucinations = []
                for ref_idx, ref in enumerate(refs):
                    source = ref.get("source", "")
                    context = ref.get("context", "")
                    
                    # If G6PD, UGT2B7, 密封, Rx, etc. are labeled as "用于辅助治疗"
                    if "用于辅助治疗" in context:
                        # Extract the drug name mentioned in the mock template
                        drug_match = re.search(r'药物【(.*?)】目前主要在临床上用于辅助治疗', context)
                        if drug_match:
                            ent_name = drug_match.group(1)
                            # If it's a known non-drug entity, it's a mock hallucination
                            if ent_name in ["密封", "Rx", "UGT2B7", "G6PD缺乏患者", "右旋糖酐 40", "尿酸转化为尿囊素", "高尿酸血症", "药物过量"]:
                                mock_hallucinations.append((ref_idx, ent_name))
                                
                    # If "在线公开检索系统抓取异常" is recorded
                    if "在线公开检索系统抓取异常" in source or "未收录或网络异常" in context:
                        mock_hallucinations.append((ref_idx, "网络异常兜底Mock"))
                
                # Record result for this line
                if corrupted_planners or noisy_planners or mock_hallucinations:
                    issues_found.append({
                        "line": line_num,
                        "Q": q[:25],
                        "corrupted_planners": corrupted_planners,
                        "noisy_planners": noisy_planners,
                        "mock_hallucinations": mock_hallucinations
                    })
                    
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                
    # Output statistics
    total_issues = len(issues_found)
    print(f"Total lines with issues: {total_issues} / 165\n")
    
    # Categorize and print details
    print("=== Category 1: Planner Name Corruption (System Prompt Leak in Planner Field) ===")
    c_count = 0
    for issue in issues_found:
        if issue["corrupted_planners"]:
            print(f"Line {issue['line']}: Q='{issue['Q']}' - Corrupted planner index: {issue['corrupted_planners']}")
            c_count += 1
    print(f"Subtotal: {c_count} lines\n")
    
    print("=== Category 2: Mock Hallucinations in Refs (Non-drugs labeled as drug, or abnormal crawl state saved) ===")
    m_count = 0
    for issue in issues_found:
        if issue["mock_hallucinations"]:
            print(f"Line {issue['line']}: Q='{issue['Q']}' - Mock items: {issue['mock_hallucinations']}")
            m_count += 1
    print(f"Subtotal: {m_count} lines\n")

    print("=== Category 3: CoT Engineering Noise (Think block contains JSON/schema/prompt meta-scaffolding) ===")
    n_count = 0
    for issue in issues_found:
        if issue["noisy_planners"]:
            # Display first 2 lines with traces for brevity
            if n_count < 10:
                print(f"Line {issue['line']}: Q='{issue['Q']}' - Noisy planners: {[(p[1], p[2]) for p in issue['noisy_planners']]}")
            n_count += 1
    print(f"Subtotal: {n_count} lines (showing first 10)\n")

if __name__ == "__main__":
    audit_dataset()
