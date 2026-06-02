import json
import re
import os

dataset_path = r"d:\REN\qa\medical_qa_dataset.jsonl"
report_output_path = r"d:\REN\qa\scratch\audit_report_100_plus.json"

# Facet classifications for mismatch detection
SIMPLE_LOOKUP_KEYWORDS = [
    "成分是什么", "主治是什么", "化学名称", "分子式", "执行标准", "批准文号", "规格是什么", 
    "贮藏方法", "有效期", "忌用人群", "用法用量是什么", "禁忌人群"
]

COMPLEX_FACETS = [
    "患者数据隐私", "药物动力学", "分子机制", "预后评估", "病理生理学", "临床治疗"
]

def audit_dataset_100_plus():
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    print(f"Auditing lines 101 to end of {dataset_path}...")
    
    records_audited = 0
    issues_found = []
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_num = idx + 1
            if line_num < 101:
                continue
                
            if not line.strip():
                continue
                
            records_audited += 1
            try:
                data = json.loads(line)
                q = data.get("Q", "")
                planners = data.get("planners", [])
                refs = data.get("refs", [])
                
                for p_idx, p in enumerate(planners):
                    facet = p.get("planner", "")
                    ans = p.get("answer", "")
                    
                    think_content = ""
                    think_match = re.search(r"<think>([\s\S]*?)</think>", ans)
                    if think_match:
                        think_content = think_match.group(1)
                    
                    # 1. Detect Forced Facet (硬套视角)
                    forced_facet_indicators = []
                    # Method A: Heuristic keywords in thinking block showing struggle
                    struggle_patterns = [
                        r"没有(直接)?(提到|提供).*(数据隐私|隐私|动力学|分子机制|机制|预后|病理生理|病理)",
                        r"视角是.*(但|而)问题是",
                        r"(为了符合|结合)这个视角",
                        r"强行将.*联系",
                        r"有些牵强",
                        r"不得不",
                        r"refs.*没有.*信息",
                        r"如何结合.*视角"
                    ]
                    for pattern in struggle_patterns:
                        if re.search(pattern, think_content):
                            forced_facet_indicators.append(f"Struggle comment matching: {pattern}")
                    
                    # Method B: Semantic clash (Simple question + complex facet)
                    is_simple_question = any(kw in q for kw in SIMPLE_LOOKUP_KEYWORDS)
                    is_complex_facet = facet in COMPLEX_FACETS
                    if is_simple_question and is_complex_facet:
                        # If it's a simple lookup but mapped to complex facet, and the think content shows forced bridging
                        if len(forced_facet_indicators) > 0 or facet == "患者数据隐私":
                            forced_facet_indicators.append(f"Semantic Clash: Simple Q with Complex Facet '{facet}'")
                    
                    # 2. CoT Quality (思维链质量)
                    cot_quality_issues = []
                    # Engineering meta prompt leaks
                    prompt_leaks = []
                    if "JSON" in think_content or "Schema" in think_content or "schema" in think_content:
                        prompt_leaks.append("JSON/Schema instruction trace")
                    if "refs" in think_content or "参考依据" in think_content or "文献" in think_content:
                        # Check if it refers to "refs" as a technical variable
                        if re.search(r"\brefs\b|refs中|根据refs", think_content):
                            prompt_leaks.append("Technical variable 'refs' leak")
                    if re.search(r'\b[P|p]\d+\b', think_content):
                        prompt_leaks.append("Step identifier P1/P2/P3 trace")
                    if "用户" in think_content or "任务" in think_content or "输入数据" in think_content:
                        prompt_leaks.append("Meta-instructions reference ('用户/任务/输入数据')")
                        
                    # Logic density check
                    reasoning_words = len(re.findall(r"因为|所以|从而|由此|基于|推导|反之|也就是说|提示|意味着|如果", think_content))
                    total_words = len(think_content)
                    logic_density = reasoning_words / max(total_words, 1)
                    
                    if len(think_content) < 150:
                        cot_quality_issues.append("Thinking chain too short")
                    if prompt_leaks:
                        cot_quality_issues.append(f"Prompt leaks found: {prompt_leaks}")
                    
                    # 3. Hallucinations & False Information (幻觉与假信息)
                    hallucinations = []
                    # A. Mock template hallucinations in refs
                    for ref_idx, ref in enumerate(refs):
                        ref_src = ref.get("source", "")
                        ref_ctx = ref.get("context", "")
                        if "用于辅助治疗" in ref_ctx:
                            drug_match = re.search(r'药物【(.*?)】目前主要在临床上用于辅助治疗', ref_ctx)
                            if drug_match:
                                ent_name = drug_match.group(1)
                                if ent_name in ["密封", "Rx", "UGT2B7", "G6PD", "发热", "糖尿病"]:
                                    hallucinations.append(f"Mock ref: Non-drug entity '{ent_name}' labeled as drug")
                        if "在线公开检索系统抓取异常" in ref_src:
                            hallucinations.append("Mock ref: Crawl error fallback used as factual source")
                    
                    # B. Check if numbers in output are made up (not in refs)
                    # Extract numbers from answer/thinking and see if they appear in refs
                    nums_in_ans = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', ans))
                    # Remove small numbers like 1, 2, 3, etc. or common years
                    filtered_nums_ans = {n for n in nums_in_ans if len(n) > 1 and not n.startswith("202")}
                    
                    refs_text = " ".join([r.get("context", "") for r in refs])
                    unsupported_nums = []
                    for num in filtered_nums_ans:
                        if num not in refs_text and f"{num}%" not in refs_text:
                            # Let's see if it's a standard clinical dose or value, or fabricated
                            unsupported_nums.append(num)
                    
                    if unsupported_nums:
                        # Some numbers might be general clinical knowledge (like 12岁, 100mg, 5g). We only flag if they look like fabricated stats
                        suspicious_nums = [n for n in unsupported_nums if "%" in n or "." in n or int(re.sub(r'[^\d]', '', n)) > 500]
                        if suspicious_nums:
                            hallucinations.append(f"Unsupported numbers in output: {suspicious_nums}")

                    # Record issues if any found
                    if forced_facet_indicators or cot_quality_issues or hallucinations:
                        issues_found.append({
                            "line": line_num,
                            "Q": q,
                            "facet": facet,
                            "forced_facet_indicators": forced_facet_indicators,
                            "cot_quality_issues": cot_quality_issues,
                            "logic_density": f"{logic_density:.4f}",
                            "hallucinations": hallucinations
                        })
                        
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                
    # Save the audit report
    report = {
        "dataset_path": dataset_path,
        "total_records_audited": records_audited,
        "total_lines_with_issues": len(issues_found),
        "issues": issues_found
    }
    with open(report_output_path, "w", encoding="utf-8") as out_f:
        json.dump(report, out_f, ensure_ascii=False, indent=2)
        
    print(f"\nAudit complete! {len(issues_found)} issues found out of {records_audited} records.")
    print(f"Report saved to {report_output_path}.\n")
    
    # Print summary of findings
    forced_count = sum(1 for i in issues_found if i["forced_facet_indicators"])
    quality_count = sum(1 for i in issues_found if i["cot_quality_issues"])
    hallucination_count = sum(1 for i in issues_found if i["hallucinations"])
    
    print(f"=== Summary ===")
    print(f"- Forced Facets (硬套视角): {forced_count} records")
    print(f"- CoT Quality Issues (思维链质量): {quality_count} records")
    print(f"- Hallucinations & Mock Errors (幻觉与假信息): {hallucination_count} records")
    print("\nTop 5 typical issues found:")
    for issue in issues_found[:5]:
        print(f"Line {issue['line']} | Q: '{issue['Q'][:20]}...' | Facet: '{issue['facet']}'")
        if issue["forced_facet_indicators"]:
            print(f"  * Forced Facet: {issue['forced_facet_indicators']}")
        if issue["cot_quality_issues"]:
            print(f"  * CoT Issues: {issue['cot_quality_issues']}")
        if issue["hallucinations"]:
            print(f"  * Hallucinations: {issue['hallucinations']}")
        print()

if __name__ == "__main__":
    audit_dataset_100_plus()
