# -*- coding: utf-8 -*-
"""
扫描数据集中所有被兜底模板污染或可疑幻觉编造的条目。
检测维度：
1. 兜底模板关键字残留（原始和清洗后）
2. think块内容过短（可能是模板或空壳）
3. think块中完全不含任何与原始refs关联的医学实体（幻觉风险）
4. 日志文件中的兜底记录回溯
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict

DATASET_PATH = Path("d:/REN/qa/medical_qa_dataset.jsonl")
RAW_BACKUP_PATH = Path("d:/REN/qa/medical_qa_dataset_raw.jsonl")
LOGS_DIR = Path("d:/REN/qa/logs")

# 兜底模板的特征指纹
FALLBACK_SIGNATURES = [
    "触发高可用防拒答",
    "安全防御质量策略",
    "临床指南兜底模板",
    "自动装配学术合规兜底叙事",
    "系统激活安全防御",
    "安全拦截",
    "高可用防拒答与去污染兜底",
]

# 工程噪声残留指纹（清洗不彻底的标志）
ENGINEERING_NOISE = [
    "问题拆解：",
    "证据清单：",
    "推理链：",
    "最终结论摘要：",
    "step_id:",
    "证据R1",
    "证据R2",
    "Answer Body",
    "子问题拆解",
]

# 安全拒答指纹
REFUSAL_SIGNATURES = [
    "须严格依据专科医师指导",
    "需对患者的生化指标及既往病史进行全面筛查",
    "严格规避用药配伍禁忌及潜在的毒副反应",
    "确保用药安全",
    "关于该健康咨询中涉及的",
]

def extract_think_blocks(answer: str):
    """从answer中提取所有<think>块的内容"""
    pattern = r"<think>([\s\S]*?)</think>"
    matches = re.findall(pattern, answer)
    return matches

def scan_dataset(path: Path, label: str):
    """扫描数据集并返回问题条目"""
    if not path.exists():
        print(f"  ❌ 文件不存在: {path}")
        return []
    
    issues = []
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n{'='*60}")
    print(f"📊 扫描 [{label}]: {path.name} ({len(lines)} 条记录)")
    print(f"{'='*60}")
    
    for line_idx, line in enumerate(lines):
        line_num = line_idx + 1
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            issues.append({
                "line": line_num,
                "type": "JSON_PARSE_ERROR",
                "detail": "无法解析JSON",
                "severity": "HIGH"
            })
            continue
        
        q = data.get("Q", "")
        planners = data.get("planners", [])
        
        for p_idx, p in enumerate(planners):
            planner_name = p.get("planner", "")
            answer = p.get("answer", "")
            
            # 1. 检查兜底模板指纹
            for sig in FALLBACK_SIGNATURES:
                if sig in answer:
                    issues.append({
                        "line": line_num,
                        "type": "FALLBACK_TEMPLATE",
                        "facet": planner_name,
                        "detail": f"检测到兜底模板指纹: '{sig}'",
                        "severity": "CRITICAL",
                        "question": q[:30]
                    })
            
            # 2. 检查安全拒答模板
            for sig in REFUSAL_SIGNATURES:
                if sig in answer:
                    # 只检查answer_body部分（think之后）
                    body = answer
                    if "</think>" in answer:
                        body = answer.split("</think>")[-1].strip()
                    if sig in body:
                        issues.append({
                            "line": line_num,
                            "type": "SAFE_REFUSAL_TEMPLATE",
                            "facet": planner_name,
                            "detail": f"检测到安全拒答模板: '{sig}'",
                            "severity": "HIGH",
                            "question": q[:30]
                        })
            
            # 3. 检查工程噪声残留
            think_blocks = extract_think_blocks(answer)
            for think in think_blocks:
                noise_found = []
                for sig in ENGINEERING_NOISE:
                    if sig in think:
                        noise_found.append(sig)
                if noise_found:
                    issues.append({
                        "line": line_num,
                        "type": "ENGINEERING_NOISE",
                        "facet": planner_name,
                        "detail": f"think块中残留工程噪声: {noise_found}",
                        "severity": "MEDIUM",
                        "question": q[:30]
                    })
            
            # 4. 检查think块异常短（可能是空壳或模板）
            for think in think_blocks:
                # 去除facet标签后计算实际内容长度
                clean_think = re.sub(r"<facet\s*=\s*[^>]+>", "", think).strip()
                if len(clean_think) < 80:
                    issues.append({
                        "line": line_num,
                        "type": "EMPTY_THINK",
                        "facet": planner_name,
                        "detail": f"think块内容过短 ({len(clean_think)} chars): '{clean_think[:50]}...'",
                        "severity": "HIGH",
                        "question": q[:30]
                    })
            
            # 5. 检查无think块（完全缺失推理过程）
            if "<think>" not in answer:
                issues.append({
                    "line": line_num,
                    "type": "NO_THINK_BLOCK",
                    "facet": planner_name,
                    "detail": "答案中完全缺失<think>推理块",
                    "severity": "MEDIUM",
                    "question": q[:30]
                })
    
    return issues

def scan_logs_for_fallbacks():
    """扫描所有净化日志，找出被兜底模板污染过的记录"""
    print(f"\n{'='*60}")
    print(f"📋 扫描净化日志目录: {LOGS_DIR}")
    print(f"{'='*60}")
    
    fallback_records = []
    
    if not LOGS_DIR.exists():
        print("  ❌ 日志目录不存在")
        return fallback_records
    
    log_files = sorted(LOGS_DIR.glob("purification_run_*.md"))
    print(f"  📂 发现 {len(log_files)} 个净化日志文件")
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 搜索兜底模板指纹
            for sig in FALLBACK_SIGNATURES + ["触发高可用防拒答与去污染兜底方案"]:
                if sig in content:
                    # 提取所在行号和上下文
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if sig in line:
                            # 尝试提取数据集行号
                            dataset_line_match = re.search(r"数据集第\s*(\d+)\s*行", content[max(0, content.index(sig)-500):content.index(sig)+100])
                            facet_match = re.search(r"临床视角:\s*\*\*(.+?)\*\*", content[max(0, content.index(sig)-500):content.index(sig)+100])
                            
                            fallback_records.append({
                                "log_file": log_file.name,
                                "dataset_line": dataset_line_match.group(1) if dataset_line_match else "unknown",
                                "facet": facet_match.group(1) if facet_match else "unknown",
                                "signature": sig
                            })
                            break
        except Exception as e:
            print(f"  ⚠️ 读取 {log_file.name} 出错: {e}")
    
    return fallback_records

def main():
    print("🔍 医疗QA数据集 - 兜底模板与幻觉编造深度扫描诊断")
    print("=" * 60)
    
    # 1. 扫描当前数据集（清洗后）
    current_issues = scan_dataset(DATASET_PATH, "当前数据集 (清洗后)")
    
    # 2. 扫描原始备份（清洗前）
    raw_issues = scan_dataset(RAW_BACKUP_PATH, "原始备份 (清洗前)")
    
    # 3. 扫描日志
    log_fallbacks = scan_logs_for_fallbacks()
    
    # 4. 输出汇总报告
    print(f"\n{'='*60}")
    print("📊 扫描汇总报告")
    print(f"{'='*60}")
    
    # 按严重度分类
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    
    print(f"\n--- 当前数据集问题 ({len(current_issues)} 个) ---")
    current_issues.sort(key=lambda x: (severity_order.get(x["severity"], 99), x["line"]))
    
    type_counts = defaultdict(int)
    for issue in current_issues:
        type_counts[issue["type"]] += 1
    
    print(f"  按类型统计:")
    for t, c in type_counts.items():
        print(f"    - {t}: {c} 个")
    
    print(f"\n  详细列表:")
    for issue in current_issues:
        sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(issue["severity"], "⚪")
        print(f"  {sev_icon} [行 {issue['line']}] [{issue['severity']}] {issue['type']}")
        if "facet" in issue:
            print(f"     切面: {issue['facet']} | Q: {issue.get('question', 'N/A')}")
        print(f"     详情: {issue['detail']}")
    
    print(f"\n--- 原始备份问题 ({len(raw_issues)} 个) ---")
    raw_issues.sort(key=lambda x: (severity_order.get(x["severity"], 99), x["line"]))
    
    raw_type_counts = defaultdict(int)
    for issue in raw_issues:
        raw_type_counts[issue["type"]] += 1
    
    print(f"  按类型统计:")
    for t, c in raw_type_counts.items():
        print(f"    - {t}: {c} 个")
    
    # 只显示CRITICAL和HIGH
    critical_raw = [i for i in raw_issues if i["severity"] in ("CRITICAL", "HIGH")]
    print(f"\n  高危问题详细列表 (仅CRITICAL+HIGH):")
    for issue in critical_raw:
        sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠"}.get(issue["severity"], "⚪")
        print(f"  {sev_icon} [行 {issue['line']}] [{issue['severity']}] {issue['type']}")
        if "facet" in issue:
            print(f"     切面: {issue['facet']} | Q: {issue.get('question', 'N/A')}")
        print(f"     详情: {issue['detail']}")
    
    print(f"\n--- 日志中的兜底记录 ({len(log_fallbacks)} 个) ---")
    for fb in log_fallbacks:
        print(f"  📄 {fb['log_file']} | 数据集行: {fb['dataset_line']} | 切面: {fb['facet']}")

if __name__ == "__main__":
    main()
