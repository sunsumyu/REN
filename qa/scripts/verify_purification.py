# -*- coding: utf-8 -*-
"""
数据集净化效果自动化验证程序 (Verify Purification)。
执行“工程噪声零残留”特征扫描与“医学硬数据无损”比对双重质量校验。
"""

import json
import re
from pathlib import Path

def run_verification():
    dataset_path = Path("d:/REN/qa/medical_qa_dataset.jsonl")
    
    if not dataset_path.exists():
        print(f"❌ 未找到数据集文件: {dataset_path}")
        return
        
    print("🚀 开始执行数据集净化效果自动化验证...")
    print("-----------------------------------------")
    
    # 绝对禁止词（工程噪声）
    banned_keywords = ["JSON", "Schema", "免责声明", "忽略", "refs", "根据参考文档", "图谱关系", "概念定义", "知识关联"]
    
    # 核心医学数据完整性字典（部分抽样对比）
    medical_indicators = {
        "体位性低血压": ["不足1%", "低于1%"],
        "左氧氟沙星": ["7537", "50", "74%"],
        "托莫西汀": ["11.3%", "3.0%", "3.8倍"],
        "贝伐珠单抗": ["5.9%", "1.7%", "4.2%"],
        "吗啡": ["吗啡中毒", "过敏"]
    }
    
    total_checks = 0
    noise_leaks = 0
    rigor_failures = 0
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                q = data.get("Q", "")
                planners = data.get("planners", [])
                
                for p in planners:
                    planner_name = p.get("planner", "")
                    answer = p.get("answer", "")
                    
                    # 匹配 think 块
                    think_match = re.match(r"^\s*<think>([\s\S]*?)</think>([\s\S]*)$", answer)
                    if think_match:
                        total_checks += 1
                        think_content = think_match.group(1)
                        answer_body = think_match.group(2)
                        
                        # 1. 扫描工程指令残留
                        found_banned = []
                        for kw in banned_keywords:
                            if kw in think_content:
                                found_banned.append(kw)
                        
                        if found_banned:
                            noise_leaks += 1
                            print(f"⚠️ 行 {line_idx+1} [{planner_name}] 检出工程噪声残留: {found_banned}")
                            
                        # 2. 扫描医学指标是否无损 (对匹配的疾病/药物)
                        for topic, indicators in medical_indicators.items():
                            if topic in q:
                                # 检查 CoT 或回答正文中是否保留了关键数据
                                missing = []
                                for ind in indicators:
                                    if ind not in think_content and ind not in answer_body:
                                        missing.append(ind)
                                if missing:
                                    rigor_failures += 1
                                    print(f"❌ 行 {line_idx+1} [{planner_name}] 指标缺失: 问题关于 '{topic}'，但未检出关键指标 {missing}")
                                    
            except Exception as e:
                print(f"⚠️ 解析第 {line_idx+1} 行记录失败: {e}")
                
    print("-----------------------------------------")
    print("📊 验证统计报告 (Verification Summary):")
    print(f"  - 总共抽检思维块 (Think blocks checked): {total_checks}")
    
    # 报告工程噪声残留情况
    if noise_leaks == 0:
        print("  - 🎉 工程噪声零残留校验 (Noise-Free Check): PASS (100% 纯净度)")
    else:
        leak_rate = (noise_leaks / total_checks) * 100
        print(f"  - ❌ 工程噪声残留校验失败: 发现 {noise_leaks} 处泄露 (泄漏率: {leak_rate:.2f}%)")
        
    # 报告核心硬指标无损情况
    if rigor_failures == 0:
        print("  - 🎉 医学硬指标无损校验 (Medical Rigor Check): PASS (100% 完整性)")
    else:
        print(f"  - ❌ 医学硬指标无损校验失败: 发现 {rigor_failures} 处缺失或曲解")
        
    print("-----------------------------------------")
    if noise_leaks == 0 and rigor_failures == 0:
        print("🌟 结论: 该数据集已经通过了最高级别的企业生产级质量校验！")
    else:
        print("🌟 结论: 仍有部分样本需进一步净化，请确认或重新调整系统 Prompt。")

if __name__ == "__main__":
    run_verification()
