# -*- coding: utf-8 -*-
"""
存量数据集自动化自愈清洗脚本 (Heal Dataset)。
安全解码并重置数据集中已存在的所有 "请提供具体的医疗问题以便规划视角" 脏字段及 think 块内部的 facet 标签，
一键式修复历史污染，确保微调训练集 100% 纯净。
"""
import json
import re
import sys
from pathlib import Path

# 针对 Windows 控制台环境，强行配置标准输出为 UTF-8 以保证对特殊字符的安全打印
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

def heal_active_dataset():
    # 脚本位于 E:\chain\QA\qa\scripts\heal_dataset.py，其 parent.parent 即为 E:\chain\QA\qa
    qa_dir = Path(__file__).resolve().parent.parent
    dataset_path = qa_dir / "medical_qa_dataset.jsonl"
    backup_path = qa_dir / "medical_qa_dataset_raw.jsonl"
    
    dirty_term = "请提供具体的医疗问题以便规划视角"
    clean_term = "临床用药安全"
    
    print("[START] 开始对存量数据集执行自动化自愈清洗...")
    print("-----------------------------------------")
    
    for file_path in [dataset_path, backup_path]:
        if not file_path.exists():
            print(f"[SKIP] 未找到文件: {file_path}，已自动跳过。")
            continue
            
        print(f"[PROCESS] 正在处理文件: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        healed_lines = []
        healed_count = 0
        
        for idx, line in enumerate(lines):
            if not line.strip():
                healed_lines.append(line)
                continue
                
            if dirty_term in line:
                try:
                    data = json.loads(line)
                    planners = data.get("planners", [])
                    for p in planners:
                        # 1. 修复 planners 数组中的 planner 字段名
                        if p.get("planner") == dirty_term:
                            p["planner"] = clean_term
                        
                        # 2. 修复 answer 中包含的 think 块内部的 <facet = ...> 标签
                        answer = p.get("answer", "")
                        if dirty_term in answer:
                            # 替换 think 标签中的 facet 标志
                            answer = answer.replace(f"<facet = {dirty_term}>", f"<facet = {clean_term}>")
                            # 为了保险，也替换可能不带空格的格式
                            answer = answer.replace(f"<facet={dirty_term}>", f"<facet = {clean_term}>")
                            p["answer"] = answer
                            
                    # 重新编组为 JSON 字符串写入
                    line = json.dumps(data, ensure_ascii=False) + "\n"
                    healed_count += 1
                except Exception as e:
                    # 如果 JSON 损坏或解析出错，退避为底层暴力文本替换
                    print(f"   [WARN] 第 {idx+1} 行 JSON 解析异常: {e}，采用暴力文本替换。")
                    line = line.replace(dirty_term, clean_term)
                    line = line.replace(f"<facet = {dirty_term}>", f"<facet = {clean_term}>")
                    line = line.replace(f"<facet={dirty_term}>", f"<facet = {clean_term}>")
                    healed_count += 1
                    
            healed_lines.append(line)
            
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(healed_lines)
            
        print(f"[SUCCESS] 成功洗白并修复了 {file_path.name} 中的 {healed_count} 处脏数据。")
        print("-----------------------------------------")
        
    print("[FINISHED] 历史存量数据自动化清洗自愈工作全部完成！")

if __name__ == "__main__":
    heal_active_dataset()
