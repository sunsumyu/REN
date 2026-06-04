#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Secondary Purification Script for Medical QA Dataset
This script cleans up residual RAG/engineering traces and meta-narrative openings 
in the purified thinking processes and summaries.
"""

import json
import os
import re
import shutil
import argparse

# List of regex replacements for RAG/engineering metadata cleanup
TEXT_REPLACEMENTS = [
    # 1. Database/RAG pipeline terminology
    (r"实体信息", "相关说明"),
    (r"实体库", "说明书"),
    (r"图谱关系", "关联信息"),
    (r"知识图谱", "医学知识库"),
    (r"JSON Schema", "数据规范"),
    (r"JSON schema", "数据规范"),
    (r"JSON对象", "结构化数据"),
    (r"JSON格式", "结构化格式"),
    (r"refs数据", "参考资料"),
    (r"输入refs", "参考资料"),
    (r"根据refs", "根据参考资料"),
    (r"基于refs", "基于参考资料"),
    (r"从refs中", "从参考资料中"),
    (r"在refs中", "在参考资料中"),
    
    # 2. Meta-narrative transitions or helper words
    (r"好的，我们从药物化学成分的视角切入，来解析这个问题。\s*", ""),
    (r"我需要遍历确证的药理描述，做一个严格的逻辑分离。\s*", ""),
    (r"我们从药物化学成分的视角切入，来解析这个问题。\s*", ""),
    (r"遍历确证的药理描述，做一个严格的逻辑分离。\s*", ""),
]

# Regex patterns for removing boilerplate meta-intros at the start of thinking block
META_INTRO_PATTERNS = [
    r"^好的，我们从[^，。]+的视角切入，来解析这个问题[。？]?\s*",
    r"^我需要[^，。]+，做一个严格的逻辑分离[。？]?\s*",
    r"^我们被要求以[^，。]+为[核主]?心视角，输出JSON对象[。？]?\s*",
    r"^首先解析任务：\s*",
]

def clean_text(text):
    if not isinstance(text, str):
        return text
    
    original = text
    
    # Apply regular replacements
    for pattern, replacement in TEXT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
        
    # Apply meta-intro patterns specifically at the beginning of the think block or text
    # Extract think block if present to apply start-of-think changes
    think_match = re.match(r"^(<think>\s*)(.*?)(\s*</think>)(.*)$", text, re.DOTALL)
    if think_match:
        prefix, think_content, suffix, rest = think_match.groups()
        cleaned_think = think_content
        for pattern in META_INTRO_PATTERNS:
            cleaned_think = re.sub(pattern, "", cleaned_think)
        text = f"{prefix}{cleaned_think}{suffix}{rest}"
    else:
        # If no think block, apply to the start of the text
        for pattern in META_INTRO_PATTERNS:
            text = re.sub(pattern, "", text)
            
    return text, text != original

def purify_dataset(input_path, output_path, dry_run=False):
    print(f"[*] Starting secondary purification...")
    
    # 1. Automated backup and redirection logic to preserve the first-purified dataset
    base_dir = os.path.dirname(os.path.abspath(input_path))
    first_purified_path = os.path.join(base_dir, "medical_qa_dataset_first_purified.jsonl")
    
    if input_path == output_path:
        if not os.path.exists(first_purified_path):
            if not dry_run:
                print(f"[+] Creating backup of the first-purified dataset: {first_purified_path}")
                shutil.copy2(input_path, first_purified_path)
            else:
                print(f"[+] [Dry Run] Will backup the first-purified dataset to: {first_purified_path}")
        else:
            print(f"[+] Found existing first-purified backup at: {first_purified_path}")
            print(f"[+] Reading from the first-purified backup to avoid double-processing.")
            input_path = first_purified_path

    print(f"[*] Input: {input_path}")
    print(f"[*] Output: {output_path}")
    print(f"[*] Mode: {'DRY RUN (No changes will be saved)' if dry_run else 'ACTIVE PURIFICATION'}")
    
    if not os.path.exists(input_path):
        print(f"[!] Error: Input file '{input_path}' does not exist.")
        return
        
    modified_count = 0
    total_count = 0
    change_log = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    output_lines = []
    
    for idx, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            output_lines.append("")
            continue
            
        total_count += 1
        try:
            data = json.loads(line)
        except Exception as e:
            print(f"[!] Warning: Error parsing line {idx}: {e}")
            output_lines.append(line)
            continue
            
        line_modified = False
        changes_in_line = []
        
        # 1. Purify planners' answers
        if "planners" in data and isinstance(data["planners"], list):
            for p_idx, planner in enumerate(data["planners"]):
                if "answer" in planner and isinstance(planner["answer"], str):
                    original_ans = planner["answer"]
                    cleaned_ans, modified = clean_text(original_ans)
                    if modified:
                        planner["answer"] = cleaned_ans
                        line_modified = True
                        changes_in_line.append(f"planners[{p_idx}].answer")
                        
        # 2. Purify summary
        if "summary" in data and isinstance(data["summary"], str):
            original_sum = data["summary"]
            cleaned_sum, modified = clean_text(original_sum)
            if modified:
                data["summary"] = cleaned_sum
                line_modified = True
                changes_in_line.append("summary")
                
        if line_modified:
            modified_count += 1
            change_log.append((idx, data.get("Q", "Unknown Q"), changes_in_line))
            output_lines.append(json.dumps(data, ensure_ascii=False))
        else:
            output_lines.append(line)
            
    print(f"\n[+] Scan completed. Total cases checked: {total_count}")
    print(f"[+] Cases with RAG/meta traces detected: {modified_count}")
    
    if modified_count > 0:
        print("\n[*] Detailed Change Log:")
        for line_num, question, fields in change_log:
            print(f"  - Line {line_num} | Q: '{question[:25]}...' | Fields modified: {', '.join(fields)}")
            
    if not dry_run:
        # Backup the current output file to .bak before overwriting
        if os.path.exists(output_path):
            backup_path = output_path + ".bak"
            print(f"[*] Creating runtime backup of current output dataset at: {backup_path}")
            shutil.copy2(output_path, backup_path)
            
        with open(output_path, "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")
        print(f"\n[+] Successfully saved purified dataset to: {output_path}")
    else:
        print("\n[*] Dry run completed. No files were written.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secondary purification of medical QA dataset.")
    parser.add_argument("--input", default="d:/REN/qa/medical_qa_dataset.jsonl", help="Path to input jsonl file")
    parser.add_argument("--output", default="d:/REN/qa/medical_qa_dataset.jsonl", help="Path to output jsonl file")
    parser.add_argument("--dry-run", action="store_true", help="Perform scan and show changes without saving")
    
    args = parser.parse_args()
    purify_dataset(args.input, args.output, args.dry_run)
