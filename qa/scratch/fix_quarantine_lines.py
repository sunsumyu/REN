# -*- coding: utf-8 -*-
import os
import json
import shutil
import datetime
from pathlib import Path

def main():
    workspace_dir = Path(__file__).resolve().parent.parent
    dataset_path = workspace_dir / "medical_qa_dataset.jsonl"
    backup_path = workspace_dir / "medical_qa_dataset_raw.jsonl"
    logs_dir = workspace_dir / "logs"
    md_path = logs_dir / "purification_run_[174-193]_20260603_182534.md"

    failures_jsonl_path = logs_dir / "purification_failures.jsonl"
    failures_backup_jsonl_path = logs_dir / "purification_failures_backup.jsonl"
    failures_md_path = logs_dir / "purification_failures.md"

    print("=== Start Offline Rollback Line Cleanup with Failure Logging & Backup ===")

    # 1. 物理删除第 193 行 (1-indexed, index = 192)
    target_idx = 192  # 193行对应0-indexed 192
    removed_data = None

    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Current dataset lines: {len(lines)}")
        if len(lines) >= 193:
            removed_line = lines.pop(target_idx)
            try:
                removed_data = json.loads(removed_line)
            except Exception as e:
                print(f"Failed to parse removed line JSON: {e}")
            print(f"Removed line 193 from dataset: {removed_line[:60]}...")
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"Updated dataset saved. New line count: {len(lines)}")
        else:
            print("Warning: dataset has fewer than 193 lines, skipping dataset deletion.")
    else:
        print("Dataset not found.")

    if backup_path.exists():
        with open(backup_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
        print(f"Current raw backup lines: {len(raw_lines)}")
        if len(raw_lines) >= 193:
            removed_raw = raw_lines.pop(target_idx)
            print(f"Removed line 193 from raw backup: {removed_raw[:60]}...")
            with open(backup_path, "w", encoding="utf-8") as f:
                f.writelines(raw_lines)
            print(f"Updated raw backup saved. New line count: {len(raw_lines)}")
        else:
            print("Warning: raw backup has fewer than 193 lines, skipping raw deletion.")
    else:
        print("Raw backup not found.")

    # 2. 修复 md 报告里的描述
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        old_pattern = "- **第 193 行**"
        new_pattern = "- **第 193 行 (已物理删除)**"
        if old_pattern in md_content:
            md_content = md_content.replace(old_pattern, new_pattern)
            print("Successfully updated line 193 description in purification MD report.")
        else:
            print("Warning: target rollback description pattern not found in MD report.")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
    else:
        print("MD Report not found.")

    # 3. 🚦 单独记录到失败 jsonl 和 md，执行文件版本安全备份
    if removed_data is not None:
        timestamp_str = datetime.datetime.now().isoformat()
        failure_reason = "planner purification failed or planner count changed"
        
        # A. 备份原有的失败 jsonl 记录文件
        if failures_jsonl_path.exists():
            try:
                shutil.copy2(failures_jsonl_path, failures_backup_jsonl_path)
                print(f"🛡️ Backup existing failure JSONL records to: {failures_backup_jsonl_path.name}")
            except Exception as e:
                print(f"Warning: failed to backup failures JSONL: {e}")

        # B. 追加记录到失败 jsonl
        failure_event = {
            "timestamp": timestamp_str,
            "original_line_number": 193,
            "reason": failure_reason,
            "data": removed_data
        }
        should_write_jsonl = True
        if failures_jsonl_path.exists():
            try:
                with open(failures_jsonl_path, "r", encoding="utf-8") as jf_read:
                    for jl in jf_read:
                        if jl.strip():
                            try:
                                je = json.loads(jl)
                                if je.get("original_line_number") == 193 or je.get("data", {}).get("Q") == removed_data.get("Q"):
                                    should_write_jsonl = False
                                    print(f"📝 Line 193 failure event is already in: {failures_jsonl_path.name}")
                                    break
                            except Exception:
                                pass
            except Exception as e:
                print(f"Warning reading failures JSONL: {e}")

        if should_write_jsonl:
            try:
                with open(failures_jsonl_path, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(failure_event, ensure_ascii=False) + "\n")
                print(f"📝 Appended line 193 failure metadata to global graveyard: {failures_jsonl_path.name}")
            except Exception as e:
                print(f"Error appending failure event: {e}")

        # C. 记录/更新失败 md 描述文件
        write_header = not failures_md_path.exists()
        q_text = removed_data.get("Q", "N/A")
        should_write_md = True
        if failures_md_path.exists():
            try:
                with open(failures_md_path, "r", encoding="utf-8") as f_md:
                    if q_text in f_md.read():
                        should_write_md = False
                        print(f"📝 Line 193 failure report is already in: {failures_md_path.name}")
            except Exception as e:
                print(f"Warning reading failures MD: {e}")

        if should_write_md:
            try:
                with open(failures_md_path, "a", encoding="utf-8") as mf:
                    if write_header:
                        mf.write("# 🩺 医疗问答思维链提纯净化「历史失败/隔离行」汇总墓地\n\n")
                        mf.write("本描述文件记录了在数据集提纯净化阶段中因语义不兼容、格式崩溃或质检评分未通过而被**物理丢弃隔离**的所有样本，作为后续诊断和微调优化的依据。\n\n")
                        mf.write("| 时间戳 | 原始行号 | 失败原因 | 主问题 (Q) | 视角 (Facets) |\n")
                        mf.write("| :--- | :--- | :--- | :--- | :--- |\n")
                    
                    planners = [p.get("planner", "") for p in removed_data.get("planners", [])]
                    mf.write(f"| {timestamp_str} | 193 | {failure_reason} | `{q_text}` | {', '.join(planners)} |\n")
                print(f"📝 Appended line 193 failure report to global graveyard: {failures_md_path.name}")
            except Exception as e:
                print(f"Error appending failure report to md: {e}")

    print("=== Cleanup Complete! ===")

if __name__ == "__main__":
    main()
