"""
restore_from_failures.py
从 purification_failures.jsonl 恢复被错误物理删除的数据（254-260行）。
用法：在 d:\REN\qa 目录下运行：python restore_from_failures.py
"""
import json
from pathlib import Path

DATASET = Path("medical_qa_dataset.jsonl")
RAW_BACKUP = Path("medical_qa_dataset_raw.jsonl")
FAILURES_LOG = Path("logs/purification_failures.jsonl")

DELETED_LINE_NUMBERS = set(range(254, 261))

if not FAILURES_LOG.exists():
    print(f"❌ {FAILURES_LOG} 不存在！")
    exit(1)

to_restore = {}
with open(FAILURES_LOG, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            orig_line = entry.get("original_line_number")
            if orig_line in DELETED_LINE_NUMBERS:
                # 记录最后一次出现的该行号数据
                to_restore[orig_line] = json.dumps(entry.get("data", {}), ensure_ascii=False) + "\n"
        except Exception as e:
            pass

if not to_restore:
    print(f"❌ 在 {FAILURES_LOG} 中未找到 254-260 行的备份数据。")
    exit(1)

sorted_lines_to_restore = [to_restore[k] for k in sorted(to_restore.keys())]

print(f"找到 {len(sorted_lines_to_restore)} 行数据可以恢复。")

def append_to_file(filepath, lines):
    with open(filepath, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
    
    with open(filepath, "r", encoding="utf-8") as f:
        print(f"  {filepath} 恢复后总行数: {len(f.readlines())}")

print("\n正在恢复主数据集...")
append_to_file(DATASET, sorted_lines_to_restore)

print("\n正在恢复 Raw 备份...")
append_to_file(RAW_BACKUP, sorted_lines_to_restore)

print("\n✅ 数据恢复完成！")
