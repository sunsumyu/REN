"""
restore_deleted_lines.py
从 raw 备份中还原被错误删除的行（254-260）到主数据集。
用法：在 d:\REN\qa 目录下运行：python restore_deleted_lines.py
"""
import json
from pathlib import Path

DATASET = Path("medical_qa_dataset.jsonl")
RAW_BACKUP = Path("medical_qa_dataset_raw.jsonl")

# 被错误删除的行号（1-indexed）
DELETED_LINE_NUMBERS = list(range(254, 261))  # 254~260

with open(RAW_BACKUP, encoding="utf-8") as f:
    raw_lines = f.readlines()

print(f"Raw backup total lines: {len(raw_lines)}")

# 提取需要还原的行（0-indexed）
to_restore = []
for ln in DELETED_LINE_NUMBERS:
    idx = ln - 1
    if idx >= len(raw_lines):
        print(f"  ⚠️  Line {ln} NOT found in raw backup (raw only has {len(raw_lines)} lines)")
        continue
    raw_line = raw_lines[idx].strip()
    if not raw_line:
        print(f"  ⚠️  Line {ln} is blank in raw backup, skipping")
        continue
    try:
        data = json.loads(raw_line)
        q = data.get("Q", "")[:50]
        print(f"  ✅ Line {ln}: Q={q}")
        to_restore.append(raw_lines[idx])
    except Exception as e:
        print(f"  ❌ Line {ln} JSON parse error: {e}")

if not to_restore:
    print("\n❌ Nothing to restore. Aborting.")
    exit(1)

print(f"\nRestoring {len(to_restore)} lines to {DATASET}...")

with open(DATASET, encoding="utf-8") as f:
    current_lines = f.readlines()

print(f"Current dataset lines before restore: {len(current_lines)}")

# 追加到末尾（因为物理删除后行号已经改变，按原顺序追加是最安全的方案）
with open(DATASET, "a", encoding="utf-8") as f:
    for line in to_restore:
        if not line.endswith("\n"):
            line += "\n"
        f.write(line)

with open(DATASET, encoding="utf-8") as f:
    final_lines = f.readlines()

print(f"Dataset lines after restore: {len(final_lines)}")
print("✅ Restore complete. Please verify the appended rows and update PURIFY_START_LINE in .env accordingly.")
