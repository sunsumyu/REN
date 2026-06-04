# -*- coding: utf-8 -*-
import json
from pathlib import Path

def main():
    workspace_dir = Path(__file__).resolve().parent.parent
    dataset_path = workspace_dir / "medical_qa_dataset.jsonl"
    backup_path = workspace_dir / "medical_qa_dataset_raw.jsonl"

    print("Checking dataset files...")
    for name, path in [("dataset", dataset_path), ("raw", backup_path)]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            print(f"{name} exists, total lines: {len(lines)}")
            if len(lines) >= 193:
                line_193 = lines[192]
                try:
                    data = json.loads(line_193)
                    print(f"Line 193 Q in {name}: {data.get('Q', 'N/A')}")
                except Exception as e:
                    print(f"Line 193 in {name} is not valid JSON: {e}")
                    print(f"Content: {line_193[:100]}")
            else:
                print(f"{name} has fewer than 193 lines!")
        else:
            print(f"{name} does not exist at {path}")

if __name__ == "__main__":
    main()
