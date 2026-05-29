import os
import re
from pathlib import Path

LOGS_DIR = Path("d:/REN/qa/logs")
DOCS_DIR = Path("d:/REN/qa/docs")

def rename_in_dir(directory):
    print(f"Scanning directory: {directory}")
    if not directory.exists():
        print(f"Directory {directory} does not exist. Skipping.")
        return
        
    for file_path in directory.glob("purification_run_*.md"):
        # Skip already renamed or special files
        if "[" in file_path.name or file_path.name in ["purification_run.md", "purification_run_recovered_partial.md"]:
            continue
            
        # Parse timestamp from file name (format purification_run_YYYYMMDD_HHMMSS.md)
        match = re.match(r"purification_run_(\d{8}_\d{6})\.md", file_path.name)
        if not match:
            continue
            
        timestamp = match.group(1)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Find all line numbers in the format "数据集第 X 行"
            line_nums = [int(n) for n in re.findall(r"数据集第\s*(\d+)\s*行", content)]
            if line_nums:
                min_line = min(line_nums)
                max_line = max(line_nums)
                new_name = f"purification_run_[{min_line}-{max_line}]_{timestamp}.md"
                new_path = file_path.parent / new_name
                print(f"Renaming {file_path.name} -> {new_name}")
                os.rename(file_path, new_path)
            else:
                print(f"Skipping {file_path.name} (no line numbers found)")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

if __name__ == "__main__":
    rename_in_dir(LOGS_DIR)
    rename_in_dir(DOCS_DIR)
    print("Done renaming logs!")
