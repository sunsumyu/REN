# -*- coding: utf-8 -*-
import os

logs_dir = "logs"
files_to_fix = ["kb_correction_audit.jsonl", "kb_correction_audit.txt"]

for filename in files_to_fix:
    path = os.path.join(logs_dir, filename)
    if os.path.exists(path):
        print(f"Fixing line endings for {path}...")
        # Read content
        with open(path, "rb") as f:
            content = f.read()
        
        # Replace CRLF (\r\n) with LF (\n)
        fixed_content = content.replace(b"\r\n", b"\n")
        
        # Write back
        with open(path, "wb") as f:
            f.write(fixed_content)
        print(f"Successfully fixed {path}. New size: {len(fixed_content)} bytes.")
    else:
        print(f"{path} does not exist.")
