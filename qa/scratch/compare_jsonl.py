import os
import json

files = ["logs/purification_audit.jsonl", "logs/kb_correction_audit.jsonl"]

for fpath in files:
    if os.path.exists(fpath):
        print(f"=== File: {fpath} ===")
        with open(fpath, "rb") as f:
            content = f.read()
        print(f"  Size: {len(content)} bytes")
        print(f"  Lines count: {len(content.split(b'\n'))}")
        # Try to parse each line as JSON
        lines = content.split(b'\n')
        for idx, line in enumerate(lines):
            if line.strip():
                try:
                    obj = json.loads(line.decode('utf-8'))
                    print(f"  Line {idx+1}: VALID JSON (Keys: {list(obj.keys())})")
                except Exception as e:
                    print(f"  Line {idx+1}: INVALID JSON! Error: {e}")
                    print(f"    Line content: {repr(line)}")
    else:
        print(f"File {fpath} does not exist.")
