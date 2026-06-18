import re
from pathlib import Path

def main():
    doc_path = Path(r"d:\REN\qa\docs\data_flow_detailed_walkthrough.md")
    if not doc_path.exists():
        print("Doc not found!")
        return

    content = doc_path.read_text(encoding="utf-8")
    
    # regex to find links like [SomeText](file:///d:/REN/qa/path/to/file.py#L123) or #L123-L145
    # e.g., (file:///d:/REN/qa/core/pipeline_workflow.py#L90)
    links = re.findall(r'\[([^\]]+)\]\(file:///d:/REN/qa/([^\)]+)\)', content)
    
    print(f"Found {len(links)} links in walkthrough. Checking...")
    for text, link in links:
        parts = link.split('#')
        rel_path = parts[0]
        line_spec = parts[1] if len(parts) > 1 else None
        
        file_path = Path(r"d:\REN\qa") / rel_path.replace('/', '\\')
        if not file_path.exists():
            print(f"❌ File does not exist: {file_path}")
            continue
            
        file_lines = file_path.read_text(encoding="utf-8").splitlines()
        
        if line_spec:
            if '-' in line_spec:
                # Range like L105-L124
                # Parse L105 and L124
                m = re.match(r'L(\d+)-L(\d+)', line_spec)
                if m:
                    start_line = int(m.group(1))
                    end_line = int(m.group(2))
                    snippet = "\n".join(file_lines[start_line-1:end_line])
                    print(f"✔ Range {rel_path}#{line_spec} -> lines {start_line}-{end_line} (total lines {len(file_lines)})")
                else:
                    print(f"❓ Unknown range spec: {line_spec}")
            else:
                line_num = int(line_spec.replace('L', ''))
                if line_num <= len(file_lines):
                    target_line = file_lines[line_num - 1]
                    print(f"✔ Line {rel_path}#{line_spec} -> '{target_line.strip()}'")
                else:
                    print(f"❌ Line number {line_num} out of bounds for {rel_path} (total lines {len(file_lines)})")
        else:
            print(f"✔ File {rel_path} (no line spec)")

if __name__ == "__main__":
    main()
