with open(r"d:\REN\qa\core\pipeline_workflow.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(270, 280):
    line = lines[i]
    print(f"Line {i+1}: {repr(line)}")
    # Print the ascii/hex values of characters in the line
    hex_vals = [hex(ord(c)) for c in line]
    print(f"Hex: {' '.join(hex_vals)}")
