import reprlib

path = "logs/kb_correction_audit.jsonl"
with open(path, "rb") as f:
    data = f.read()

print("File size:", len(data))
print("Bytes (start):", repr(data[:200]))
print("Bytes (end):", repr(data[-200:]))
# Check if there are any null bytes
print("Contains null bytes:", b"\x00" in data)
