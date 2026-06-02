import json
import os

def restore_lines():
    dataset_path = r"d:\REN\qa\medical_qa_dataset.jsonl"
    backup_path = r"d:\REN\qa\medical_qa_dataset_raw.jsonl"
    
    if not os.path.exists(backup_path):
        print("❌ Backup file not found!")
        return
        
    print(f"Reading backup file: {backup_path}")
    with open(backup_path, 'r', encoding='utf-8') as f:
        backup_lines = f.readlines()
        
    print(f"Reading active dataset: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        active_lines = f.readlines()
        
    # Restore lines 120 to 127 (which is index 119 to 126)
    print("Restoring lines 120 to 127...")
    for i in range(119, 127):
        if i < len(backup_lines):
            print(f"Restoring/appending line {i+1}...")
            if i < len(active_lines):
                active_lines[i] = backup_lines[i]
            else:
                active_lines.append(backup_lines[i])
            
    print(f"Writing restored lines back to: {dataset_path}")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.writelines(active_lines)
        
    print("🎉 Restoration completed successfully!")

if __name__ == "__main__":
    restore_lines()
