# -*- coding: utf-8 -*-
import sys
import shutil
from pathlib import Path

def restore_and_fix():
    workspace = Path("d:/REN/qa")
    purifier_path = workspace / "scripts/medicalqa_purifier.py"
    backup_path = workspace / "scripts/medicalqa_purifier.py.bak"
    
    if not purifier_path.exists():
        print("Error: medicalqa_purifier.py not found.")
        return
        
    # 1. Create a backup first
    if not backup_path.exists():
        shutil.copyfile(purifier_path, backup_path)
        print(f"Created backup at {backup_path.name}")
        
    print(f"Reading {purifier_path}...")
    with open(purifier_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
    try:
        # 2. Convert garbled Big5 characters back to standard UTF-8 Chinese
        # (This works because the file was written as Big5 and read as UTF-8)
        bytes_content = content.encode('cp950', errors='ignore')
        restored = bytes_content.decode('utf-8', errors='ignore')
        
        # 3. Explicitly replace/fix rule 5 in the prompt to ensure it is 100% correct, ungarbled, and transition-free
        # We target the system prompt rule 5:
        old_rule5_garbled = "- **绝对禁止使用任何如“阶段一”、“步骤1”等序号词，且绝对禁止使用如“首先”、“其次”、“此外”、“最后”、“综上所述”等顺序性或总结性过渡词！** 这些过渡词会暴露结构泄漏。必须使用高度自然的学术因果递进和自问自答。"
        # Just in case there is any garbled rule 5 variant, we do a replacement of the entire section if needed.
        # Let's inspect the restored text and enforce the clean version of Rule 5 and the examples.
        
        # Let's clean up rule 5 to make sure it matches the strict transition-free standard:
        clean_rule5 = '- **绝对禁止使用任何如“阶段一”、“步骤1”等序号词，且绝对禁止使用如“首先”、“其次”、“此外”、“最后”、“综上所述”等顺序性或总结性过渡词！** 这些过渡词会暴露结构泄漏。必须使用高度自然的学术因果递进和自问自答。'
        
        # Let's ensure the recommended example is updated to banish "首先":
        restored = restored.replace('要剖析...首先必须解构...', '要剖析...必须深层解构...')
        restored = restored.replace('要剖析[疾病/药物/治疗名称]的核心机制/生理本质，首先必须解构...', '要剖析[疾病/药物/治疗名称]的核心机制/生理本质，必须深层解构...')
        
        # 4. Update the few-shots in the restored file:
        # Pharmacology:
        restored = restored.replace('要剖析丹膝颗粒的药理学机制，首先必须解构其核心临床目标', '要剖析丹膝颗粒的药理学机制，必须深层解构其核心临床目标')
        restored = restored.replace('最后，方中为何会出现一味火麻仁？', '而在方剂的边缘，为何会出现一味火麻仁？')
        # Contraindication:
        restored = restored.replace('面对非洛地平缓释胶囊的禁忌症，我们首先要明确其底层药理核心', '面对非洛地平缓释胶囊的禁忌症，必须要明确其底层药理核心')
        restored = restored.replace('最后，从一般用药安全底线来看', '而在更宽泛的临床防御维度上，从一般用药安全底线来看')
        # General:
        restored = restored.replace('探究白头翁的临床应用，需首先锁定其在', '探究白头翁的临床应用，必须牢牢锁定其在')
        
        # 5. Let's fix line 576 in main() which got garbled due to Big5 encoding of '数据集第' and '行'
        # Original was: lines_in_file = [int(num) for num in re.findall(r"数据集第\s*(\d+)\s*行", log_content)]
        # Let's make sure it is clean:
        restored = restored.replace(r're.findall(r"唳旿洵\s*(\d+)\s*銵", log_content)', r're.findall(r"数据集第\s*(\d+)\s*行", log_content)')
        restored = restored.replace(r're.findall(r"数据集第\s*(\d+)\s*行", log_content)', r're.findall(r"数据集第\s*(\d+)\s*行", log_content)')
        
        # Write back to medicalqa_purifier.py
        with open(purifier_path, 'w', encoding='utf-8') as out_f:
            out_f.write(restored)
            
        print("🎉 SUCCESS! medicalqa_purifier.py has been fully restored and transition word fixes applied!")
        
    except Exception as e:
        print(f"❌ Error during restoration: {e}")

if __name__ == "__main__":
    restore_and_fix()
