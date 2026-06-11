# -*- coding: utf-8 -*-
"""
递归遍历目录，利用 LibreOffice 将旧版 .doc 转换为 .docx 格式
"""
import os
import subprocess
import argparse

def convert_doc_to_docx(root_dir: str, soffice_path: str):
    print(f"==========================================")
    print(f"Starting recursive .doc -> .docx conversion")
    print(f"Target Directory: {root_dir}")
    print(f"LibreOffice Path: {soffice_path}")
    print(f"==========================================\n")
    
    if not os.path.exists(soffice_path):
        print(f"[Error] LibreOffice executable not found at: {soffice_path}")
        print("Please check the path and try again.")
        return
        
    # 收集所有的 .doc 文件
    doc_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 过滤临时文件 ~$xxx.doc，并且确保以 .doc 结尾（排除 .docx）
            if filename.lower().endswith(".doc") and not filename.startswith("~$"):
                doc_files.append(os.path.join(dirpath, filename))
                
    if not doc_files:
        print("No .doc files found in the specified directory.")
        return
        
    print(f"Found {len(doc_files)} '.doc' files to convert. This may take a while...\n")
    
    success_count = 0
    for idx, doc_path in enumerate(doc_files):
        print(f"[{idx+1}/{len(doc_files)}] Converting: {os.path.basename(doc_path)}")
        
        # 确保输出到 .doc 所在的同一个子文件夹
        out_dir = os.path.dirname(doc_path)
        
        cmd = [
            soffice_path,
            "--headless",
            "--convert-to", "docx",
            "--outdir", out_dir,
            doc_path
        ]
        
        try:
            # 调用命令行并等待执行完成
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 验证生成的 docx 是否存在
            expected_docx = doc_path + "x"
            if os.path.exists(expected_docx):
                success_count += 1
            else:
                print(f"  [Warning] Command succeeded but {expected_docx} was not found.")
                
        except subprocess.CalledProcessError as e:
            print(f"  [Error] Failed to convert: {e.stderr.strip() if e.stderr else 'Unknown LibreOffice error'}")
            
    print("\n==========================================")
    print(f"Conversion finished! Successfully converted {success_count} / {len(doc_files)} files.")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively convert .doc to .docx using LibreOffice.")
    parser.add_argument("--dir", type=str, required=True, help="Root directory containing .doc files.")
    parser.add_argument("--soffice", type=str, default=r"d:\Program Files\LibreOffice\program\soffice.exe", help="Path to soffice.exe.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"[Error] Directory not found: {args.dir}")
    else:
        convert_doc_to_docx(args.dir, args.soffice)
