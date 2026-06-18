# -*- coding: utf-8 -*-
"""
递归删除目录下的旧版 .doc 文件
"""
import os
import argparse

def delete_doc_files(root_dir: str):
    print(f"==========================================")
    print(f"Starting to clean up old .doc files")
    print(f"Target Directory: {root_dir}")
    print(f"==========================================\n")
    
    if not os.path.exists(root_dir):
        print(f"[Error] Directory not found: {root_dir}")
        return

    deleted_count = 0
    # 递归遍历目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 严格匹配以 .doc 结尾的文件（忽略大小写），且避开 .docx
            if filename.lower().endswith(".doc") and not filename.startswith("~$"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {os.path.basename(file_path)}")
                    deleted_count += 1
                except Exception as e:
                    print(f"[Error] Failed to delete {os.path.basename(file_path)}: {e}")
                    
    print(f"\n==========================================")
    print(f"Cleanup finished! Successfully deleted {deleted_count} '.doc' files.")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively delete .doc files in a directory.")
    parser.add_argument("--dir", type=str, required=True, help="Root directory containing .doc files to delete.")
    
    args = parser.parse_args()
    delete_doc_files(args.dir)
