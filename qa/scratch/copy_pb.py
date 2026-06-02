import shutil
import os
import glob

src_dir = r"C:\Users\cf\.gemini\antigravity-ide\conversations"
dst_dir = r"d:\REN\qa\scratch"

def copy_files():
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist!")
        return
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        
    print(f"Copying files from {src_dir} to {dst_dir}...")
    pb_files = glob.glob(os.path.join(src_dir, "*.pb"))
    print(f"Found {len(pb_files)} .pb files.")
    for f in pb_files:
        try:
            shutil.copy(f, dst_dir)
            print(f"Copied {os.path.basename(f)}")
        except Exception as e:
            print(f"Failed to copy {os.path.basename(f)}: {e}")

if __name__ == "__main__":
    copy_files()
