import os
import glob
import time

docs_dir = "d:/REN/qa/docs"
files = glob.glob(os.path.join(docs_dir, "*"))
files.sort(key=os.path.getmtime, reverse=True)

print("Files in docs sorted by modification time:")
for f in files:
    mtime = os.path.getmtime(f)
    print(f"{os.path.basename(f)}: {time.ctime(mtime)} ({mtime})")
