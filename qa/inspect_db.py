import os
import sqlite3

def inspect_db(db_path, table_name=None):
    print(f"\n==========================================")
    print(f"Inspecting database: {db_path}")
    print(f"==========================================")
    
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' does not exist on disk.")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables present: {tables}")
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  - Table '{table}': {count} rows")
            except Exception as e:
                print(f"  - Table '{table}': could not count rows ({e})")
                
        if table_name and table_name in tables:
            print(f"\nSchema of table '{table_name}':")
            cursor.execute(f"PRAGMA table_info({table_name})")
            info = cursor.fetchall()
            for col in info:
                print(f"  Col ID: {col[0]}, Name: {col[1]}, Type: {col[2]}, NotNull: {col[3]}, Default: {col[4]}, PK: {col[5]}")
                
            print(f"\nFirst 5 rows of '{table_name}':")
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()
            for row in rows:
                print(row)
                
    except Exception as e:
        print(f"Error inspecting DB '{db_path}': {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def inspect_vector_index(index_path):
    print(f"\n==========================================")
    print(f"Inspecting FAISS index: {index_path}")
    print(f"==========================================")
    if os.path.exists(index_path):
        size = os.path.getsize(index_path)
        print(f"FAISS index file exists! Size: {size} bytes ({size / 1024:.2f} KB)")
        try:
            import faiss
            index = faiss.read_index(index_path)
            print(f"Successfully loaded index using FAISS.")
            print(f"  - Dimension: {index.d}")
            print(f"  - Total vectors: {index.ntotal}")
        except Exception as e:
            print(f"Error loading index with FAISS: {e}")
    else:
        print(f"FAISS index file '{index_path}' does not exist on disk.")

def find_index_files(start_dir):
    print(f"\n==========================================")
    print(f"Searching for .index files under: {start_dir}")
    print(f"==========================================")
    found = []
    try:
        for root, dirs, files in os.walk(start_dir):
            # Skip directories like .git or __pycache__ to avoid slow search
            if any(part in root for part in ['.git', '__pycache__', '.cache']):
                continue
            for file in files:
                if file.endswith(".index"):
                    path = os.path.join(root, file)
                    found.append(path)
                    print(f"Found: {path} ({os.path.getsize(path)} bytes)")
        if not found:
            print("No .index files found in this directory tree.")
    except Exception as e:
        print(f"Error searching for files: {e}")

if __name__ == "__main__":
    inspect_db("qa_datasets.db", "datasets")
    inspect_db("local_rag.db", "local_rag_index")
    inspect_vector_index("local_rag_vector.index")
    find_index_files("d:\\REN")
