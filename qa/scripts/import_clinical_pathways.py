# -*- coding: utf-8 -*-
"""
临床路径二进制文档 (.doc) 批量提取与增量向量化导入工具。
1. 采用 MarkItDown 作为首选解析引擎，提取出整洁的 Markdown 文本；
2. 采用 pywin32 + python-docx 作为备份降级方案，以应对缺依赖报错；
3. 将提取的文本按 400 字滑动窗口进行 Chunking 切片；
4. 增量写入 SQLite 本地 RAG 库，并对表中全量数据重新构建 FAISS 索引以保持对齐。
"""

import os
import sys
import sqlite3
import re
import argparse

# 尝试导入向量引擎依赖
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    DEPS_OK = True
except ImportError:
    DEPS_OK = False

# 默认病种临床路径物理路径
DEFAULT_SOURCE_DIR = r"C:\Users\cf\Downloads\1733999360046_18385\224个病种临床路径（2019年版）"
DEFAULT_DB_PATH = "local_rag.db"
DEFAULT_INDEX_PATH = "local_rag_vector.index"
MODEL_NAME = "shibing624/text2vec-base-chinese"

def sanitize_markdown(text: str) -> str:
    """
    清洗文本中的 HTML 标签与多余空行
    """
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_entity_name(filename: str) -> str:
    """
    从临床路径文件名中抽取出纯净的病种名称
    """
    # 移除“临床路径”、“2019年版”、“（内科）”以及后缀名
    name = re.sub(r'(?:临床路径)?(?:（2019年版）|\(2019年版\))?\.doc[x]?$', '', filename).strip()
    name = re.sub(r'（[^）]+）|\([^)]+\)', '', name).strip()
    return name[:20]

def extract_text_from_doc(doc_path: str) -> str:
    """
    文档文本抽取器 (双轨制安全提取)
    """
    # 轨道一：使用微软 MarkItDown 提取 (Markdown 格式更利于切片)
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        print(f"-> Extracting '{os.path.basename(doc_path)}' using MarkItDown...")
        res = md.convert(doc_path)
        if res and res.text_content:
            return res.text_content
    except Exception as e:
        print(f"-> [Info] MarkItDown extraction failed or not installed: {e}. Trying pywin32 fallback...")

    # 轨道二：自动降级为 pywin32 COM 另存为 docx，再使用 python-docx 解析
    try:
        import win32com.client as win32
        import docx
        
        abs_doc_path = os.path.abspath(doc_path)
        abs_docx_path = abs_doc_path + "x" # .docx
        
        # 启动 Word/WPS
        word = win32.gencache.EnsureDispatch('Word.Application')
        word.Visible = False
        doc = word.Documents.Open(abs_doc_path)
        # FileFormat=16 代表 docx 格式
        doc.SaveAs(abs_docx_path, FileFormat=16)
        doc.Close()
        word.Quit()
        
        # 读取另存后的 Docx 文件
        doc_obj = docx.Document(abs_docx_path)
        paragraphs = [p.text for p in doc_obj.paragraphs]
        full_text = "\n".join(paragraphs)
        
        # 物理清理临时文件
        if os.path.exists(abs_docx_path):
            os.remove(abs_docx_path)
            
        print(f"-> [Success] Successfully extracted using pywin32 Office Bridge.")
        return full_text
    except Exception as ex:
        print(f"-> [Error] Failed to extract '{os.path.basename(doc_path)}' using pywin32: {ex}")
        return ""

def slice_text(text: str, chunk_size: int = 400, overlap: int = 50):
    """
    对文本进行固定长度的滑动窗口切片
    """
    clean_text = sanitize_markdown(text)
    chunks = []
    i = 0
    while i < len(clean_text):
        chunk = clean_text[i:i+chunk_size]
        if len(chunk.strip()) > 10:  # 忽略没有实际字数的空片段
            chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

def scan_doc_files(source_dir: str):
    """
    递归扫描目录下的所有 .doc 文件
    """
    doc_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".doc") and not file.startswith("~$"):
                doc_files.append(os.path.join(root, file))
    return sorted(doc_files)

def run_import(source_dir: str, db_path: str, index_path: str, max_docs: int = 30):
    print("==================================================")
    print("Clinical Pathways Import & Vectorization Pipeline")
    print("==================================================")
    
    if not os.path.exists(source_dir):
        print(f"[Error] Source directory does not exist: {source_dir}")
        sys.exit(1)
        
    doc_files = scan_doc_files(source_dir)
    print(f"Found {len(doc_files)} total .doc clinical pathway documents.")
    
    # 限制处理的文档篇数，防止 CPU OOM 或过载
    if max_docs > 0 and len(doc_files) > max_docs:
        print(f"Limiting import count to first {max_docs} documents for safety (Change via args).")
        target_files = doc_files[:max_docs]
    else:
        target_files = doc_files
        
    print(f"Preparing to process {len(target_files)} documents...")

    # 1. 扫描并加载现存的 SQLite 数据以执行增量去重
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 确保数据库表和 FTS 索引存在（防止 local_rag.db 被删除后运行报错）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS local_rag_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT,
            source TEXT,
            context TEXT,
            category TEXT
        );
    """)
    fts_ok = True
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS local_rag_fts_index USING fts5(
                source,
                context,
                entity_name,
                category UNINDEXED,
                tokenize="unicode61"
            );
        """)
    except sqlite3.OperationalError:
        fts_ok = False
        
    cursor.execute("SELECT entity_name, context FROM local_rag_index;")
    existing_records = set((row[0], row[1]) for row in cursor.fetchall())
    print(f"Database loaded. Already contains {len(existing_records)} clinical facts.")

    new_facts_count = 0
    # 2. 逐一提取和切片
    for idx, filepath in enumerate(target_files):
        print(f"\n[{idx+1}/{len(target_files)}] Processing: {os.path.basename(filepath)}")
        text_content = extract_text_from_doc(filepath)
        
        if not text_content or len(text_content.strip()) < 50:
            print(f"-> [Warning] Skip empty or failed document: {os.path.basename(filepath)}")
            continue
            
        chunks = slice_text(text_content)
        print(f"-> Sliced into {len(chunks)} chunks.")
        
        entity = clean_entity_name(os.path.basename(filepath))
        source_name = f"refs:《国家卫健委-2019版临床路径-{os.path.splitext(os.path.basename(filepath))[0]}》"
        
        for chunk_idx, chunk in enumerate(chunks):
            # 去重检测
            if (entity, chunk) not in existing_records:
                # 写入 SQLite Ordinary 表
                cursor.execute("""
                    INSERT INTO local_rag_index (entity_name, source, context, category)
                    VALUES (?, ?, ?, ?);
                """, (entity, f"{source_name}-段{chunk_idx+1}", chunk, "临床诊疗"))
                
                # 写入 SQLite FTS 表
                if fts_ok:
                    cursor.execute("""
                        INSERT INTO local_rag_fts_index (entity_name, source, context, category)
                        VALUES (?, ?, ?, ?);
                    """, (entity, f"{source_name}-段{chunk_idx+1}", chunk, "临床诊疗"))
                    
                existing_records.add((entity, chunk))
                new_facts_count += 1
                
    conn.commit()
    print(f"\n==========================================")
    print(f"SQLite Write Completed. Added {new_facts_count} new facts.")
    
    # 3. 读取 SQLite 全量数据重新生成 FAISS 向量索引以保证主键严格对齐
    cursor.execute("SELECT id, context FROM local_rag_index ORDER BY id;")
    db_rows = cursor.fetchall()
    conn.close()
    
    print(f"Total clinical facts in DB: {len(db_rows)}")
    
    if not DEPS_OK:
        print("[Warning] FAISS or SentenceTransformer not installed. Skip rebuilding vector index.")
        return
        
    contexts = [row[1] for row in db_rows]
    print(f"\nEncoding all {len(contexts)} text fragments to vectors using '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(contexts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # L2 归一化以支持内积余弦相似度
        faiss.normalize_L2(embeddings)
        
        print("Rebuilding FAISS IndexFlatIP index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # 物理保存
        faiss.write_index(index, index_path)
        print(f"🎉 🎉 Success! Rebuilt FAISS index saved to '{index_path}'.")
        print(f"Total registered vectors: {index.ntotal}")
    except Exception as e:
        print(f"[Error] Failed to build FAISS index: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical pathways doc importer.")
    parser.add_argument("--src", type=str, default=DEFAULT_SOURCE_DIR, help="Source .doc files path.")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="Path to local_rag.db.")
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX_PATH, help="Path to local_rag_vector.index.")
    parser.add_argument("--limit", type=int, default=0, help="Deprecated.")
    parser.add_argument("--max_docs", type=int, default=30, help="Max doc limit (Default 30, use 0 for all).")
    
    # 支持命令行接收全量导入参数
    args, unknown = parser.parse_known_args()
    
    max_d = args.max_docs
    # 兼容直接传 --all 
    if "--all" in sys.argv:
        max_d = 0
        
    run_import(args.src, args.db, args.index, max_docs=max_d)
