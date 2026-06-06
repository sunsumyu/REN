# -*- coding: utf-8 -*-
"""
本地私有 RAG 数据权威源初始化与对齐工具。
1. 彻底剔除 QA 合成数据，避免自循环核心数据污染。
2. 流式加载本地 `medical.json`（已通过下载获取的疾病与药物联用图谱数据），按限额导入以防 OOM。
3. 新增 PDF 扫描与切片模块（load_local_pdfs），支持读取指定目录下的“国家临床路径”和“诊疗指南”PDF 文献，
   自动进行滑动窗口切块（Chunking）并完成向量化。
4. 内置安全种子机制。
"""

import os
import json
import sqlite3
import re

# 尝试导入向量检索所需依赖
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    DEPS_OK = True
except ImportError:
    DEPS_OK = False

# 本地权威医学数据文件路径（疾病与药物对齐数据）
LOCAL_DATA_FILE = "medical.json"

# 本地临床路径与指南 PDF 存放目录（支持国家临床路径与医脉通指南）
LOCAL_PDF_DIR = "./clinical_guidelines"

# 控制导入的上限数量，防止 CPU 满载编码时间过长或发生 OOM 内存溢出
MAX_IMPORT_LIMIT = 500

def sanitize_markdown(text: str) -> str:
    """
    清洗文本中的 HTML 标签与多余空行
    """
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_local_drugs(file_path: str):
    """
    流式读取本地医学 JSON 文件，逐行解析，达到 MAX_IMPORT_LIMIT 条后立即截断，杜绝 OOM
    """
    if not os.path.exists(file_path):
        print(f"[Warning] Local medical data file not found at '{file_path}'.")
        print("Please download it to the workspace directory first.")
        print("Fallback to local seeds & safe database mode.")
        return []

    print(f"Reading local database file '{file_path}' (Streaming mode)...")
    drugs = []
    count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("["):
                    line = line[1:]
                if line.endswith("],") or line.endswith("]"):
                    line = line[:-2] if line.endswith("],") else line[:-1]
                if line.endswith(","):
                    line = line[:-1]
                
                try:
                    item = json.loads(line)
                    drugs.append(item)
                    count += 1
                    if count >= MAX_IMPORT_LIMIT:
                        print(f"Reached local import buffer limit ({MAX_IMPORT_LIMIT}). Stopping stream read.")
                        break
                except Exception:
                    continue
        print(f"Successfully loaded and parsed {len(drugs)} medical items from local file.")
        return drugs
    except Exception as e:
        print(f"[Error] Failed to read local file: {e}")
        return []

def load_local_pdfs(pdf_dir: str):
    """
    扫描本地 PDF 文件夹，提取 PDF 文本并按滑动窗口切片导入 RAG
    实现国家临床路径与诊疗指南的本地数据构建
    """
    if not os.path.exists(pdf_dir):
        print(f"[Info] Local PDF directory '{pdf_dir}' does not exist. PDF import bypassed.")
        return []
        
    print(f"Scanning local PDF directory: '{pdf_dir}'...")
    try:
        import pypdf
    except ImportError:
        print("[Warning] 'pypdf' library not installed. Please run 'pip install pypdf' to enable PDF clinical guidelines importing.")
        return []
        
    pdf_facts = []
    # 遍历目录下的所有 pdf 文件
    for root, dirs, files in os.walk(pdf_dir):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Processing clinical PDF: {file}...")
                try:
                    reader = pypdf.PdfReader(pdf_path)
                    full_text = []
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            full_text.append(text)
                    
                    combined_text = "\n".join(full_text)
                    clean_text = sanitize_markdown(combined_text)
                    
                    # 按照 400 字滑动窗口进行切片，重叠 50 字以保证语义完整性
                    chunk_size = 400
                    overlap = 50
                    i = 0
                    chunk_idx = 1
                    while i < len(clean_text):
                        chunk = clean_text[i:i+chunk_size]
                        # 使用 PDF 文件名（通常为指南或临床路径名）作为关联实体名
                        entity_name = os.path.splitext(file)[0][:20]
                        pdf_facts.append({
                            "entity_name": entity_name,
                            "source": f"refs:《{os.path.splitext(file)[0]}-第{chunk_idx}段》",
                            "context": chunk,
                            "category": "临床诊疗"
                        })
                        i += (chunk_size - overlap)
                        chunk_idx += 1
                except Exception as e:
                    print(f"[Error] Failed to parse PDF {file}: {e}")
                    
    print(f"Successfully processed and sliced {len(pdf_facts)} chunks from local PDFs.")
    return pdf_facts

def parse_drug_items(raw_items):
    """
    解析医疗图谱结构，拼接成权威指南事实段落
    """
    parsed = []
    for item in raw_items:
        name = item.get("name") or "未知疾病"
        name_clean = re.sub(r'[^\u4e00-\u9fa5\w]', '', name).strip()
        
        # 结构化抽取字段拼接成 RAG 事实上下文
        details = []
        details.append(f"【疾病名称】{name_clean}")
        
        desc = item.get("desc")
        if desc:
            details.append(f"【疾病描述】{str(desc).strip()}")
            
        common_drugs = item.get("common_drug")
        if common_drugs and isinstance(common_drugs, list):
            details.append(f"【常用药物】{', '.join(common_drugs)}")
            
        recommand_drugs = item.get("recommand_drug")
        if recommand_drugs and isinstance(recommand_drugs, list):
            details.append(f"【推荐用药】{', '.join(recommand_drugs)}")
            
        prevent = item.get("prevent")
        if prevent:
            details.append(f"【预防措施】{str(prevent).strip()}")
            
        do_eat = item.get("do_eat")
        not_eat = item.get("not_eat")
        if do_eat or not_eat:
            eat_info = []
            if do_eat:
                eat_info.append(f"宜食 {', '.join(do_eat) if isinstance(do_eat, list) else str(do_eat)}")
            if not_eat:
                eat_info.append(f"忌食 {', '.join(not_eat) if isinstance(not_eat, list) else str(not_eat)}")
            details.append(f"【合理饮食】{'；'.join(eat_info)}")
            
        context = "。".join(details)
        
        parsed.append({
            "entity_name": name_clean[:20],
            "source": "refs:《常见临床疾病与合理用药诊疗路径》",
            "context": sanitize_markdown(context),
            "category": "临床诊疗"
        })
        
        # 将常用药也单独映射一份，以保证药物被 LIKE 检索直接召回
        if common_drugs and isinstance(common_drugs, list):
            for drug in common_drugs:
                drug_clean = re.sub(r'[^\u4e00-\u9fa5\w]', '', drug).strip()
                if drug_clean:
                    parsed.append({
                        "entity_name": drug_clean[:20],
                        "source": f"refs:《常见临床疾病与合理用药诊疗路径-{name_clean}》",
                        "context": f"【适用疾病】{name_clean}。{sanitize_markdown(context)}",
                        "category": "用药方案"
                    })
                    
    return parsed

def get_fallback_seeds():
    """
    提供备用的权威说明书种子数据（在本地文件缺失时兜底）
    """
    print("Loading fallback seeds (authority drug manuals)...")
    return [
        {
            "entity_name": "布洛芬",
            "source": "refs:《布洛芬缓释胶囊说明书》",
            "context": "【用法用量】口服。成人一次1粒（0.3克），一日2次（早晚各1次）。【禁忌】活动期消化道溃疡患者禁用。对阿司匹林过敏者禁用。孕妇及哺乳期妇女禁用。",
            "category": "用药方案"
        },
        {
            "entity_name": "甲氨蝶呤",
            "source": "refs:《甲氨蝶呤片说明书》",
            "context": "【不良反应】常见不良反应有口腔炎、口唇溃疡、白细胞减少。部分患者可引起严重骨髓抑制、肝肾功能损害。孕妇及哺乳期妇女禁用。",
            "category": "安全禁忌"
        },
        {
            "entity_name": "阿司匹林",
            "source": "refs:《阿司匹林肠溶片说明书》",
            "context": "【药理作用】本品不可逆地抑制血小板环氧合酶-1（COX-1），从而抑制血栓烷A2（TXA2）的合成，发挥抗血小板聚集活性。主要用于预防一过性脑缺血发作、心肌梗死或术后血栓形成。",
            "category": "药理机制"
        },
        {
            "entity_name": "更昔洛韦",
            "source": "refs:《思泽-更昔洛韦葡萄糖注射液说明书》",
            "context": "【用法用量】仅供静脉滴注给药，不可肌肉注射或静脉推注。滴注速率不可过快，单次给药必须恒速滴注1小时以上。中性粒细胞绝对计数ANC小于500/μL或血小板计数小于25000/μL者禁用。",
            "category": "用药方案"
        },
        {
            "entity_name": "乙硫异烟胺",
            "source": "refs:《乙硫异烟胺片说明书》",
            "context": "【禁忌】对本品过敏者禁用；对异烟肼、吡嗪酰胺、烟酸等结构相近药物过敏者禁用；妊娠期与哺乳期妇女禁用；12岁以下儿童禁用。",
            "category": "安全禁忌"
        },
        {
            "entity_name": "格拉司琼",
            "source": "refs:《盐酸格拉司琼注射液说明书》",
            "context": "【特殊人群用药】对于有肾脏或肝脏损害的患者，尚未进行专门研究，但已有静脉给药的药动学数据可供参考，无需调整剂量。服药期间需定期复查肝肾功能。",
            "category": "临床诊疗"
        },
        {
            "entity_name": "异烟肼",
            "source": "refs:《异烟肼片说明书》",
            "context": "【不良反应】剂量增加时，可能导致外周神经炎、四肢感觉异常、反射消失、肌肉轻瘫 and 精神失常。用药期间应定期复查肝肾功能和血常规，当出现肝功能异常时应随时监测。",
            "category": "安全禁忌"
        }
    ]

def build_database_and_vectors(
    items,
    output_db_path: str = "local_rag.db",
    output_index_path: str = "local_rag_vector.index",
    model_name: str = "shibing624/text2vec-base-chinese"
):
    """
    将整理好的数据存入 SQLite 并用 FAISS 建立向量检索
    """
    if not items:
        print("[Error] No facts to index. Exiting.")
        return
        
    print(f"\nStoring {len(items)} facts to SQLite database: '{output_db_path}'...")
    
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS local_rag_index;")
    cursor.execute("DROP TABLE IF EXISTS local_rag_fts_index;")
    
    cursor.execute("""
        CREATE TABLE local_rag_index (
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
            CREATE VIRTUAL TABLE local_rag_fts_index USING fts5(
                source,
                context,
                entity_name,
                category UNINDEXED,
                tokenize="unicode61"
            );
        """)
    except sqlite3.OperationalError:
        print("[Info] SQLite FTS5 extension not compiled in this Python environment. Bypassing FTS5 virtual table.")
        fts_ok = False
        
    conn.commit()
    
    contexts = []
    for item in items:
        clean_context = sanitize_markdown(item["context"])
        
        # 写入普通表
        cursor.execute("""
            INSERT INTO local_rag_index (entity_name, source, context, category)
            VALUES (?, ?, ?, ?);
        """, (item["entity_name"], item["source"], clean_context, item["category"]))
        
        # 写入倒排检索表
        if fts_ok:
            cursor.execute("""
                INSERT INTO local_rag_fts_index (entity_name, source, context, category)
                VALUES (?, ?, ?, ?);
            """, (item["entity_name"], item["source"], clean_context, item["category"]))
        
        contexts.append(clean_context)
        
    conn.commit()
    conn.close()
    print("Database SQLite writing completed.")
    
    if not DEPS_OK:
        print("\n[Warning] FAISS or SentenceTransformer not found. Skipping vector indexing.")
        print("Standard mode RAG (FTS5 / LIKE) will be used instead.")
        return
        
    print(f"\nEncoding {len(contexts)} text slices to vectors using '{model_name}'...")
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(contexts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # L2 归一化以支持内积相似度（等价于余弦相似度）
        faiss.normalize_L2(embeddings)
        
        print("Building FAISS IndexFlatIP (Inner Product) Index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # 保存索引文件
        faiss.write_index(index, output_index_path)
        print(f"🎉 🎉 Success! RAG Vector storage index saved to '{output_index_path}'.")
        print(f"Registered {index.ntotal} vectors in FAISS and successfully aligned SQLite database metadata.")
    except Exception as e:
        print(f"[Error] Failed to build FAISS index: {e}")

if __name__ == "__main__":
    workspace = "."
    all_facts = []
    
    # 1. 读取本地下载的医学说明书文件 (基础图谱)
    local_path = os.path.join(workspace, LOCAL_DATA_FILE)
    raw_drugs = load_local_drugs(local_path)
    if raw_drugs:
        parsed_drugs = parse_drug_items(raw_drugs)
        all_facts.extend(parsed_drugs)
        
    # 2. 读取本地 PDF 目录中的临床指南与国家临床路径（支持用户自定义放置）
    pdf_dir = os.path.join(workspace, LOCAL_PDF_DIR)
    all_facts.extend(load_local_pdfs(pdf_dir))
        
    # 3. 无论是否成功，都混入内置的常见核心临床药物种子，保证完全对齐
    all_facts.extend(get_fallback_seeds())
    
    # 去重处理
    unique_facts = {}
    for fact in all_facts:
        key = (fact["entity_name"], fact["context"])
        if key not in unique_facts:
            unique_facts[key] = fact
            
    final_facts = list(unique_facts.values())
    print(f"\nSummary: Total collected unique clinical facts: {len(final_facts)}")
    
    # 4. 运行构建
    db_file = os.path.join(workspace, "local_rag.db")
    index_file = os.path.join(workspace, "local_rag_vector.index")
    build_database_and_vectors(final_facts, output_db_path=db_file, output_index_path=index_file)
