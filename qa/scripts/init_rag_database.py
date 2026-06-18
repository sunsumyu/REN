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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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

# 本地已净化的临床路径 Markdown 目录
LOCAL_MD_DIR = "C:/Users/cf/Downloads/1733999360046_18385/224个病种临床路径（2019年版）/purified_markdown"

# 开源药品说明书数据库本地存放路径
OPEN_SOURCE_DRUG_FILE = "./open_source_drugs.json"

# 控制导入的上限数量，防止 CPU 满载编码时间过长或发生 OOM 内存溢出
MAX_IMPORT_LIMIT = int(os.getenv("MAX_IMPORT_LIMIT", "10000"))

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
                    
                    # 按照 400 字滑动窗口进行语义切片，重叠 50 字以保证语义完整性
                    chunk_size = 400
                    overlap = 50
                    
                    import sys
                    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                    from utils.text_splitter import semantic_slice_text
                    
                    chunks = semantic_slice_text(clean_text, chunk_size, overlap)
                    chunk_idx = 1
                    for chunk in chunks:
                        # 使用 PDF 文件名（通常为指南或临床路径名）作为关联实体名
                        entity_name = os.path.splitext(file)[0][:20]
                        pdf_facts.append({
                            "entity_name": entity_name,
                            "source": f"refs:《{os.path.splitext(file)[0]}-第{chunk_idx}段》",
                            "context": chunk,
                            "category": "临床诊疗"
                        })
                        chunk_idx += 1
                except Exception as e:
                    print(f"[Error] Failed to parse PDF {file}: {e}")
                    
    print(f"Successfully processed and sliced {len(pdf_facts)} chunks from local PDFs.")
    return pdf_facts

def load_local_markdowns(md_dir: str):
    """
    扫描本地 Markdown 文件夹，读取 .md 文件并按滑动窗口切片导入 RAG
    支持导入已净化的国家临床路径数据
    """
    if not os.path.exists(md_dir):
        print(f"[Info] Local Markdown directory '{md_dir}' does not exist. MD import bypassed.")
        return []
        
    print(f"Scanning local Markdown directory: '{md_dir}'...")
    md_facts = []
    
    # 遍历目录下的所有 .md 文件
    for root, dirs, files in os.walk(md_dir):
        for file in files:
            if file.endswith(".md"):
                md_path = os.path.join(root, file)
                try:
                    with open(md_path, "r", encoding="utf-8") as f:
                        clean_text = sanitize_markdown(f.read())
                    
                    # 按照 400 字滑动窗口进行语义切片，重叠 50 字以保证语义完整性
                    chunk_size = 400
                    overlap = 50
                    
                    import sys
                    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                    from utils.text_splitter import semantic_slice_text
                    
                    chunks = semantic_slice_text(clean_text, chunk_size, overlap)
                    chunk_idx = 1
                    for chunk in chunks:
                        # 使用文件名作为关联实体名
                        entity_name = os.path.splitext(file)[0][:20]
                        md_facts.append({
                            "entity_name": entity_name,
                            "source": f"refs:《{os.path.splitext(file)[0]}-第{chunk_idx}段》",
                            "context": chunk,
                            "category": "临床诊疗"
                        })
                        chunk_idx += 1
                except Exception as e:
                    print(f"[Error] Failed to parse MD {file}: {e}")
                    
    print(f"Successfully processed and sliced {len(md_facts)} chunks from local Markdowns.")
    return md_facts

def load_medical_cache_db(cache_db_path: str):
    """
    从本地 APIGateway 的缓存数据库中，加载爬取缓存的真实药物与文献数据。
    这是非循环验证下补齐物理背景知识的关键。
    """
    if not os.path.exists(cache_db_path):
        print(f"[Info] Medical cache DB '{cache_db_path}' does not exist.")
        return []
        
    print(f"Reading records from medical cache DB: '{cache_db_path}'...")
    cache_facts = []
    try:
        conn = sqlite3.connect(cache_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 确认表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='api_cache';")
        if not cursor.fetchone():
            print("[Warning] Table 'api_cache' not found in cache DB.")
            conn.close()
            return []
            
        cursor.execute("SELECT query, service_name, response_json FROM api_cache;")
        rows = cursor.fetchall()
        for row in rows:
            query = row["query"]
            resp_json = row["response_json"]
            
            if not resp_json:
                continue
                
            try:
                items = json.loads(resp_json)
                if not isinstance(items, list):
                    continue
                for item in items:
                    source = item.get("source", "Refs: 外部公开信息")
                    context = item.get("context", "")
                    category = item.get("category", "药品信息")
                    
                    if not context:
                        continue
                        
                    # 避免导入“网络异常”或“抓取异常”的无效无用缓存，确保数据质量
                    if "抓取异常" in source or "未收录" in context:
                        continue
                        
                    cache_facts.append({
                        "entity_name": query[:20],
                        "source": source,
                        "context": sanitize_markdown(context),
                        "category": category
                    })
            except Exception as parse_err:
                continue
                
        conn.close()
        print(f"Successfully loaded {len(cache_facts)} items from medical cache DB.")
        return cache_facts
    except Exception as e:
        print(f"[Error] Failed to read medical cache DB: {e}")
        return []

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
            "entity_name": "甘露特钠胶囊",
            "source": "refs:《甘露特钠胶囊说明书》",
            "context": "【主要成份】本品主要成份为甘露特钠。【辅料】玉米淀粉、滑石粉、硬脂酸镁；明胶空心胶囊。【适应症】用于轻度至中度阿尔茨海默病，改善患者认知功能。【禁忌】对本品主要成份或任何辅料过敏者禁用。",
            "category": "用药方案"
        },
        {
            "entity_name": "葶苈大枣泻肺汤",
            "source": "refs:《金匮要略-肺痿肺痈咳嗽上气病脉证治》",
            "context": "【方剂出处】汉·张仲景《金匮要略》：“肺痈喘不得卧，葶苈大枣泻肺汤主之。”【组成用法】葶苈子（熬令黄色，捣丸如弹子大）一枚，大枣十二枚。以水三升，先煮大枣取二升，去滓，纳葶苈，煮取一升，顿服。【功能主治】泻肺祛痰，下气定喘。主治肺痈喘不得卧，或面目浮肿，胸胁胀满，痰涎壅盛。",
            "category": "用药方案"
        },
        {
            "entity_name": "葶苈子",
            "source": "refs:《中华人民共和国药典-中药材》",
            "context": "【药材性状】本品呈扁卵形。表面黄棕色或红棕色。气微，味微辛、苦，略带粘性。【功能与主治】泻肺平喘，利水消肿。用于痰涎壅盛，喘咳不得卧，水肿，小便不利，胸腹积水。【用法与用量】3～9g，包煎。",
            "category": "临床诊疗"
        },
        {
            "entity_name": "方剂",
            "source": "refs:《中医学基础理论》",
            "context": "【方剂定义】方剂是中医在辨证审因确定治法的基础上，按照组方原则（君臣佐使），选择合适的中药，酌定用量及剂型，妥善配伍而成的治疗处方，是中医临床用药的主要形式。",
            "category": "临床诊疗"
        },
        {
            "entity_name": "过敏",
            "source": "refs:《临床医学免疫学基础》",
            "context": "【病理机制】过敏反应又称变态反应，是指已免疫的机体再次接受相同抗原刺激时所发生的反应。其特征为反应迅速、消退较快，一般不破坏组织细胞，但可引起严重的生理功能紊乱，甚至过敏性休克。",
            "category": "安全禁忌"
        },
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
            "context": "【不良反应】剂量增加时，可能导致外周神经炎、四肢感觉异常、反射消失、肌肉轻瘫 and 精神失常。用药期间应定期复查肝肾功能 and 血常规，当出现肝功能异常时应随时监测。",
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
            category TEXT,
            icd_code TEXT,
            standard_days TEXT
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
                icd_code UNINDEXED,
                standard_days UNINDEXED,
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
        icd_code = item.get("icd_code")
        standard_days = item.get("standard_days")
        
        # 写入普通表
        cursor.execute("""
            INSERT INTO local_rag_index (entity_name, source, context, category, icd_code, standard_days)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (item["entity_name"], item["source"], clean_context, item["category"], icd_code, standard_days))
        
        # 写入倒排检索表
        if fts_ok:
            cursor.execute("""
                INSERT INTO local_rag_fts_index (entity_name, source, context, category, icd_code, standard_days)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (item["entity_name"], item["source"], clean_context, item["category"], icd_code, standard_days))
        
        contexts.append(clean_context)
        
    conn.commit()
    conn.close()
    print("Database SQLite writing completed.")
    
    vector_enabled = os.getenv("LOCAL_RAG_VECTOR_ENABLED", "true").lower() in ("true", "1")
    if not DEPS_OK or not vector_enabled:
        print(f"\n[Warning] FAISS or SentenceTransformer not available or Vector mode disabled (LOCAL_RAG_VECTOR_ENABLED={os.getenv('LOCAL_RAG_VECTOR_ENABLED')}). Skipping vector indexing.")
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
    
    # 2.5 读取本地已净化的临床路径 Markdown 文件
    md_dir = os.path.normpath(LOCAL_MD_DIR)
    all_facts.extend(load_local_markdowns(md_dir))
    
    # 2.8 从 APIGateway 爬取缓存库（medical_cache.db）中导入真实外部知识
    cache_db_path = os.path.join(workspace, "medical_cache.db")
    all_facts.extend(load_medical_cache_db(cache_db_path))
    
    # 2.9 下载并导入 GitHub 开源药品说明书数据库
    drug_file = os.path.join(workspace, OPEN_SOURCE_DRUG_FILE)
    if download_open_source_drugs(drug_file):
        all_facts.extend(load_open_source_drugs(drug_file))
        
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
