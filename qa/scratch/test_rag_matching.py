ng: utf-8 -*-
"""
RAG 向量库及检索匹配机制自诊断测试脚本
通过读取 medical_qa_dataset.jsonl 中已有的问答对与事实合约，
模拟多种 RAG 检索策略（标准硬过滤、无过滤、降低阈值等），
分析找回率并生成诊断报告，避免消耗过多大模型 Token。
"""

import os
import json
import asyncio
import re
import sys
from dotenv import load_dotenv

# 确保能导入项目中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.local_rag import LocalRAGService

def extract_entity_name(source: str, context: str) -> str:
    # 从 source 中提取实体名
    if "实体库:" in source:
        m = re.search(r'实体库:([^》\s]+)', source)
        if m:
            return m.group(1)
    elif "图谱关系:" in source:
        m = re.search(r'图谱关系:([^》\s]+)', source)
        if m:
            parts = m.group(1).split("-")
            return parts[0] if parts else m.group(1)
    elif "常见临床疾病与合理用药诊疗路径-" in source:
        m = re.search(r'常见临床疾病与合理用药诊疗路径-([^》\s]+)', source)
        if m:
            return m.group(1)
            
    # 从 context 中提取
    if context.startswith("概念定义:"):
        m = re.search(r'概念定义:\s*([^\s(（]+)', context)
        if m:
            return m.group(1)
    elif context.startswith("知识关联:"):
        m = re.search(r'【([^】]+)】', context)
        if m:
            return m.group(1)
            
    return ""

def is_rag_hit(res, exp) -> bool:
    """
    非循环验证下的 RAG 找回成功判定逻辑。
    因为 RAG 库使用的是原始权威源 medical.json，而数据集里的 source 指向旧的图数据库（如 refs:《实体库:xxx》），
    二者 source 字段不可能字面相等。
    如果检索出的结果对应的实体与期望实体一致，或者检索出的内容与期望内容有实质性重叠，则判定为召回成功。
    """
    exp_entity = exp.get("entity_name", "").strip()
    
    res_entity = ""
    if hasattr(res, "metadata") and isinstance(res.metadata, dict):
        res_entity = res.metadata.get("entity_name", "").strip()
    if not res_entity and hasattr(res, "entity_name"):
        res_entity = res.entity_name
        
    def clean_name(n):
        if not n:
            return ""
        n = n.lower()
        n = re.sub(r'[\s\-~_\(\)（）\+®™/\.\,，。]', '', n)
        suffixes = ["片", "胶囊", "注射液", "口服溶液", "口服液", "胶浆", "颗粒", "软膏", "凝胶", "滴眼液", "泡腾片", "贴膏"]
        for suf in suffixes:
            if n.endswith(suf) and len(n) > len(suf):
                n = n[:-len(suf)]
        return n
        
    if exp_entity and res_entity:
        c_exp = clean_name(exp_entity)
        c_res = clean_name(res_entity)
        if c_exp and c_res and (c_exp in c_res or c_res in c_exp):
            return True
            
    exp_context = exp.get("context_preview", "").strip()
    res_context = res.context.strip() if hasattr(res, "context") else ""
    
    if exp_context and res_context:
        c_exp_ctx = re.sub(r'[^\u4e00-\u9fa5\w]', '', exp_context)
        c_res_ctx = re.sub(r'[^\u4e00-\u9fa5\w]', '', res_context)
        if c_exp_ctx and c_res_ctx:
            if c_exp_ctx in c_res_ctx or c_res_ctx in c_exp_ctx:
                return True
                
        if "-" in exp_entity:
            parts = [clean_name(p) for p in exp_entity.split("-") if p.strip()]
            if len(parts) >= 2:
                if all(p in c_res_ctx for p in parts):
                    return True
                    
    return False

async def main():
    # 1. 加载环境变量与配置
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(workspace_dir, "medical_qa_dataset.jsonl")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    print("Initializing LocalRAGService...")
    local_rag = LocalRAGService(workspace_dir=workspace_dir)
    print(f"Local RAG Status - Vector Enabled: {local_rag.vector_enabled}")
    print(f"Similarity Threshold in .env: {local_rag.similarity_threshold}")

    # 读取数据集前 322 行 (或者全量，按需调整)
    max_lines = 322
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx > max_lines:
                break
            try:
                data = json.loads(line.strip())
                records.append((idx, data))
            except Exception as e:
                print(f"Error parsing line {idx}: {e}")

    print(f"Loaded {len(records)} records for analysis.")

    # 统计指标
    stats = {
        "total_queries": 0,
        "total_expected_facts": 0,
        "strategy_standard_hits": 0,  # 策略 A: 实体硬匹配 + env 阈值
        "strategy_no_entity_filter_hits": 0, # 策略 B: 无实体硬匹配 + env 阈值
        "strategy_relaxed_hits": 0,  # 策略 C: 无实体硬匹配 + 降低阈值 (0.45)
        "failures": []
    }

    for line_idx, record in records:
        q = record.get("Q", "").strip()
        evidence_contract = record.get("evidence_contract", {})
        facts = evidence_contract.get("facts", []) if isinstance(evidence_contract, dict) else []
        
        if not q or not facts:
            continue
            
        stats["total_queries"] += 1
        
        # 提取期望找回的 facts（仅限本地导入的 RAG 数据，剔除外部 PubMed 和网页抓取）
        expected_sources = []
        for fact in facts:
            source = fact.get("source", "")
            context_preview = fact.get("context_preview", "")
            entity_name = extract_entity_name(source, context_preview)
            
            # 过滤条件：只取本地导入的实体库、说明书或临床路径 refs，排除 PubMed 和在线搜索引擎抓取数据
            if source and not any(x in source for x in ["PubMed", "在线公开", "抓取异常"]):
                expected_sources.append({
                    "source": source,
                    "entity_name": entity_name,
                    "context_preview": context_preview
                })
                stats["total_expected_facts"] += 1

        if not expected_sources:
            continue

        # 执行测试检索
        for exp in expected_sources:
            target_source = exp["source"]
            entity_name = exp["entity_name"]
            
            # --- 策略 A: 标准检索 ---
            # 直接调用 search(q, entity_name)
            results_standard = await local_rag.search(q, entity_name)
            hit_standard = any(is_rag_hit(res, exp) for res in results_standard)
            if hit_standard:
                stats["strategy_standard_hits"] += 1

            # --- 策略 B: 无实体硬匹配 ---
            results_no_ent = await local_rag.search(q, "")
            hit_no_ent = any(is_rag_hit(res, exp) for res in results_no_ent)
            if hit_no_ent:
                stats["strategy_no_entity_filter_hits"] += 1

            # --- 策略 C: 降低阈值并无实体过滤 ---
            # 临时修改阈值以执行模拟
            original_threshold = local_rag.similarity_threshold
            local_rag.similarity_threshold = 0.40
            results_relaxed = await local_rag.search(q, "")
            local_rag.similarity_threshold = original_threshold
            
            hit_relaxed = any(is_rag_hit(res, exp) for res in results_relaxed)
            if hit_relaxed:
                stats["strategy_relaxed_hits"] += 1

            # 记录失败详情，便于深度分析
            if not hit_standard:
                stats["failures"].append({
                    "line": line_idx,
                    "query": q,
                    "target_entity": entity_name,
                    "target_source": target_source,
                    "hit_standard": hit_standard,
                    "hit_no_entity": hit_no_ent,
                    "hit_relaxed": hit_relaxed
                })

    # 输出结果报告
    total_expected = stats["total_expected_facts"]
    print("\n" + "="*50)
    print("📋 RAG VECTOR MATCHING DIAGNOSTIC REPORT")
    print("="*50)
    print(f"Total Unique Queries Evaluated : {stats['total_queries']}")
    print(f"Total Expected Facts to Retrieve: {total_expected}")
    print("-"*50)
    
    rec_std = (stats["strategy_standard_hits"] / total_expected) * 100 if total_expected else 0
    rec_no_ent = (stats["strategy_no_entity_filter_hits"] / total_expected) * 100 if total_expected else 0
    rec_rel = (stats["strategy_relaxed_hits"] / total_expected) * 100 if total_expected else 0
    
    print(f"Strategy A (Standard: Entity Match + Env Threshold {original_threshold}) Recall: {rec_std:.2f}% ({stats['strategy_standard_hits']}/{total_expected})")
    print(f"Strategy B (No Entity Filtering + Env Threshold {original_threshold}) Recall   : {rec_no_ent:.2f}% ({stats['strategy_no_entity_filter_hits']}/{total_expected})")
    print(f"Strategy C (No Entity Filtering + Relaxed Threshold 0.40) Recall      : {rec_rel:.2f}% ({stats['strategy_relaxed_hits']}/{total_expected})")
    print("="*50)
    
    # 打印前 10 个典型失败案例并分析
    print("\n🔍 TOP 10 TYPICAL FAILURE CASES FOR ANALYSIS:")
    failures_to_print = stats["failures"][:10]
    for i, fail in enumerate(failures_to_print, 1):
        print(f"\n[{i}] Dataset Line: {fail['line']}")
        print(f"  - Query        : '{fail['query']}'")
        print(f"  - Target Entity: '{fail['target_entity']}'")
        print(f"  - Target Source: '{fail['target_source']}'")
        print(f"  - Hit Standard ?: {fail['hit_standard']}")
        print(f"  - Hit No-Entity?: {fail['hit_no_entity']}")
        print(f"  - Hit Relaxed  ?: {fail['hit_relaxed']}")
        
        # 给出诊断建议
        if not fail["hit_standard"] and fail["hit_no_entity"]:
            print("  💡 DIAGNOSIS: Blocked strictly by ENTITY FILTERING. The source matches textually but the entity name constraint caused a mismatch.")
        elif not fail["hit_no_entity"] and fail["hit_relaxed"]:
            print("  💡 DIAGNOSIS: Blocked by SIMILARITY THRESHOLD. The embedding score is between 0.40 and 0.60.")
        else:
            print("  💡 DIAGNOSIS: MISSED ENTIRELY. Not found in FAISS/FTS top-15, or similarity was below 0.40. Missing reference or vector index mismatch.")

    # 导出诊断日志到 docs
    report_output_path = os.path.join(workspace_dir, "docs", "rag_matching_diagnostic_report.md")
    
    markdown_content = f"""# 📊 RAG 检索找回率自诊断评估报告

本报告对本地私有 RAG 模块（Tier 1）在数据集第 1 到 {max_lines} 行中的找回率进行全量评估，分析检索漏检的根源，并指导策略调优。

## 📈 评估量化结果

- **评估总问题数**: {stats['total_queries']}
- **期望找回的事实条数**: {total_expected}

| 检索策略组合 | 找回成功数 (Hits) | 找回率 (Recall) | 瓶颈诊断 |
| :--- | :---: | :---: | :--- |
| **策略 A (标准：实体硬过滤 + .env 阈值 {original_threshold})** | {stats['strategy_standard_hits']} | **{rec_std:.2f}%** | 线上主干策略，存在较强的实体过滤截断漏检 |
| **策略 B (宽松：无实体过滤 + .env 阈值 {original_threshold})** | {stats['strategy_no_entity_filter_hits']} | **{rec_no_ent:.2f}%** | 解除实体包含截断后的基线召回表现 |
| **策略 C (极简：无实体过滤 + 降低阈值 0.40)** | {stats['strategy_relaxed_hits']} | **{rec_rel:.2f}%** | 解除实体截断并放宽相似度门禁后的理论召回上限 |

## 🔍 典型漏检案例诊断与治理建议
"""
    for i, fail in enumerate(stats["failures"][:20], 1):
        diagnosis = ""
        if not fail["hit_standard"] and fail["hit_no_entity"]:
            diagnosis = "🔴 **实体过滤硬阻断**：文档实际在 Top-15 候选里，但因为传入的 `entity_name` 没能和文档里的 `entity_name` 产生互包含匹配而被硬过滤直接抛弃。"
        elif not fail["hit_no_entity"] and fail["hit_relaxed"]:
            diagnosis = "⚠️ **相似度门禁偏高**：向量相似度落在了 `0.40` 到 `{original_threshold}` 之间，被 `.env` 中的高阈值拦截。"
        else:
            diagnosis = "❌ **完全漏检**：文本在 FAISS 和 FTS5 联合召回的 Top-15 之外，或者相似度极其低（< 0.40），可能因为向量表征存在偏差或本地库本身未录入相应知识。"
            
        markdown_content += f"""
### ❌ 案例 {i} (数据集第 {fail['line']} 行)
- **大模型提问 (Query)**: `{fail['query']}`
- **目标实体 (Entity)**: `{fail['target_entity']}`
- **目标事实源 (Source)**: `{fail['target_source']}`
- **诊断分析**: {diagnosis}
"""
    
    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"\n💾 Diagnostic markdown report written to: {report_output_path}")

    local_rag.close()

if __name__ == "__main__":
    asyncio.run(main())
