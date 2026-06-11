# -*- coding: utf-8 -*-
"""
验证脚本：测试优化后的双轨 Hybrid RAG + RRF 检索效果
"""
import sys
import os
import asyncio

# 确保工作区根目录在 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.local_rag import LocalRAGService

async def main():
    rag = LocalRAGService(workspace_dir=".")
    
    print("==================================================")
    print("🔍 开始验证双轨 Hybrid RAG + RRF 检索器效果")
    print("==================================================\n")
    
    # 测试用例 1: 七氟烷 (特定药物)
    query_1 = "七氟烷的用法用量和吸入浓度是多少？"
    entity_1 = "七氟烷"
    print(f"--- 测试用例 1 (药物查询) ---")
    print(f"查询: '{query_1}'")
    print(f"限定实体: '{entity_1}'")
    
    results_1 = await rag.search(query_1, entity_1)
    if not results_1:
        print("❌ 未能召回任何相关内容 (可能需要先重建索引)\n")
    else:
        print(f"✅ 成功召回 {len(results_1)} 条内容:")
        for idx, res in enumerate(results_1):
            score = res.metadata.get('similarity_score', 'N/A')
            method = res.metadata.get('retrieval_method', '未知')
            print(f"  [{idx + 1}] 来源: {res.source}")
            print(f"      检索方式: {method} | 相似度得分: {score}")
            print(f"      完整正文内容:\n{res.context.strip()}\n")

    # 测试用例 2: 寻常型天疱疮 (常见疾病)
    query_2 = "寻常型天疱疮有什么典型的临床表现和皮损特征？"
    entity_2 = "寻常型天疱疮"
    print(f"--- 测试用例 2 (疾病查询) ---")
    print(f"查询: '{query_2}'")
    print(f"限定实体: '{entity_2}'")
    
    results_2 = await rag.search(query_2, entity_2)
    if not results_2:
        print("❌ 未能召回任何相关内容\n")
    else:
        print(f"✅ 成功召回 {len(results_2)} 条内容:")
        for idx, res in enumerate(results_2):
            score = res.metadata.get('similarity_score', 'N/A')
            method = res.metadata.get('retrieval_method', '未知')
            print(f"  [{idx + 1}] 来源: {res.source}")
            print(f"      检索方式: {method} | 相似度得分: {score}")
            print(f"      完整正文内容:\n{res.context.strip()}\n")
            
    rag.close()

if __name__ == "__main__":
    asyncio.run(main())
