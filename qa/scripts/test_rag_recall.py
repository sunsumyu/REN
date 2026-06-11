# -*- coding: utf-8 -*-
"""
测试脚本：从本地向量库中召回数据并打印查询词和返回内容
"""
import sys
import os
import asyncio
import argparse

# 确保能将当前项目根目录加入环境变量，以便导入 retrieval 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from retrieval.local_rag import LocalRAGService

async def test_recall(query: str, entity_name: str):
    print(f"==================================================")
    print(f"🔍 启动 RAG 向量检索测试")
    print(f"==================================================")
    print(f"查询问题 (Query)    : {query}")
    print(f"核心实体 (Entity)   : {entity_name}")
    print(f"--------------------------------------------------\n")
    
    # 初始化本地私有 RAG 模块（会自动加载 sqlite 和 faiss 索引）
    rag = LocalRAGService(workspace_dir=".")
    
    try:
        # 执行检索
        results = await rag.search(query, entity_name)
        
        if not results:
            print("❌ 未能从本地知识库召回任何相关数据。")
            return
            
        print(f"✅ 成功召回 {len(results)} 条相关内容：\n")
        
        for idx, res in enumerate(results):
            # 获取检索的元数据（包含得分和检索通道）
            score = res.metadata.get('similarity_score', 'N/A')
            method = res.metadata.get('retrieval_method', '未知方式')
            
            if isinstance(score, float):
                score_str = f"{score:.4f}"
            else:
                score_str = str(score)
                
            print(f"=== 🏆 结果 [{idx + 1}] | 相似度得分: {score_str} | 检索通道: {method} ===")
            print(f"【文献来源】: {res.source}")
            print(f"【正文片段】:\n{res.context}\n")
            
    finally:
        # 释放资源
        rag.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test local vector RAG retrieval.")
    # 默认值设置为您刚刚正在查看的寻常型天疱疮
    parser.add_argument("-q", "--query", type=str, default="寻常型天疱疮的临床表现和皮损特征是什么？", help="要查询的问题")
    parser.add_argument("-e", "--entity", type=str, default="寻常型天疱疮", help="用于 SQL 退回检索的核心实体名")
    
    args = parser.parse_args()
    
    # 因为检索模块是基于异步 asyncio 编写的，所以在此使用 asyncio.run 执行
    asyncio.run(test_recall(args.query, args.entity))
