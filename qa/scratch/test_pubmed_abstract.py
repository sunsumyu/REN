# -*- coding: utf-8 -*-
import asyncio
import sys
import os

# 将项目根目录添加到 python 搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.api_gateway import APIGatewayService

async def test():
    service = APIGatewayService(db_dir=".")
    # 测试检索一个词，例如 "3/4级血胆红素升高"
    term = "3/4级血胆红素升高"
    print(f"Testing fetch_pubmed_abstracts for term: '{term}'...")
    
    # 绕过缓存进行测试，可以直接修改或临时调用内部逻辑，
    # 或者我们临时删除该实体的缓存再检索。
    # 让我们通过清理特定的缓存，再调用搜索
    import sqlite3
    conn = sqlite3.connect("medical_cache.db")
    cursor = conn.cursor()
    cache_key = service._get_cache_key("pubmed", term)
    cursor.execute("DELETE FROM api_cache WHERE cache_key = ?", (cache_key,))
    conn.commit()
    conn.close()
    print("Deleted old cache for term.")
    
    results = await service.fetch_pubmed_abstracts(term, limit=1)
    if not results:
        print("No results returned.")
        return
        
    for idx, item in enumerate(results):
        print(f"Result [{idx+1}]:")
        print(f"  Source: {item['source']}")
        print(f"  Context:")
        print(item['context'])
        print(f"  Abstract Metadata: {item['metadata'].get('abstract')}")

if __name__ == "__main__":
    asyncio.run(test())
