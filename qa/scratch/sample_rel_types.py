# -*- coding: utf-8 -*-
"""
采样图谱关系类型分布脚本
目的：了解知识图谱中实际出现的 relationship 字段值，
     用于校准 pipeline_workflow.py 中图谱维度丰富度网关的 _HIGH_VALUE_REL_TYPES 集合。
"""
import asyncio
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api_client import APIClient
import logging
logging.disable(logging.CRITICAL)  # 关闭日志避免干扰输出


async def main():
    client = APIClient()
    rel_counter = Counter()
    entity_profiles = []  # 记录每次抽取到的实体+其所有关系

    print("开始采样图谱关系类型... (采样 30 次)")
    for i in range(30):
        try:
            graph_data = await client.fetch_random_knowledge_graph(count=1)
            rels = graph_data.get("relationships", [])
            entities = graph_data.get("entities", [])
            entity_name = entities[0].get("name", "未知") if entities else "未知"
            rel_types = [r.get("relationship", "").strip() for r in rels]
            unique_types = list(set(rel_types))
            for rt in unique_types:
                rel_counter[rt] += 1
            entity_profiles.append({
                "entity": entity_name,
                "rel_count": len(rels),
                "unique_rel_types": unique_types
            })
        except Exception as e:
            print(f"  第{i+1}次采样失败: {e}")
        await asyncio.sleep(0.2)

    print("\n===== 各实体的关系类型分布 =====")
    for p in entity_profiles:
        print(f"  [{p['entity']}] → {p['unique_rel_types']}")

    print("\n===== 所有关系类型出现次数（Top 50）=====")
    for rel_type, count in rel_counter.most_common(50):
        print(f"  {count:3d}次  '{rel_type}'")

    await client.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
