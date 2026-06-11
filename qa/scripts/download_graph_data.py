# -*- coding: utf-8 -*-
"""
从远端图谱 API 全量拉取实体和关系，并下载至本地 local_rag.db 中。
通过在 random 请求时控制 exclude_ids，避免 414 错误并最大限度去重拉取。
"""

import asyncio
import json
import sqlite3
import re
import sys
import os
import httpx

# 确保能引入项目中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_client import APIClient

def sanitize_markdown(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

async def download_graph_data():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(workspace_dir, "local_rag.db")
    
    print("==========================================================")
    print("🚀 开始连接远端图数据库进行全量实体与关系下载...")
    print("==========================================================")
    
    client = APIClient()
    all_entities = {}
    all_relationships = {}
    
    # 设定循环捞取次数，以获取尽可能多的实体与关系
    max_loops = 200
    consecutive_empty = 0
    
    for i in range(max_loops):
        # 为了防止 GET 参数超长 (414 Too Long)，我们只保留最近拉取到的 200 个 ID 进行排除
        exclude_list = list(all_entities.keys())[-200:]
        exclude_str = ",".join(exclude_list)
        
        print(f"-> 正在进行第 {i+1}/{max_loops} 次随机游走拉取 (去重实体: {len(all_entities)}, 关系: {len(all_relationships)})...")
        try:
            params = {
                "count": 45,
                "knowledgeBaseId": 201,
                "hopCount": 2
            }
            if exclude_str:
                params["entityIds"] = exclude_str
                
            resp = await client.httpx_client.get(
                "https://ai.yzint.cn/api/knowledge/v1/graph/entity/random",
                params=params,
                timeout=20.0
            )
            
            if resp.status_code == 200:
                resp_data = resp.json()
                if resp_data.get("success"):
                    data = resp_data.get("data", {})
                    entities = data.get("entities", [])
                    relationships = data.get("relationships", [])
                    
                    if not entities and not relationships:
                        consecutive_empty += 1
                        if consecutive_empty >= 5:
                            print("连续多次未返回新数据，下载结束。")
                            break
                        continue
                        
                    consecutive_empty = 0
                    
                    # 收集实体
                    for ent in entities:
                        ent_id = str(ent.get("id"))
                        if ent_id not in all_entities:
                            all_entities[ent_id] = {
                                "name": ent.get("name"),
                                "type": ent.get("type"),
                                "description": ent.get("description", "暂无描述")
                            }
                            
                    # 收集关系
                    for rel in relationships:
                        src_name = rel.get("sourceName")
                        tgt_name = rel.get("targetName")
                        rel_type = rel.get("relationship")
                        desc = rel.get("description", "").strip() or "暂无详细描述"
                        
                        if src_name and tgt_name and rel_type:
                            rel_key = f"{src_name}-{tgt_name}-{rel_type}"
                            if rel_key not in all_relationships:
                                all_relationships[rel_key] = {
                                    "sourceName": src_name,
                                    "targetName": tgt_name,
                                    "relationship": rel_type,
                                    "description": desc
                                }
                else:
                    print(f"接口返回失败: {resp_data.get('msg')}")
            else:
                print(f"接口报错，状态码: {resp.status_code}")
                
        except Exception as e:
            print(f"第 {i+1} 次拉取异常: {e}")
            await asyncio.sleep(1.0)
            
        await asyncio.sleep(0.1)
        
    print(f"\n下载结束。共收集到 {len(all_entities)} 个去重实体，{len(all_relationships)} 条关系。")
    if not all_entities and not all_relationships:
        await client.close()
        return
        
    # 保存备份
    backup_path = os.path.join(workspace_dir, "graph_downloaded.json")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump({
            "entities": list(all_entities.values()),
            "relationships": list(all_relationships.values())
        }, f, ensure_ascii=False, indent=2)
    print(f"本地备份已写入: {backup_path}")
    
    # 写入 SQLite 数据库
    print("正在导入到 local_rag.db...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取现有库中已存在的 sources
    cursor.execute("SELECT source FROM local_rag_index;")
    existing_sources = {row[0] for row in cursor.fetchall()}
    
    imported_count = 0
    
    # 1. 导入实体
    for ent in all_entities.values():
        name = ent["name"]
        desc = ent["description"].strip() or "暂无详细描述"
        source_name = f"refs:《实体库:{name}》"
        
        if source_name in existing_sources:
            continue
            
        context_text = f"概念定义: {name} (类型: 医疗实体) - {desc}"
        
        try:
            cursor.execute("""
                INSERT INTO local_rag_index (entity_name, source, context, category, icd_code, standard_days)
                VALUES (?, ?, ?, '实体概念', 'N/A', 'N/A');
            """, (name, source_name, context_text))
            
            try:
                cursor.execute("""
                    INSERT INTO local_rag_fts_index (source, context, entity_name, category, icd_code, standard_days)
                    VALUES (?, ?, ?, '实体概念', 'N/A', 'N/A');
                """, (source_name, context_text, name))
            except Exception:
                pass
            imported_count += 1
        except Exception as e:
            pass
            
    # 2. 导入关系
    for rel in all_relationships.values():
        src = rel["sourceName"]
        tgt = rel["targetName"]
        rtype = rel["relationship"]
        desc = rel["description"]
        
        source_name = f"refs:《图谱关系:{src}-{tgt}-{rtype}》"
        if source_name in existing_sources:
            continue
            
        context_text = f"知识关联: 【{src}】与【{tgt}】存在【{rtype}】关系。说明：{desc}"
        
        try:
            # 将关系实体名称也保存，利于 entity_name 精准对撞
            cursor.execute("""
                INSERT INTO local_rag_index (entity_name, source, context, category, icd_code, standard_days)
                VALUES (?, ?, ?, '知识关联', 'N/A', 'N/A');
            """, (src, source_name, context_text))
            
            try:
                cursor.execute("""
                    INSERT INTO local_rag_fts_index (source, context, entity_name, category, icd_code, standard_days)
                    VALUES (?, ?, ?, '知识关联', 'N/A', 'N/A');
                """, (source_name, context_text, src))
            except Exception:
                pass
            imported_count += 1
        except Exception as e:
            pass
            
    conn.commit()
    conn.close()
    print(f"🎉 成功同步导入 {imported_count} 条图谱事实到 local_rag.db！")
    
    await client.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(download_graph_data())
