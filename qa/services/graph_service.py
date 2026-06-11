# -*- coding: utf-8 -*-
import logging
import re
from typing import Dict, Any, Set
import config

logger = logging.getLogger("MedicalQA.GraphService")

from abc import ABC, abstractmethod

class IGraphService(ABC):
    @abstractmethod
    async def fetch_random_knowledge_graph(self, count: int = config.DEFAULT_ENTITY_COUNT, kb_id: int = config.DEFAULT_KNOWLEDGE_BASE_ID, hop_count: int = config.DEFAULT_HOP_COUNT) -> Dict[str, Any]:
        pass

def is_empty_or_placeholder(val: Any) -> bool:
    """
    判断一个值是否为无效的占位符/空值（包含 None, 0, 0.0, 以及空字符串）。
    """
    if val is None:
        return True
    if isinstance(val, (int, float)) and val == 0:
        return True
    if isinstance(val, str):
        return not val.strip()
    return False

def is_medical_entity(entity: Dict[str, Any]) -> bool:
    """
    判断实体是否是合法医疗实体（过滤掉软件缺陷测试等非医疗脏数据）。
    """
    name = str(entity.get("name", "")).strip()
    etype = str(entity.get("type", "")).strip().lower()
    desc = str(entity.get("description", "")).strip()
    
    # 1. 检查黑名单类型
    dirty_types = {"defect", "module", "version", "task", "project", "test_case", "bug", "issue"}
    if etype in dirty_types:
        return False
        
    # 2. 检查黑名单特异性中英文关键词（针对 MeterSphere、缺陷测试、发版评估等脏数据）
    dirty_keywords = [
        "metersphere", "skill ui", "发版风险", "发版评估", "缺陷评估", 
        "缺陷清单", "奇门易知", "智能体创建", "智能体相关缺陷", 
        "自建skill", "缺陷登记", "缺陷相关问题", "缺陷等级", "测试用例"
    ]
    name_lower = name.lower()
    desc_lower = desc.lower()
    
    for kw in dirty_keywords:
        if kw in name_lower or kw in desc_lower:
            return False
            
    # 3. 正则匹配独立的英文词 bug/defect
    if re.search(r'\b(bug|bugs|defect|defects)\b', name_lower) or re.search(r'\b(bug|bugs|defect|defects)\b', desc_lower):
        return False
        
    return True

def merge_records(r1: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, Any]:
    """
    智能合并两个属性字典，确保在去重的同时，最大化保留有效且完整的数据。
    """
    merged = {}
    all_keys = set(r1.keys()).union(r2.keys())
    for k in all_keys:
        v1 = r1.get(k)
        v2 = r2.get(k)
        
        empty1 = is_empty_or_placeholder(v1)
        empty2 = is_empty_or_placeholder(v2)
        
        if empty1 and not empty2:
            merged[k] = v2
        elif empty2 and not empty1:
            merged[k] = v1
        elif empty1 and empty2:
            # 两个都是空值/占位符，优先保留非 None 的值
            merged[k] = v1 if v1 is not None else v2
        else:
            # 两个都是有效值，进行去重及去噪合并
            s1, s2 = str(v1).strip(), str(v2).strip()
            if k in {"description", "relationship"}:
                if s1.replace(" ", "") == s2.replace(" ", ""):
                    merged[k] = v1 if len(s1) < len(s2) else v2
                else:
                    merged[k] = v1 if len(s1) > len(s2) else v2
            else:
                # 其它类型字段，优先保留字符串表示较长（信息更全）的
                merged[k] = v1 if len(s1) >= len(s2) else v2
    return merged


class GraphService(IGraphService):
    def __init__(self, llm_service):
        # We inject the LLMService (which wraps the Async HTTP client)
        self.llm_service = llm_service
        self.seen_entity_ids: Set[str] = set()

    async def fetch_random_knowledge_graph(
        self, 
        count: int = config.DEFAULT_ENTITY_COUNT,
        kb_id: int = config.DEFAULT_KNOWLEDGE_BASE_ID,
        hop_count: int = config.DEFAULT_HOP_COUNT
    ) -> Dict[str, Any]:
        """
        Fetch a random medical subgraph. Automatically passes seen entity IDs to exclude them.
        Filters out non-medical dirty data and performs cascading cleanup on relationships.
        Supports up to 3 retry attempts to fill the count of clean entities.
        """
        accumulated_entities = []
        accumulated_relationships = []
        seen_ids_this_request = set()
        
        max_attempts = 3
        attempt = 0
        for attempt in range(max_attempts):
            current_needed = max(1, count - len(accumulated_entities))
            
            # 排除已访问过的实体以及本次多轮请求中已经收集到的 clean 实体，防止重复拉取
            exclude_ids = self.seen_entity_ids.union(seen_ids_this_request)
            exclude_ids_str = ",".join(list(exclude_ids))
            
            params = {
                "count": current_needed,
                "knowledgeBaseId": kb_id,
                "hopCount": hop_count
            }
            if exclude_ids_str:
                params["entityIds"] = exclude_ids_str
                
            logger.info(f"Fetching random KG data (attempt {attempt + 1}/{max_attempts}, count={current_needed}, excluding {len(exclude_ids)} entities)...")
            
            response_data = await self.llm_service._request_with_retry("GET", config.GRAPH_API_URL, params=params)
            
            if not response_data.get("success", False):
                raise Exception(f"Knowledge Graph API failed: {response_data.get('msg', 'Unknown error')}")
                
            graph_data = response_data.get("data", {})
            entities = graph_data.get("entities", [])
            relationships = graph_data.get("relationships", [])
            
            # 1. 智能去重合并当前批次的实体
            unique_entities = {}
            for entity in entities:
                entity_id = str(entity.get("id"))
                if entity_id not in unique_entities:
                    unique_entities[entity_id] = entity
                else:
                    unique_entities[entity_id] = merge_records(unique_entities[entity_id], entity)
            
            deduped_entities = list(unique_entities.values())
            
            # 2. 对当前批次实体进行医疗属性判定过滤，分离干净实体和脏实体
            clean_entities = []
            dirty_entity_ids = set()
            dirty_entity_names = set()
            
            for entity in deduped_entities:
                entity_id = str(entity.get("id"))
                entity_name = str(entity.get("name", "")).strip()
                if is_medical_entity(entity):
                    clean_entities.append(entity)
                    seen_ids_this_request.add(entity_id)
                else:
                    dirty_entity_ids.add(entity_id)
                    if entity_name:
                        dirty_entity_names.add(entity_name)
                    logger.warning(f"Filtered out non-medical dirty entity: id={entity_id}, name='{entity_name}', type='{entity.get('type')}'")
            
            # 3. 去重合并当前批次的关系
            unique_relationships = {}
            for rel in relationships:
                rel_id = rel.get("id")
                if rel_id:
                    rel_key = str(rel_id)
                else:
                    rel_key = f"{rel.get('source')}-{rel.get('target')}-{rel.get('relationship')}"
                    
                if rel_key not in unique_relationships:
                    unique_relationships[rel_key] = rel
                else:
                    unique_relationships[rel_key] = merge_records(unique_relationships[rel_key], rel)
            
            deduped_relationships = list(unique_relationships.values())
            
            # 4. 级联清理当前批次的关系（清除指向被过滤脏实体的关系）
            clean_relationships = []
            for rel in deduped_relationships:
                src_id = str(rel.get("source")) if rel.get("source") is not None else None
                tgt_id = str(rel.get("target")) if rel.get("target") is not None else None
                src_name = str(rel.get("sourceName")).strip() if rel.get("sourceName") is not None else None
                tgt_name = str(rel.get("targetName")).strip() if rel.get("targetName") is not None else None
                
                keep = True
                if src_id in dirty_entity_ids or tgt_id in dirty_entity_ids:
                    keep = False
                if src_name in dirty_entity_names or tgt_name in dirty_entity_names:
                    keep = False
                    
                if keep:
                    clean_relationships.append(rel)
                else:
                    logger.warning(f"Cascading cleaned relationship: {src_name} --({rel.get('relationship')})--> {tgt_name}")
            
            # 5. 累加本次拉取并清洗后的实体和关系
            accumulated_entities.extend(clean_entities)
            accumulated_relationships.extend(clean_relationships)
            
            # 6. 如果满足实体数量 count，提前终止重试
            if len(accumulated_entities) >= count:
                break
                
        # 再次进行整体合并去重（防止跨 attempt 重复）
        final_entities = {}
        for entity in accumulated_entities:
            entity_id = str(entity.get("id"))
            if entity_id not in final_entities:
                final_entities[entity_id] = entity
            else:
                final_entities[entity_id] = merge_records(final_entities[entity_id], entity)
                
        final_relationships = {}
        for rel in accumulated_relationships:
            rel_id = rel.get("id")
            if rel_id:
                rel_key = str(rel_id)
            else:
                rel_key = f"{rel.get('source')}-{rel.get('target')}-{rel.get('relationship')}"
            if rel_key not in final_relationships:
                final_relationships[rel_key] = rel
            else:
                final_relationships[rel_key] = merge_records(final_relationships[rel_key], rel)
                
        # 整理输出图谱数据结构
        graph_data = {
            "entities": list(final_entities.values()),
            "relationships": list(final_relationships.values())
        }
        
        # 将本次返回的干净实体加入 seen_entity_ids 缓存中
        new_entities_count = 0
        for entity in graph_data["entities"]:
            entity_id = str(entity.get("id"))
            if entity_id not in self.seen_entity_ids:
                self.seen_entity_ids.add(entity_id)
                new_entities_count += 1
                
        logger.info(f"Successfully retrieved {len(graph_data['entities'])} clean entities (registered {new_entities_count} new entities), {len(graph_data['relationships'])} clean relationships after {attempt + 1} attempt(s).")
        return graph_data

