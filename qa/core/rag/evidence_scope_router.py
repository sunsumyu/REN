# -*- coding: utf-8 -*-
import logging
from enum import Enum
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger("MedicalQA.EvidenceScopeRouter")

# Attempt to load Faiss and LocalEmbeddingEngine for Scheme 3 Fallback
try:
    import faiss
    from retrieval.local_rag import LocalEmbeddingEngine
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False
    LocalEmbeddingEngine = None

class ScopeType(str, Enum):
    CORE = "CORE"            # 核心证据：必须参与推理，必须体现在最终答案中
    BOUNDARY = "BOUNDARY"    # 边界限制：仅限 answer_body 和 think 都最多一句的窄边界
    BLOCKED = "BLOCKED"      # 强制封锁：即使 RAG 召回了也严禁触碰，从 Prompt 中物理屏蔽，防止反向激活
    UNUSED = "UNUSED"        # 冗余屏蔽：直接从 Prompt 排除，不喂给模型


class EvidenceScopeRouter:
    """证据作用域路由，对检索召回的知识按问答意图进行精细化分级与隔离控制 (Metadata & Semantic Embedding)"""
    
    INTENT_CONFIG = {
        "DOSAGE_LIMIT": {
            "META_CORE": {"entity_type": ["Dosage", "用法用量"], "category": ["剂量", "用法", "用量"]},
            "META_BOUNDARY": {"category": ["不良反应", "副作用", "禁忌"]},
            "META_BLOCKED": {"entity_type": ["Mechanism", "Target", "靶点", "通路"]},
            "ANCHOR": "药品的规定剂量、用法用量、儿童或成人用量、单次或一日极量",
            "BLOCKED_ANCHOR": "微观的分子机制、受体结合信号通路或基因分型靶点"
        },
        "COMPONENT": {
            "META_CORE": {"entity_type": ["Component", "Ingredient", "辅料"], "category": ["成分", "组成", "配方"]},
            "META_BOUNDARY": {"category": ["功效", "主治", "适应症"]},
            "META_BLOCKED": {"category": ["不良反应", "副作用", "禁忌", "禁用"]},
            "ANCHOR": "药品的成分、组成、配方、辅料、配料或包含的化学物质",
            "BLOCKED_ANCHOR": "药品的不良反应、副作用、禁忌和禁用人群"
        },
        "STORAGE": {
            "META_CORE": {"entity_type": ["Storage", "贮藏"], "category": ["贮藏", "储存", "存放", "保存"]},
            "META_BOUNDARY": {"category": ["包装", "规格"]},
            "META_BLOCKED": {"entity_type": ["Mechanism"], "category": ["疗效", "不良反应", "禁用", "禁忌"]},
            "ANCHOR": "药品的贮藏条件、储存温度、存放方式、保存要求",
            "BLOCKED_ANCHOR": "药品的分子机制、疗效、临床试验、不良反应、禁用和禁忌"
        },
        "PACKAGING": {
            "META_CORE": {"entity_type": ["Packaging", "包装"], "category": ["包装", "规格", "每盒", "片数", "装量"]},
            "META_BOUNDARY": {"category": ["储存", "贮藏", "保存"]},
            "META_BLOCKED": {"entity_type": ["Mechanism"], "category": ["疗效", "不良反应", "禁用", "禁忌"]},
            "ANCHOR": "药品的包装、规格、每盒数量、片数、装量或包装材质",
            "BLOCKED_ANCHOR": "药品的分子机制、疗效、不良反应、副作用、禁用和禁忌"
        },
        "CONTRAINDICATION": {
            "META_CORE": {"relationship": ["has_contraindication", "禁忌使用"], "category": ["禁忌", "禁用", "慎用"]},
            "META_BOUNDARY": {"category": ["不良反应", "副作用", "用法", "用量"]},
            "META_BLOCKED": {"entity_type": ["Mechanism", "Pharmacokinetics", "清除率", "半衰期"]},
            "ANCHOR": "药品的绝对禁忌人群、严重过敏反应和禁止联合使用的配伍",
            "BLOCKED_ANCHOR": "分子结构、受体通路、血浆半衰期等深层药代动力学参数"
        },
        "GENERAL_MEDICAL": {
            "META_CORE": {"category": ["机制", "药理", "疗效", "主治", "适应症", "临床试验", "证据"]},
            "META_BOUNDARY": {},
            "META_BLOCKED": {},
            "ANCHOR": "药品的药理机制、临床疗效、主治适应症、临床试验证据",
            "BLOCKED_ANCHOR": ""
        }
    }
    
    def __init__(self):
        self.embedding_model = None
        if VECTOR_AVAILABLE:
            try:
                self.embedding_model = LocalEmbeddingEngine().get_model()
                logger.info("EvidenceScopeRouter successfully loaded LocalEmbeddingEngine for Semantic Routing.")
            except Exception as e:
                logger.warning(f"EvidenceScopeRouter failed to load LocalEmbeddingEngine: {e}")

    def _check_metadata(self, metadata: dict, category: str, rules: dict) -> bool:
        if not rules:
            return False
        
        # Check category match first
        if "category" in rules and category:
            for allowed_cat in rules["category"]:
                if allowed_cat in category:
                    return True
                    
        # Check metadata match
        if not metadata:
            return False
            
        for rule_field, allowed_values in rules.items():
            if rule_field == "category":
                continue
            meta_val = metadata.get(rule_field)
            if not meta_val:
                continue
            if any(str(av).lower() in str(meta_val).lower() for av in allowed_values):
                return True
                
        return False

    async def route_references(self, query: str, intent: str, refs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        import asyncio
        routed = {
            "CORE": [],
            "BOUNDARY": [],
            "BLOCKED": [],
            "UNUSED": []
        }
        
        if not refs:
            return routed
            
        is_general = (intent not in self.INTENT_CONFIG) or (intent == "GENERAL_MEDICAL")
        config = self.INTENT_CONFIG.get(intent, self.INTENT_CONFIG["GENERAL_MEDICAL"])
        
        meta_core = config.get("META_CORE", {})
        meta_boundary = config.get("META_BOUNDARY", {})
        meta_blocked = config.get("META_BLOCKED", {})
        
        anchor_text = config.get("ANCHOR", "")
        blocked_anchor_text = config.get("BLOCKED_ANCHOR", "")
        
        anchor_vec = None
        blocked_anchor_vec = None
        
        if self.embedding_model and anchor_text:
            try:
                texts_to_encode = [anchor_text]
                if blocked_anchor_text:
                    texts_to_encode.append(blocked_anchor_text)
                    
                vecs = await asyncio.to_thread(self.embedding_model.encode, texts_to_encode, show_progress_bar=False)
                vecs = np.array(vecs).astype('float32')
                faiss.normalize_L2(vecs)
                
                anchor_vec = vecs[0:1]
                if blocked_anchor_text:
                    blocked_anchor_vec = vecs[1:2]
            except Exception as e:
                logger.error(f"Failed to encode anchors: {e}")

        semantic_refs_to_process = []
        
        for ref in refs:
            if not isinstance(ref, dict):
                continue
                
            metadata = ref.get("metadata", {})
            category = ref.get("category", "")
            
            # --- Scheme 2: Metadata Routing (Highest Priority) ---
            if self._check_metadata(metadata, category, meta_blocked) and not is_general:
                routed["BLOCKED"].append(ref)
                continue
            elif self._check_metadata(metadata, category, meta_core):
                routed["CORE"].append(ref)
                continue
            elif self._check_metadata(metadata, category, meta_boundary) and not is_general:
                routed["BOUNDARY"].append(ref)
                continue
                
            semantic_refs_to_process.append(ref)
            
        if not semantic_refs_to_process:
            return routed
            
        # --- Scheme 3: Semantic Embedding Routing Fallback (Batched) ---
        if self.embedding_model and anchor_vec is not None:
            full_texts = [f"{ref.get('source', '')} {ref.get('context', '')}" for ref in semantic_refs_to_process]
            
            try:
                ref_vecs = await asyncio.to_thread(self.embedding_model.encode, full_texts, show_progress_bar=False)
                ref_vecs = np.array(ref_vecs).astype('float32')
                faiss.normalize_L2(ref_vecs)
                
                for i, ref in enumerate(semantic_refs_to_process):
                    core_score = np.dot(anchor_vec[0], ref_vecs[i])
                    blocked_score = np.dot(blocked_anchor_vec[0], ref_vecs[i]) if blocked_anchor_vec is not None else 0.0
                    
                    assigned = False
                    if not is_general and blocked_score > 0.65 and blocked_score > core_score:
                        routed["BLOCKED"].append(ref)
                        assigned = True
                    elif core_score > 0.60:
                        routed["CORE"].append(ref)
                        assigned = True
                    elif not is_general and core_score > 0.45:
                        routed["BOUNDARY"].append(ref)
                        assigned = True
                        
                    if not assigned:
                        if is_general:
                            routed["BOUNDARY"].append(ref)
                        else:
                            routed["UNUSED"].append(ref)
            except Exception as e:
                logger.debug(f"Batch embedding fallback routing failed: {e}")
                for ref in semantic_refs_to_process:
                    if is_general:
                        routed["BOUNDARY"].append(ref)
                    else:
                        routed["UNUSED"].append(ref)
        else:
            for ref in semantic_refs_to_process:
                if is_general:
                    routed["BOUNDARY"].append(ref)
                else:
                    routed["UNUSED"].append(ref)

        return routed
