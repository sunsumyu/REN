# -*- coding: utf-8 -*-
import logging
from enum import Enum
from typing import List, Dict, Any

logger = logging.getLogger("MedicalQA.EvidenceScopeRouter")

class ScopeType(str, Enum):
    CORE = "CORE"            # 核心证据：必须参与推理，必须体现在最终答案中
    BOUNDARY = "BOUNDARY"    # 边界限制：仅限 answer_body 和 think 都最多一句的窄边界
    BLOCKED = "BLOCKED"      # 强制封锁：即使 RAG 召回了也严禁触碰，从 Prompt 中物理屏蔽，防止反向激活
    UNUSED = "UNUSED"        # 冗余屏蔽：直接从 Prompt 排除，不喂给模型


class EvidenceScopeRouter:
    """证据作用域路由，对检索召回的知识按问答意图进行精细化分级与隔离控制"""
    
    def __init__(self):
        pass

    def route_references(self, query: str, intent: str, refs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        routed = {
            "CORE": [],
            "BOUNDARY": [],
            "BLOCKED": [],
            "UNUSED": []
        }
        
        for ref in refs or []:
            if not isinstance(ref, dict):
                continue
            ctx = ref.get("context", "").lower()
            src = ref.get("source", "").lower()
            
            # 根据意图进行证据分级控制规则定义
            if intent == "DOSAGE_LIMIT":
                # 剂量题：剂量、用量、极量是核心
                if any(k in ctx or k in src for k in ["剂量", "用量", "极量", "成人", "儿童", "用法", "一日", "单次"]):
                    routed["CORE"].append(ref)
                # 禁忌、不良反应是边界 (只允许简短的一句警示，不要展开)
                elif any(k in ctx or k in src for k in ["禁忌", "慎用", "过敏", "不良反应", "副作用", "禁用"]):
                    routed["BOUNDARY"].append(ref)
                # 微观分子机制、通路为 BLOCKED，防止反向激活
                elif any(k in ctx or k in src for k in ["分子机制", "信号通路", "受体", "基因型", "靶点"]):
                    routed["BLOCKED"].append(ref)
                else:
                    routed["UNUSED"].append(ref)
                    
            elif intent == "COMPONENT":
                # 成分题：成分、组成、配方是核心
                if any(k in ctx or k in src for k in ["成分", "组成", "配方", "包含哪些", "辅料", "主要化学", "配料"]):
                    routed["CORE"].append(ref)
                # 功效主治是边界
                elif any(k in ctx or k in src for k in ["功效", "主治", "适应症", "作用"]):
                    routed["BOUNDARY"].append(ref)
                # 不良反应、禁忌是 BLOCKED
                elif any(k in ctx or k in src for k in ["不良反应", "副作用", "禁忌", "禁用"]):
                    routed["BLOCKED"].append(ref)
                else:
                    routed["UNUSED"].append(ref)
                    
            elif intent == "STORAGE":
                # 储存题：贮藏、保存是核心
                if any(k in ctx or k in src for k in ["贮藏", "储存", "存放", "保存", "温度", "阴凉", "常温"]):
                    routed["CORE"].append(ref)
                # 包装规格是边界
                elif any(k in ctx or k in src for k in ["包装", "规格"]):
                    routed["BOUNDARY"].append(ref)
                # 疗效和机制是 BLOCKED
                elif any(k in ctx or k in src for k in ["机制", "疗效", "临床试验", "不良反应", "禁用", "禁忌"]):
                    routed["BLOCKED"].append(ref)
                else:
                    routed["UNUSED"].append(ref)
                    
            elif intent == "PACKAGING":
                # 包装题：包装、规格是核心
                if any(k in ctx or k in src for k in ["包装", "规格", "每盒", "片数", "装量", "袋"]):
                    routed["CORE"].append(ref)
                # 贮藏是边界
                elif any(k in ctx or k in src for k in ["储存", "贮藏", "保存"]):
                    routed["BOUNDARY"].append(ref)
                # 疗效和机制是 BLOCKED
                elif any(k in ctx or k in src for k in ["机制", "疗效", "副作用", "不良反应", "禁用", "禁忌"]):
                    routed["BLOCKED"].append(ref)
                else:
                    routed["UNUSED"].append(ref)
                    
            elif intent == "CONTRAINDICATION":
                # 禁忌题：禁用、禁忌、慎用是核心
                if any(k in ctx or k in src for k in ["禁忌", "禁用", "过敏", "慎用", "不能用", "不宜"]):
                    routed["CORE"].append(ref)
                # 不良反应、用法用量是边界
                elif any(k in ctx or k in src for k in ["不良反应", "副作用", "用法", "用量", "剂量"]):
                    routed["BOUNDARY"].append(ref)
                # 微观分子通路、药代动力学参数为 BLOCKED
                elif any(k in ctx or k in src for k in ["分子机制", "信号通路", "受体", "基因型", "清除率", "半衰期"]):
                    routed["BLOCKED"].append(ref)
                else:
                    routed["UNUSED"].append(ref)
                    
            else:
                # 通用医学意图 (GENERAL_MEDICAL)
                # 默认机制、疗效、临床证据是 CORE
                if any(k in ctx or k in src for k in ["机制", "药理", "分子", "通路", "疗效", "主治", "适应症", "临床试验", "证据"]):
                    routed["CORE"].append(ref)
                # 其他信息为 BOUNDARY
                else:
                    routed["BOUNDARY"].append(ref)
                    
        return routed
