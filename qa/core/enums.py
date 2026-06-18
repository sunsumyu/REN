# -*- coding: utf-8 -*-
from enum import Enum

class FacetAction(str, Enum):
    """针对切面的治理动作"""
    KEEP = "KEEP"  # 保持
    RENAME = "RENAME"  # 重命名并修复
    DROP = "DROP"  # 废弃/脏切面
    REDIRECT_SIMPLE = "REDIRECT_SIMPLE"  # 重定向至极简模式并重命名为规范切面

class CompatibilityLevel(str, Enum):
    """兼容性评估结果"""
    COMPATIBLE = "COMPATIBLE"  # 强兼容
    COMPATIBLE_SIMPLE = "COMPATIBLE_SIMPLE"  # 简单兼容，极简推理
    FORCED_SKIP = "FORCED_SKIP"  # 强套偏题，直接舍弃

class FacetCategory(str, Enum):
    """
    医学/药理类别枚举类

    该类定义了用于分类医学或药理学信息的不同维度，继承自 str 和 Enum，
    确保每个成员既是字符串又是唯一的枚举值。

    Attributes:
        COMPOSITION (str): 成分组成
        EFFICACY (str): 疗效
        DOSAGE (str): 用法用量
        CONTRAINDICATION (str): 禁忌症
        ADVERSE_REACTION (str): 不良反应
        PHARMACOKINETICS (str): 药代动力学
        MECHANISM_BOUNDARY (str): 作用机制与边界
        STORAGE_QUALITY (str): 贮藏与质量
        POPULATION_SAFETY (str): 人群安全性
        CLINICAL_EVIDENCE (str): 临床证据
        OTHER_MEDICAL (str): 其他医学相关信息
    """
    COMPOSITION = "composition"
    EFFICACY = "efficacy"
    DOSAGE = "dosage"
    CONTRAINDICATION = "contraindication"
    ADVERSE_REACTION = "adverse_reaction"
    PHARMACOKINETICS = "pharmacokinetics"
    MECHANISM_BOUNDARY = "mechanism_boundary"
    STORAGE_QUALITY = "storage_quality"
    POPULATION_SAFETY = "population_safety"
    CLINICAL_EVIDENCE = "clinical_evidence"
    OTHER_MEDICAL = "other_medical"

class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
