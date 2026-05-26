# -*- coding: utf-8 -*-
"""
统一归一化的临床文献与药品标准数据 Schema。
采用 Pydantic 模型，确保各级检索服务输出的数据在结构上完全对齐。
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class NormalizedClinicalRef(BaseModel):
    """
    三级检索架构的统一归一化参考文献/知识结构体
    """
    source: str = Field(
        description="数据源权威标记。例如: 'refs:《中国药典》', 'refs:《丁香园指南》', 'refs:药智网API'"
    )
    context: str = Field(
        description="核心文献段落或权威用药指导内容，完全脱敏且不含广告等噪音"
    )
    category: str = Field(
        default="通用",
        description="内容所属医学门类，例如: 药理机制、用药禁忌、不良反应、临床文献"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据扩展信息。例如: {'PMID': '12345'}, {'UpdateDate': '2026-05'}"
    )

    def to_pipeline_format(self) -> Dict[str, str]:
        """
        转换为问答管道 (pipeline.py) 支持的 Grounding Context 字典格式
        """
        return {
            "source": self.source,
            "context": self.context
        }
