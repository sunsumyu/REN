from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class FacetPlan(BaseModel):
    facets: List[str] = Field(
        description="为输入的医疗问题规划学术、临床与药理切面视角（数量必须在 2 到 8 个之间），请根据问题本身的专业特性定制化规划，不限制名称（如：分子机制、古籍收采、药代动力学、特殊人群安全等）。",
        min_items=2,
        max_items=8
    )

class EvidenceItem(BaseModel):
    source: str = Field(description="证据来源文献/说明书/图谱节点，格式统一为：refs:《文献/说明书名》")
    location: str = Field(description="具体位置，如章节、条款号、页码、段落号")
    summary: str = Field(description="提取的证据事实要点，必须真实、客观")

class ReasoningStep(BaseModel):
    step_id: str = Field(description="推理步骤ID，如 P1, P2...")
    logic: str = Field(description="推理逻辑说明：哪几条证据推导出了什么医学结论")

class FacetQAOutput(BaseModel):
    sub_questions: List[str] = Field(description="将原 Q 拆解后的临床子问题清单")
    evidences: List[EvidenceItem] = Field(description="对齐的核心证据清单")
    reasoning_chains: List[ReasoningStep] = Field(description="从证据到最终结论的严密逻辑推理链")
    final_conclusion_summary: str = Field(description="一句话最终医学结论摘要，用于数据集自检")
    answer_body: str = Field(description="最终生成的专业医学回答正文。要求结构清晰、分点排版，绝不能包含工具/检索等过程痕迹")
