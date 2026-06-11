import re
from typing import List

from pydantic import BaseModel, Field, field_validator

from core.enums import FacetCategory, RiskLevel


FACET_FORBIDDEN_PATTERNS = [
    r"示例\s*视角",
    r"缺少医疗问题",
    r"请提供",
    r"请输入",
    r"数据不足",
    r"无法规划",
    r"rigorous\s+data\s+processing\s+api",
    r"json\s*schema",
    r"\bschema\b",
    r"\bjson\b",
    r"\bapi\b",
    r"\bsystem\b",
    r"\bassistant\b",
    r"\buser\b",
    r"```",
    r"\{|\}|\[|\]",
]


class FacetCandidate(BaseModel):
    label: str = Field(
        description="短医学视角名，2-16个中文字符或短词组，不得包含提示语、示例、Schema、JSON/API/System等工程文本。",
        min_length=2,
        max_length=16,
    )
    category: FacetCategory = Field(description="该视角所属的医学/药理类别枚举。")
    answer_scope: str = Field(
        description="一句话说明该视角如何回答主问题，不得复述Schema或输出提示语。",
        min_length=4,
        max_length=80,
    )
    why_relevant: str = Field(
        description="一句话说明该视角与主问题的直接相关性。",
        min_length=4,
        max_length=80,
    )
    risk_level: RiskLevel = Field(
        description="该视角诱发无依据外推的风险等级。简单事实题的深机制视角通常为 medium/high。"
    )

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        from core.pipeline_workflow import validate_facet_label
        ok, err = validate_facet_label(value)
        if not ok:
            raise ValueError(err)
        return value.strip()


    @field_validator("answer_scope", "why_relevant")
    @classmethod
    def validate_explanation(cls, value: str) -> str:
        text = value.strip()
        lowered = text.lower()
        for pattern in FACET_FORBIDDEN_PATTERNS:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                raise ValueError(f"facet explanation contains forbidden pattern {pattern}")
        return text

class FacetPlan(BaseModel):
    facets: List[FacetCandidate] = Field(
        description="为输入的医疗问题规划2到8个合法医学视角候选。每个候选必须是结构化对象，不能是普通字符串、提示语、示例占位符或Schema文本。",
        min_length=2,
        max_length=8,
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
