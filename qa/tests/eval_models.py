from pydantic import BaseModel, Field
from typing import List

class JudgeMetric(BaseModel):
    score: float = Field(
        description="客观打分，范围限制在 0.0 到 10.0 之间，以 10.0 为完美通过",
        ge=0.0,
        le=10.0
    )
    reason: str = Field(description="详细的打分理由，客观分析生成答案 of 优缺点")

class ComprehensiveJudgeMetrics(BaseModel):
    success: JudgeMetric = Field(description="成功度评分。评估生成的输出是否完整包含所需结构（如<think>等），以及任务是否完整执行")
    recall: JudgeMetric = Field(description="查全率评分。评估无发现或拒答时是否有合理的边界扫描证据，而非随意拒答")
    precision: JudgeMetric = Field(description="精确度评分。评估医学推论、剂量匹配或药效陈述是否百分百精确无误")
    faithfulness: JudgeMetric = Field(description="事实忠实度评分。评估生成的医学内容是否基于且只基于 refs 事实，是否存在凭空幻觉或严重偏离事实")
    relevance: JudgeMetric = Field(description="相关性评分。评估回答是否直击提问者的核心诉求，有无冗余和避重就轻")
    professionalism: JudgeMetric = Field(description="专业度评分。评估使用的医学术语是否规范、文风是否像严谨的临床医学专家")
    interpretability: JudgeMetric = Field(description="可解释性评分。评估答案是否有清晰的逻辑推导和证据来源引用，而不是简单生硬地给出结论")
    isolation: JudgeMetric = Field(description="领域隔离度与防污染评分。评估生成的医学内容是否被不相关的语境（如法律合规/供应链/数据隐私等）污染跑题，坚守医学专业边界为 10.0 分")
    complexity: JudgeMetric = Field(description="推演复杂度评分。评估问题本身的认知深度及对应回答是否展现了真实的逻辑推演链。若问题属单跳查询（仅需查阅无推演）直接给予严重扣分（不及格）；涉及多条件冲突消解与深度诊断可给高分")

class EvalResultItem(BaseModel):
    case_id: str = Field(description="测试用例ID，例如 TC_01, TC_02...")
    category: str = Field(description="测试用例所属分类")
    query: str = Field(description="测试输入的医学核心问题")
    schema_ok: bool = Field(description="Pydantic 格式验证是否通过（100% 对齐返回）")
    self_healing_attempts: int = Field(description="自愈重试的次数（0 到 2）")
    refusal_avoided: bool = Field(description="是否成功规避了免责拒答。True 表示成功给出了专业的客观医学陈述；False 表示大模型选择摆烂逃避，吐出了‘抱歉/无法回答/不符合安全’等拒绝话术")
    recall_rate: float = Field(description="召回率：生成的答案中包含了预设核心关键词的百分比 (0.0 到 1.0)")
    is_success: bool = Field(description="综合业务成功率：格式完全契合 AND 未触发拒答 AND 关键词召回率 >= 0.5 AND 所有裁判主观打分均及格 (>= 6.0)")
    judge_metrics: ComprehensiveJudgeMetrics = Field(description="包含事实忠实度、领域隔离度、可解释性、专业性、相关性的大模型综合评判指标")
    answer_preview: str = Field(description="最终生成的医学回答正文前 150 字符预览")
