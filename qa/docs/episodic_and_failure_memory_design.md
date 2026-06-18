# 🧠 基于成功与失败经验双向沉淀的自演进提示词优化架构设计方案 (Dual-Loop Cognitive Memory & Prompt Evolution System)

在医疗问答数据集生成及提纯管线中，随着生成量级扩大，系统面临两大挑战：
1. **好的生成模式无法复用**：已通过质检的高评分优质 CoT（思维链）无法为后续相似实体提供动态少样本（Few-shot）引导，导致系统平均质量无法随时间演进。
2. **相同的错误重复犯**：失败的模式（如特定疾病机制脑补、RAG 噪音泄露、特定药物药代动力学过度推导）被直接丢弃或盲目重试，缺乏从“历史墓地”（`purification_failures.jsonl`）中学习并产生“免疫力”的机制。

为了解决这一问题，本方案设计了一套**双向双循环（成功/失败双记忆库）的自演进提示词优化系统**，将“成功经验”转化为“动态 Few-Shot”，将“失败教训”泛化总结为“动态/静态负向提示词（Negative Prompts/Constraints）”。

---

## Ⅰ. 架构概览 (System Architecture)

系统由**成功记忆循环**与**失败免疫循环**双轨组成，并包含**在线检索注入**与**离线提示词演进**两个维度：

```mermaid
graph TD
    subgraph 在线生成与纯化 (Online Loop)
        Query[主问题 Query & 切面 Facet] --> Router[双记忆库相似度检索]
        Router -->|召回成功样例| Prompt_FS[注入动态 Few-Shot 示例]
        Router -->|召回失败规避约束| Prompt_Neg[注入动态负向约束 Negative Prompt]
        Prompt_FS & Prompt_Neg --> LLM_Gen[LLM 生成 / 提纯模型]
        LLM_Gen --> Judge[质量门控裁判 LLM-as-Judge]
        Judge -->|通过 (评分 >= 9.5)| Succ_Dump[成功沉淀]
        Judge -->|拒绝 (格式崩溃/幻觉/扣分)| Fail_Dump[失败隔离]
    end

    subgraph 成功记忆通道 (Success Memory)
        Succ_Dump --> Vector_Store_Succ[(Success FAISS/SQLite Vector DB)]
        Vector_Store_Succ -.->|供在线检索| Router
    end

    subgraph 失败免疫通道 (Failure Memory)
        Fail_Dump --> Fail_Log[purification_failures.jsonl]
        Fail_Log --> Generalizer[失败泛化总结器 LLM Generalizer]
        Generalizer --> Fail_Pat_DB[(Failure Patterns DB)]
        Fail_Pat_DB -.->|提供负向约束| Router
        Generalizer -->|高频常驻错误| Static_Prompt_Opt[静态提示词优化器]
        Static_Prompt_Opt -->|更新| Static_Prompts[静态系统提示词 prompts.py]
    end
```

---

## Ⅱ. 核心设计细节 (Core Components)

### 1. 成功经验沉淀：动态 Few-Shot 机制

*   **沉淀条件**：当样本通过 `LLMJudgeStrategy` 质检，且 `semantic_purity_score >= 95`、`medical_rigor_score >= 95`，无任何事实性错误标记（`factual_errors` 为空）时，被定义为 **“黄金生成案例（Golden Case）”**。
*   **向量库存储**：提取黄金案例的 `(Query, Facet)` 进行语义嵌入（Embedding），将三元组 `(Query, Facet, Purified_Thought)` 存入本地成功经验库（Success Vector DB）。
*   **动态召回**：在调用大模型生成相似问题时，计算当前 Query 的向量，检索相似度 $\ge 0.82$ 且 Facet 匹配的黄金案例。如果有，将其作为 1-Shot 注入当前提示词中。

---

### 2. 失败经验泛化：从“案发现场”到“避坑指南”

对于运行失败的样本（格式崩溃、事实幻觉、扣分未过），系统不会简单丢弃，而是从“微观自愈”和“宏观免疫”两个维度进行泛化学习。

#### A. 微观层：单次生成内“带诊断意见的局部自愈”（Intra-turn Local Retry Feedback）
*   **机制**：当 Judge 打分未达标时，提取 Judge 返回的错误清单 `factual_errors` 以及扣分理由 `reason`，通过局部 Feedback 模板重新灌回消息链，引导模型在下一轮尝试中改正。

#### B. 宏观层：跨样本“历史失败模式泛化总结与动态约束”（Inter-turn Dynamic Constraints / Negative Prompting）
*   **泛化器（LLM Generalizer）**：定期扫描 `purification_failures.jsonl`，将相同药物类别、疾病类别或相同错误大类（如“药代动力学漂移”）的错误日志进行聚合，调用 LLM 将零散的错误日志归纳总结为**“抽象特征”**与**“避坑约束规则”**。

##### 💡 失败泛化提示词示例（LLM Generalizer Prompt）：
```text
你是一个医疗数据集流水线质量分析专家。下面是同一疾病大类/切面中高频提纯失败的 5 条错误日志：
---
[错误 1]: 问题关于西尼莫德，思维链大篇幅推导了其在肝脏中的 CYP2C9 代谢参数，被裁判判定为超出 refs 范围胡编事实。
[错误 2]: 问题关于塞来昔布，模型推理了 CYP2C9 强弱代谢型人群的药代动力学暴露量差异，被裁判判定为脑补机制且与 Answer 边界不匹配。
...
---
请完成两项任务：
1. 泛化出这一类失败问题的【医学特征/偏离规律】：（如“针对选择性 COX-2 抑制剂及 S1P 受体调节剂，模型易脑补 CYP 酶代谢通路及 PK 参数暴露量差异”）。
2. 生成针对这一特征的【避坑约束规则（Negative Prompt Rule）】：（如“避坑限制：严禁推演任何未在 refs 中明确记载的肝脏 CYP 酶代谢参数或 PK 代谢暴露量，仅说明说明书已知适应症。”）。
```

*   **负向提示词库（Failure Patterns DB）**：将泛化出的【医学特征】作为 Key 执行 Embedding 索引，将【避坑约束规则】作为 Value 存储。
*   **动态负向提示词注入（Negative Prompting）**：新问题（如“艾瑞昔布的代谢与排泄”）输入时，计算其语义向量并检索“失败特征库”。若召回相似度 $\ge 0.80$ 的特征，则**动态拼接负向约束**：
    ```text
    【⚠️ 历史相似失败规避红线（根据相似失败记录自动注入）】：
    - 避坑限制：严禁推演任何未在 refs 中明确记载的肝脏 CYP 酶代谢参数或 PK 代谢暴露量，仅基于已知文献说明其代谢路径。
    ```

---

### 3. 静态提示词演进：自动化“抗体”生成

*   **常驻红线演进**：如果某一类错误在全局统计中发生频次极高（如“输出开头包含‘好的，分析如下’等答题客套话”占到了失败总数的 30% 以上），离线优化器将自动向静态提示词文件（如 [purification_prompts.py](file:///d:/REN/qa/core/purification_prompts.py)）中追加硬性红线。
*   **禁词表自增长**：当失败日志中某些工程词（如 `实体库`、`关系图谱`）高频出现且导致 Purity 扣分时，自动将这些词归入 [answer_guard.py](file:///d:/REN/qa/strategies/quality_gate/answer_guard.py) 的 `BANNED_KEYWORDS` 禁词表中，实现防御机制的主动升级。

---

## Ⅲ. 代码架构蓝图 (Code Blueprints)

### 1. 失败经验管理器 (`FailureMemoryManager`)

新建或在 `services/` 下扩展失败管理器，负责失败记录的 Embedding 索引、检索和注入。

```python
# services/failure_memory.py
import faiss
import numpy as np
from typing import Dict, Any, List, Tuple

class FailureMemoryManager:
    def __init__(self, llm_service, dimension: int = 768):
        self.llm_service = llm_service
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.patterns: List[Dict[str, str]] = []  # 存放 {"pattern_desc": ..., "negative_rule": ...}

    def add_failure_pattern(self, pattern_desc: str, negative_rule: str, vector: np.ndarray):
        """将泛化出的失败模式加入数据库"""
        self.index.add(np.array([vector], dtype=np.float32))
        self.patterns.append({
            "pattern_desc": pattern_desc,
            "negative_rule": negative_rule
        })

    def retrieve_negative_prompts(self, query_vector: np.ndarray, threshold: float = 0.85) -> List[str]:
        """检索出与当前 Query 最相似的历史失败模式，并返回对应的防坑约束"""
        if self.index.ntotal == 0:
            return []
        
        # 搜索最近邻
        distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k=2)
        rules = []
        for dist, idx in zip(distances[0], indices[0]):
            # 距离越小，相似度越高（L2 距离转化为相似度判定）
            if idx != -1 and idx < len(self.patterns):
                # 示例门限判断
                if dist < threshold:  
                    rules.append(self.patterns[idx]["negative_rule"])
        return rules
```

### 2. 在管线中的集成挂载点

在 [purification_engine.py](file:///d:/REN/qa/core/purification_engine.py#L126) 的 `purify_single_think` 中集成动态负向提示词检索：

```python
# core/purification_engine.py

# 1. 动态召回成功 Few-shot
query_vector = await self.llm_service.get_embedding(q)
dynamic_few_shot = self.action_cache.retrieve_few_shot(query_vector, top_k=1)
use_few_shot = dynamic_few_shot[0]["purified_think"] if dynamic_few_shot else few_shot

# 2. 动态召回失败免疫规避约束
negative_rules = self.failure_memory.retrieve_negative_prompts(query_vector)
negative_prompt_addition = ""
if negative_rules:
    negative_prompt_addition = "\n\n【⚠️ 历史相似失败规避红线（系统自动防御注入）】:\n" + \
                               "\n".join(f"- {rule}" for rule in negative_rules)

# 3. 组装 Prompt
prompt = f"""{use_few_shot}
...
{anchors_prompt}{negative_prompt_addition}

问题: {q}
原始思维链 (CoT) 内容:
..."""
```

---

## Ⅳ. 落地步骤建议 (Roadmap)

1.  **静态规则硬化先行（第一阶段）**：
    *   将 `purification_failures.jsonl` 中由于方括号、客套词等高频引发拦截的特征，硬化为 [purification_prompts.py](file:///d:/REN/qa/core/purification_prompts.py) 的静态 System Prompt 指令。
2.  **微观重试自愈改造（第二阶段）**：
    *   实现生成管线中 `answer_single_facet` 的“带诊断意见反馈重试”机制。
3.  **失败经验泛化模块开发（第三阶段）**：
    *   开发离线/准实时脚本 `scripts/generalize_failures.py`，实现高频失败模式的聚类和 LLM 抽象泛化。
4.  **向量库双记忆检索系统上线（第四阶段）**：
    *   引入本地向量索引，打通“生成/提纯”与“成功经验/失败防坑”的闭环，实现完全的自上演自进化数据集生产。
