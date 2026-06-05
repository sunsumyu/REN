# 医疗问答提纯企业级详细优化方案

本文档是医疗问答提纯管线的企业级修订版实施方案。目标不是只靠更强提示词“压住问题”，而是把事实来源、生成边界、裁判依据、局部失败、验证验收和可观测性拆成清晰的数据契约与工程模块。

---

## 1. 总体判断

当前提纯管线的主要风险来自五条链路：

1. `source` 与 RAG 工程标签进入生成模型，导致 `refs/实体库/图谱/信源/概念定义` 等残留。
2. Answer Body 先被缩窄，但 CoT 重写没有使用缩窄后的 Answer，导致 Think 宽于 Answer。
3. Judge 不带可核验事实包，只能依据 `raw_think` 和自身参数知识打分，容易误判。
4. 方括号、JSON 字符等本地熔断规则过粗，存在正常医学文本误杀。
5. planner 失败按整行回滚，导致成功切面被连坐，行号漂移和失败记录膨胀。

企业级方案必须满足：

- 生成模型只看干净事实，不看工程 provenance。
- 系统内部保留事实来源，可审计、可追溯。
- Answer 与 Think 在同一事实边界内对齐。
- Judge 使用同一个事实包做循证核验。
- 局部失败可隔离，成功结果可保留。
- 所有指标可测量，有基线、有回归集、有灰度和回滚策略。

---

## 2. 目标架构

```text
原始 refs
  |
  v
FactPack Builder
  |-- CleanedFact[]          -> 给 Purifier / Judge 使用
  |-- ProvenanceMap          -> 仅内部审计使用，不进入生成 prompt
  |-- EvidenceScope          -> CORE / BOUNDARY / BLOCKED / UNUSED
  |
  v
Answer Body Rewriter
  |-- 输出 narrowed_answer
  |-- 使用 FactPack 校验 answer facts
  |
  v
CoT Purifier
  |-- 输入 Q + planner + CORE facts + narrowed_answer boundary
  |-- 不输入 raw source / refs labels
  |
  v
Validators
  |-- Structure Gate
  |-- Leakage Scanner
  |-- Fact Entailment Check
  |
  v
Reference-Guided Judge
  |-- 输入 CleanedFact[] + purified_think
  |
  v
Planner Transaction Manager
  |-- success
  |-- partial_success
  |-- quarantined
  |-- rollback
```

---

## 3. 数据契约

### 3.1 CleanedFact

`CleanedFact` 是进入生成模型和 Judge 的唯一事实单元。它必须去除所有工程来源词。

```json
{
  "fact_id": "fact_0001",
  "claim_text": "索磷布韦维帕他韦片适用于基因1-6型慢性丙型肝炎病毒成人感染患者。",
  "scope": "CORE",
  "confidence": "HIGH",
  "intent": "INDICATION",
  "entities": ["索磷布韦维帕他韦片", "慢性丙型肝炎病毒", "基因1-6型"],
  "numeric_values": ["1-6型"],
  "source_ids": ["src_0001"]
}
```

字段说明：

- `fact_id`: 内部事实编号，可进入 Judge，但默认不得出现在生成结果中。
- `claim_text`: 干净自然医学事实，不含 `refs/实体库/图谱/数据源/抓取异常`。
- `scope`: `CORE`、`BOUNDARY`、`BLOCKED`、`UNUSED`。
- `confidence`: `HIGH`、`MEDIUM`、`LOW`。
- `intent`: 该事实支持的问题类型。
- `entities`: 事实涉及的医学实体。
- `numeric_values`: 数值、剂量、比例、基因型等硬指标。
- `source_ids`: 对应 provenance source，不给生成模型展开。

### 3.2 ProvenanceSource

`ProvenanceSource` 只保存在审计日志或中间元数据中，不进入 Purifier prompt。

```json
{
  "source_id": "src_0001",
  "original_source": "refs:《实体库:索磷布韦维帕他韦片》",
  "original_context": "概念定义: 索磷布韦维帕他韦片...",
  "source_type": "GRAPH_ENTITY",
  "is_retrieval_error": false,
  "raw_line_number": 231
}
```

### 3.3 FactPack

```json
{
  "question": "索磷布韦维帕他韦片可用于治疗哪些基因型的慢性丙型肝炎感染？",
  "intent": "INDICATION",
  "facts": [],
  "provenance_map": {},
  "blocked_reasons": []
}
```

FactPack 是生成、裁判、验证的共同事实边界。

---

## 4. 组件改造方案

### 4.1 FactPack Builder

目标：

- 从 refs 中抽取干净事实。
- 去掉 source 标签和抓取异常。
- 标注证据 scope。
- 保留 provenance map。

输入示例：

```json
{
  "source": "refs:《实体库:索磷布韦维帕他韦片》",
  "context": "概念定义: 索磷布韦维帕他韦片是一种用于治疗基因1-6型慢性丙型肝炎病毒成人感染患者的药物。"
}
```

输出给模型的事实：

```text
索磷布韦维帕他韦片适用于基因1-6型慢性丙型肝炎病毒成人感染患者。
```

内部保留：

```text
fact_0001 -> src_0001 -> refs:《实体库:索磷布韦维帕他韦片》
```

关键规则：

- `【未收录或网络异常】` 类记录不生成 CleanedFact。
- `概念定义:`、`知识关联:`、`类型:` 等结构标签只用于解析，不进入 claim_text。
- `source` 不进入生成 prompt。
- 如果 context 仅是检索失败、无事实内容，标为 `UNUSED`。
- 如果 fact 与问题直接相关，标为 `CORE`。
- 如果 fact 只是边界提醒，标为 `BOUNDARY`。
- 如果 fact 容易诱导跑偏，标为 `BLOCKED`。

---

### 4.2 Evidence Scope Router

目标：

让问题只接触必要事实，防止简单题被旁路证据写宽。

scope 定义：

- `CORE`: 直接回答问题，必须进入生成和 Judge。
- `BOUNDARY`: 相关但不是主答案，最多允许一句边界提醒。
- `BLOCKED`: 有事实价值但会诱导偏题，不进入生成。
- `UNUSED`: 重复、无关、抓取异常、无事实内容。

示例：

问题：

```text
索磷布韦维帕他韦片可用于治疗哪些基因型的慢性丙型肝炎感染？
```

CORE：

```text
适用于基因1-6型慢性HCV成人感染患者。
适用于所有主要HCV基因型。
```

BOUNDARY：

```text
适用对象为成人患者。
```

BLOCKED：

```text
呕吐后是否补服。
肾功能损害剂量。
头痛、疲劳、恶心等安全性。
利巴韦林联合方案细节。
```

企业级要求：

- 路由结果必须写入审计日志。
- 每条 BLOCKED fact 必须记录 blocked_reason。
- Judge 可看到 BLOCKED 的存在，但默认不把它作为 answer/think 必须覆盖内容。

---

### 4.3 Answer Body Rewriter

目标：

Answer Body 先缩窄，但缩窄结果必须可被 FactPack 支撑。

流程：

```text
original_answer
  -> rewrite_answer_body(Q, CORE facts)
  -> narrowed_answer
  -> answer_fact_check(narrowed_answer, CORE + BOUNDARY facts)
```

约束：

- narrowed_answer 中每个医学事实必须被 CleanedFact 支撑。
- 如果 Answer Rewriter 删除了 CORE fact，必须报警。
- 如果 Answer Rewriter 引入了 unsupported fact，必须回退或重写。
- narrowed_answer 校验通过后，才能作为 CoT 的事实边界。

这一步解决“错误 Answer 被 CoT 强制贴合”的风险。

---

### 4.4 CoT Purifier

目标：

修复 `purified_answer` 参数空转问题，让 CoT 重写围绕已验收的 Answer Body 展开。

Purifier 输入：

```text
Q
planner
CORE facts
必要 BOUNDARY facts
narrowed_answer
raw_think without engineering tags
```

Purifier 不得看到：

```text
refs:《实体库...》
图谱关系
source
context
在线公开检索系统抓取异常
```

提示词核心约束：

```text
下方 narrowed_answer 是最终回答正文的事实边界。
Think 只能推导 narrowed_answer 中出现的核心医学事实。
不得讨论 narrowed_answer 未出现的旁路成分、机制、用法、人群、试验或风险。
```

证据等级：

- 简单事实题：强制简化推理。
- 证据稀疏题：禁止机制补偿。
- 机制证据充分题：允许高熵推理。

注意：证据等级不能只靠关键词判断，应由 `intent + fact scope + fact count + supported relation type` 综合决定。

---

### 4.5 Reference-Guided Judge

目标：

Judge 不再盲评，而是基于 CleanedFact 做事实核验。

Judge 输入：

```text
Q
planner
CleanedFact[]
purified_think
narrowed_answer
raw_think optional
```

Judge 不直接输入 raw source。raw source 只用于审计。

评分规则：

- purified_think 中新增事实若被 CleanedFact 支撑，不判幻觉。
- purified_think 中新增事实若没有 CleanedFact 支撑，判事实外推。
- 若 Think 覆盖范围宽于 narrowed_answer，判 Think/Answer 不一致。
- 若 Think 少于 Answer 必要推导，判 reasoning insufficient。

Judge 输出必须包含：

```json
{
  "semantic_purity_score": 95,
  "medical_rigor_score": 95,
  "logical_depth_score": 90,
  "unsupported_claims": [],
  "leakage_terms": [],
  "answer_think_mismatch": [],
  "reason": "..."
}
```

---

### 4.6 Structure Gate

目标：

替代粗暴字符拦截，避免误杀正常医学方括号。

问题：

当前 `[`、`]` 被视为非法字符，容易误杀正常表达。

改造原则：

- 不因单个字符判定格式崩溃。
- 检测真正的 JSON 字段泄露。
- 检测结构化 schema 残留。
- 检测未闭合 JSON 片段。

建议规则：

- 命中 `"sub_questions":`、`"evidences":`、`"reasoning_chains":`、`"step_id":` 等，直接拦截。
- 文本整体可被解析为 dict/list，说明退化为结构化输出，拦截。
- 大片段含 `{`、`}` 且伴随 JSON key，拦截。
- 方括号单独出现不拦截。

实现注意：

- 优先使用 `json.loads`，失败后再考虑 `ast.literal_eval`。
- 不要只检测 startswith `{` and endswith `}`，因为模型可能输出半截 JSON。
- 对普通医学表达中的 `[1]`、`[Epclusa]`、`[CYP2B6*6]` 放行。

---

### 4.7 Leakage Scanner

目标：

全字段扫描工程残留。

扫描字段：

```text
think
answer_body
summary
planner
```

高置信拦截词：

```text
JSON
Schema
refs
RAG
source
context
实体库
图谱关系
知识图谱
数据源
在线公开检索
网络异常
抓取
未收录
```

软残留候选词：

```text
信源
实体信息
概念定义
知识关联
查询确证记录
现有素材
条目
收录
根据参考资料
最终一个逻辑闭环形成
```

处理策略：

- 高置信命中：退回重写或隔离。
- 软残留命中：语义自愈后复检。
- 重复命中：写入 leakage candidate pool。

候选词池治理：

- 自动收集失败日志中的新词。
- 人工审核后进入正式禁词表。
- 禁词表版本化，记录生效时间和来源。

---

### 4.8 Semantic Wash

目标：

清洗残留表达，但不伪造来源。

原则：

- 不把 `实体库` 替换成 `说明书`。
- 不把 `信源` 替换成 `临床文献`，除非 provenance source_type 确认是文献。
- 优先删除来源腔，保留事实。

推荐替换：

```text
查询确证记录可见 -> 删除该引导词，保留后面的事实
现有素材中并未指明 -> 现有事实不能支持
药品概念定义中明确列出 -> 相关资料明确列出 / 相关事实显示
信源的术语分界 -> 资料中的术语分界 / 原始表述中的术语分界
知识库同时收录了 -> 相关资料还提到
```

禁止替换：

```text
实体库 -> 说明书
知识库 -> 循证文献
信源 -> 临床指南
现有素材 -> 药品说明书
```

---

### 4.9 Planner Transaction Manager

目标：

把整行一票否决改为 planner 级局部事务。

状态定义：

```text
line_status:
  success
  partial_success
  rollback
  quarantined

planner_status:
  success
  failed
  quarantined
  pruned
  unchanged
```

局部保留策略：

- 若所有 planner 失败，整行 rollback。
- 若部分 planner 成功，行状态为 `partial_success`。
- 失败 planner 不直接丢弃，进入 `quarantined_planners`。
- 成功 planner 可落地，但必须重生成 summary。
- summary 只能基于成功 planner。

数据结构建议：

```json
{
  "Q": "...",
  "planners": [],
  "quarantined_planners": [],
  "summary": "...",
  "purification_status": {
    "line_status": "partial_success",
    "failed_planners": ["药理机制"],
    "schema_version": "purify_v2"
  }
}
```

风险控制：

- 下游如果不支持 `partial_success`，则先只在审计日志中启用，不写入主数据。
- 灰度期保留原始行备份。
- 对 partial_success 行建立重试队列。

---

## 5. 灰度与回滚策略

### 5.1 Feature Flags

建议增加独立开关：

```text
PURIFY_FACTPACK_ENABLED
PURIFY_SOURCE_SANITIZE_ENABLED
PURIFY_ANSWER_BOUNDARY_ENABLED
PURIFY_REFERENCE_GUIDED_JUDGE_ENABLED
PURIFY_PARTIAL_SALVAGE_ENABLED
PURIFY_STRICT_LEAKAGE_SCAN_ENABLED
```

### 5.2 灰度阶段

第一阶段：旁路观察

- 构建 FactPack，但不影响主流程。
- 记录 FactPack 与原始 refs 的差异。
- 统计 leakage scanner 命中率。

第二阶段：影子裁判

- Reference-Guided Judge 与旧 Judge 并行打分。
- 不改变最终写回，只比较误判差异。

第三阶段：小批量启用

- 对固定回归集和 20-50 行灰度样本启用。
- 观察 rollback rate、leakage rate、unsupported claim rate。

第四阶段：主流程启用

- 分批扩大范围。
- 保留旧流程快速回退开关。

### 5.3 回滚策略

- 任一新模块导致数据结构无法被下游读取，立即关闭对应 feature flag。
- 新 Judge 与旧 Judge 分歧超过阈值，回退到影子模式。
- partial_success 行比例异常升高，关闭局部事务写入，只保留审计。

---

## 6. 测试与回归集

### 6.1 黄金样本集

必须固定一组人工审计样本：

- RAG 残留样本。
- 方括号正常医学样本。
- 简单适应症题。
- 机制题。
- refs 稀疏题。
- 多 planner 部分失败题。
- Judge 曾误判事实幻觉题。

### 6.2 单元测试

应覆盖：

- FactPack Builder 是否剥离 source。
- Retrieval error 是否被标为 UNUSED。
- EvidenceScope 是否能区分 CORE/BLOCKED。
- Structure Gate 是否放行 `[CYP2B6*6]`。
- Structure Gate 是否拦截 JSON schema 残留。
- Leakage Scanner 是否扫描 think、answer、summary。
- Answer boundary 是否禁止 Think 写宽。

### 6.3 回归指标

每次改动必须输出：

```text
leakage_rate
unsupported_claim_rate
answer_think_mismatch_rate
judge_disagreement_rate
rollback_rate
partial_success_rate
row_salvage_rate
latency_p50 / p95
```

---

## 7. 验收指标

指标必须基于固定回归集和真实灰度批次统计，不使用拍脑袋估计。

### 7.1 必达指标

- 工程噪声高置信残留率：0。
- 方括号正常医学表达误杀率：0。
- FactPack provenance 可追溯率：100%。
- Answer 中 unsupported medical claim：0。
- Think 中 unsupported high-risk claim：0。

### 7.2 观测指标

- Judge 误判率：相比旧流程下降。
- 整行 rollback rate：相比旧流程下降。
- partial_success salvage rate：相比旧流程上升。
- 平均生成延迟：不得超过旧流程 20%。
- leakage scanner 误报率：人工抽样评估。

### 7.3 阈值示例

正式阈值必须由基线批次确认。初始建议：

```text
leakage_rate_high_confidence = 0
unsupported_claim_rate <= 1%
answer_think_mismatch_rate <= 1%
rollback_rate 降幅 >= 30%
latency_p95 增幅 <= 20%
```

---

## 8. 大规模 Token 暴涨诊断与优化方案 (Anthropic & Hello-Agents 架构实践)

在大规模跑批与提纯过程中，随着 RAG 事实扩展与深度推理要求提升，极易遇到 LLM 输入（Prompt）Token 大于 10000 导致调用成本高昂甚至超时熔断的风险。参考 Anthropic 智能体架构白皮书与 Hello-Agents 框架，诊断原因及对应优化设计如下：

### 8.1 5 大 Token 暴涨根因诊断
1. **RAG 检索上下文和参考文献文本过载（生成阶段）**：
   在 [PipelineWorkflow._prepare_context_and_refs](file:///d:/REN/qa/core/pipeline_workflow.py#L112) 中，通过三级检索架构获取医学定义、PubMed 文献摘要和网页爬取碎片。由于提取的文献摘要和网页描述通常较长，且随着实体数量的增加，这些 `refs` 被累加并拼接进 [L3_DYNAMIC_CONTEXT_TEMPLATE](file:///d:/REN/qa/prompts.py#L210)，导致调用单个切面问答时，输入 prompt 的 `refs` 块部分就能占到 6,000 ~ 9,000 个 token，使得整体输入很容易突破 10,000。
2. **多切面回答的综合凝练聚合（生成阶段）**：
   在 [PipelineWorkflow.synthesize_answers](file:///d:/REN/qa/core/pipeline_workflow.py#L692) 中，系统需要将所有切面生成的回答正文（`answer_body`）聚合在一起并传入 [MULTI_ANSWER_SYNTHESIS_TEMPLATE](file:///d:/REN/qa/prompts.py#L327)。若存在 6 ~ 8 个切面且每个切面的回答正文篇幅较长（如 1,000 ~ 1,500 token），聚合后的 answers 参数总长度便在 6,000 ~ 12,000 token 之间，使得最终总结阶段的输入 token 稳定超过 10,000。
3. **多轮对话历史的不断累积（生成阶段）**：
   在 [PipelineWorkflow.generate_next_question](file:///d:/REN/qa/core/pipeline_workflow.py#L720) 中，使用 [NEXT_QUESTION_TEMPLATE](file:///d:/REN/qa/prompts.py#L406) 生成下一轮追问。随着对话轮次的推进（例如到了第 3、4 轮），除了携带的图谱 `context_list` 外，`history`（包含之前所有轮次的问题 Q 与摘要总结 summary）也在线性增长。多轮历史累积使得后面轮次在生成下一个问题时的输入 token 极易突破 10,000。
4. **结构化输出自愈重试的上下文堆叠（服务调用阶段）**：
   in [LLMService.call_llm_structured](file:///d:/REN/qa/services/llm_service.py#L377) 中，如果大模型返回的内容未能通过 Pydantic 校验，系统会进行最多 2 次自愈重试。在重试时，系统会将上一轮错误的 assistant 回复（包含超长 reasoning 推理，可能达数千个 token）和 Pydantic 校验错误信息直接 append 到消息上下文列表中。这种历史堆叠机制导致在重试时，输入 token 会以一倍、两倍的速度暴增，非常容易突破 10,000。
5. **提炼净化阶段超长少样本（Few-shot）和原始思维链（Raw Think）叠加（提纯阶段）**：
   在 [PurificationEngine.purify_single_think](file:///d:/REN/qa/core/purification_engine.py#L67) 中执行思维链重构。提示词拼接了专为医疗推理设计的 few-shot 范例（如 [FEW_SHOT_PHARMACOLOGY](file:///d:/REN/qa/core/purification_prompts.py#L92) 等，本身包含大量物理内容，达 3,000 ~ 4,000 token）、长系统提示词指令（~1,500 token）、锚点确证事实（`anchors_prompt`），以及原本就包含大量 RAG/工程噪声的超长原始思维链内容 `stripped_think`（常在 2,000 ~ 4,000 token 左右）。基础输入加上这些超长内容直接推高了基础 Prompt token，使其轻易越过 10,000 的阈值。

### 8.2 基于 Anthropic & Hello-Agents 的架构优化设计
1. **RAG 证据域精准分流路由（Routing 模式）**：
   - *优化思路*：在证据域 [EvidenceScopeRouter](file:///d:/REN/qa/core/rag/evidence_scope_router.py) 或 FactPack 事实分发时，不再对所有切面 Worker 喂入全量 `refs` 或事实包。根据当前切面类型进行精准分流路由，只将与当前切面直接相关的核心事实分发给对应的 Planner Worker。
   - *效果*：Worker 输入的 context 载荷可降低 70% 以上。
2. **Workers 结果轻量摘要合并（Orchestrator-Worker 模式）**：
   - *优化思路*：修改 [FacetQAOutput](file:///d:/REN/qa/models.py) 的输出 Schema，强制要求 Worker 额外吐出一个 150 字以内的精炼要点摘要（`summary_bullet_points`）。Orchestrator 接收到所有 Workers 结果后，仅将要点摘要聚合并输入 [MULTI_ANSWER_SYNTHESIS_TEMPLATE](file:///d:/REN/qa/prompts.py#L327) 进行融合成文，而不需要读取完整的正文列表。
   - *效果*：综合总结阶段的输入 Token 降至 1,500 左右，大幅降低模型注意力负载。
3. **记忆状态语义压缩（Stateful Memory 压缩）**：
   - *优化思路*：引入滑动窗口记忆与语义摘要记忆。在追问任务中，仅保留上一轮对话（第 $N-1$ 轮）的完整细节，而对于更早的历史轮次（$1$ 到 $N-2$ 轮），在后台将其 Q&A 内容进行语义高度压缩，合成为一段简洁的对话状态描述注入 prompt，避免多轮历史无节制线性增长。
   - *效果*：对话历史 Token 开销由 $O(N)$ 增长变为常数级控制。
4. **自愈重试的思考链状态剪枝（Evaluator-Optimizer 状态清理）**：
   - *优化思路*：在大模型自愈重试回贴历史时，通过正则匹配强行过滤并剔除 Assistant 上一次输出的 `<think>...</think>` 推理内容。仅保留需要重构的 malformed JSON 字段正文与 Pydantic ValidationError 描述回贴给模型。
   - *效果*：避免重试输入 Token 因携带废弃推理而发生一倍、两倍的体积膨胀，重试阶段输入极其纯净。
5. **少样本动态路由分发（Dynamic Few-Shot Routing）**：
   - *优化思路*：将提纯 Purifier 的 few-shot 范例从硬编码全量导入升级为动态检索机制。建立小粒度切面 few-shot 数据库，根据当前的切面类型（`planner`），检索并注入且仅注入 1 个最适合的少样本示范，避免长篇 few-shot 全量灌入造成的常态化 Token 负担。
   - *效果*：初始 Prompt Token 基础开销减少 2,000 ~ 3,000 Tokens。

---

## 9. 推荐落地顺序

### 阶段一：止血

1. Structure Gate 去掉方括号硬拦截。
2. Leakage Scanner 扩展为全字段扫描。
3. 禁词表版本化。

风险低，收益快。

### 阶段二：事实边界

1. 建立 FactPack Builder。
2. 生成 prompt 不再注入 raw source。
3. Answer Rewriter 使用 CORE facts。
4. Answer 先过 fact check，再进入 CoT boundary。

这是核心工程改造。

### 阶段三：循证裁判

1. Judge 接入 CleanedFact。
2. 新旧 Judge 影子运行。
3. 对分歧样本做人审抽样。

### 阶段四：局部事务

1. partial_success 先只写审计日志。
2. 下游兼容后再写入数据集。
3. 建立 quarantined_planners 和重试队列。

### 阶段五：证据路由精细化

1. 路由规则从关键词升级为 intent + scope + fact 支撑关系。
2. 对简单题默认屏蔽旁路机制/药代/安全性证据。

---

## 10. 最终企业级判定标准

该方案落地后，只有满足以下条件，才能称为企业级：

1. 有稳定的数据契约：FactPack、ProvenanceMap、PurificationStatus。
2. 有可回放的审计日志。
3. 有固定黄金回归集。
4. 有 feature flags 和回滚路径。
5. 有新旧流程对照数据。
6. 有量化指标，而非主观“质量提升”。
7. 有 partial_success 的下游兼容策略。
8. 所有输出医学事实都能追溯到 CleanedFact 或被明确标注为合理边界说明。

---

## 11. 简短结论

企业级提纯不是把 prompt 写得更强，而是把事实、来源、生成、裁判、验证、事务和审计拆开。

本方案的核心落点是：

```text
生成模型只负责表达。
FactPack 负责事实边界。
ProvenanceMap 负责来源追溯。
Judge 负责循证核验。
Validator 负责格式和残留拦截。
Transaction Manager 负责失败隔离与数据保留。
```

这才是可以长期运行、可审计、可回滚、可扩展的医疗问问提纯管线。

