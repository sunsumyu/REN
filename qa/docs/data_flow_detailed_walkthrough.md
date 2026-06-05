# 🩺 医疗问答数据生成与提纯全链路数据流转详细设计说明书

本技术规格说明文档旨在详细剖析本项目（`Medical QA Facet Synthesis & CoT Purification`）中**第一阶段（数据对话合成）**与**第二阶段（思维链提纯净化）**的端到端数据流转过程。文档按照“先梳理核心主线流程，再扩展到枝叶实现”的逻辑设计，并附带可直接点击跳转的代码节点（包含具体的类名与方法名）。

---

## 1. 📖 全景数据流转总线图

整个系统数据流转主要围绕两个阶段的核心入口展开，数据交汇点为落盘的原始数据集。

```text
【知识图谱数据库 qa_datasets.db】 
        │
        ▼ (RAG 检索 & 种子问题生成)
【第一阶段：生成轨 Generation】 ───> 核心调度器: PipelineWorkflow
        │ (切面生成、并发Agent深度问答、综合摘要)
        ▼
【原始数据集 medical_qa_dataset.jsonl】 
        │
        ▼ (增量读取 & FactPack 事实包构建)
【第二阶段：提纯轨 Purification】 ──> 核心调度器: medicalqa_purifier.py
        │ (事实脱敏、回答重写、CoT净化、本地网关拦截、循证裁判、事务管理)
        ▼
【最终高熵纯微调数据集 medical_qa_dataset.jsonl】
```

---

## 2. 🚀 第一阶段：数据集多视角多轮对话生成流转详析

第一阶段的核心目标是：从医学图谱与文献中，基于 Agent 仿真技术，自动合成为一个具有多分析切面（如药理、用药禁忌、不良反应等）、带有探索性 `<think>` 推理过程的高质量多轮问答数据集。

### 2.1 核心主线数据流
1. **主控运行与接口代理**：由运行脚本或 [main.py](file:///d:/REN/qa/main.py) 触发，实例化后向兼容的 Proxy 代理类 [MedicalQAPipeline](file:///d:/REN/qa/pipeline.py#L69)。代理类在初始化时拉起核心工作流并注入依赖。
2. **工作流调度**：通过代理类将方法调用路由至 [PipelineWorkflow](file:///d:/REN/qa/core/pipeline_workflow.py#L90) 类这一核心编排器，通过 [generate_single_round](file:///d:/REN/qa/core/pipeline_workflow.py#L745) 执行单轮问答生成循环，并使用 [generate_multi_round_dataset](file:///d:/REN/qa/core/pipeline_workflow.py#L800) 控制多轮会话的进化状态。

---

### 2.2 详细步骤与代码实现节点 (枝叶分展)

#### 第一步：图谱抽取与三级分层检索
* **核心方法**：[PipelineWorkflow._prepare_context_and_refs](file:///d:/REN/qa/core/pipeline_workflow.py#L112)
* **枝叶流转**：
  * 从 `qa_datasets.db` 提取实体（Entity）与关系（Relationship）。
  * 提取后，将实体名送入统一检索管理中心 [RetrievalManager.get_grounding_references](file:///d:/REN/qa/retrieval/retrieval_manager.py#L29)。
  * **三级检索过滤**：首先调用本地私有库 RAG (Tier 1)，若无则降级调用垂直医药 API 网关 (Tier 2)，若无则降级调用受限域名白名单联网搜索 (Tier 3)。最终输出格式化参考依据 `refs`。

#### 第二步：种子问题 Q 生成
* **核心方法**：[PipelineWorkflow.generate_initial_question](file:///d:/REN/qa/core/pipeline_workflow.py#L188)
* **枝叶流转**：
  * 将格式化的上下文数据输入给轻量级大模型（model_pool = `"lightweight"`）。
  * 模型基于提示词模板 `QUESTION_CREATOR_TEMPLATE` 生成一系列核心问题候选，并随机抽取一个作为当前行的主问题 $Q$。

#### 第三步：多视角切面规划与合并预处理
* **核心方法**：[PipelineWorkflow.plan_facets](file:///d:/REN/qa/core/pipeline_workflow.py#L215) 规划切面；[PipelineWorkflow.preprocess_facets](file:///d:/REN/qa/core/pipeline_workflow.py#L289) 合并切面。
* **枝叶流转**：
  * 使用 `FACET_PLANNER_TEMPLATE` 指引轻量模型输出切面列表。
  * **切面数量对齐机制**：若规划出切面数量大于8，启动 Reducer（缩减）；在2至8之间，启动 Expander（丰富）；为2个则保持互补。在此过程中，使用 [validate_facet_label](file:///d:/REN/qa/core/pipeline_workflow.py#L49) 硬性阻断含有 `JSON` 等非合规命名，确保切面字面量的医学合法性。

#### 第四步：切面兼容性治理与证据路由 (Evidence Scope Routing)
* **核心类/方法**：
  * 兼容过滤器：`core/governance/facet_strategy.py` 中的 `FacetGovernanceFilter` 评估兼容决策。
  * 证据域路由：[EvidenceScopeRouter.route_references](file:///d:/REN/qa/core/rag/evidence_scope_router.py#L21) 过滤 refs 边界。
* **枝叶流转**：
  * 生成切面后，利用小模型分析主问题与各切面的兼容度，执行 `DROP`（丢弃）、`RENAME`（重命名自愈）或 `REDIRECT_SIMPLE`（极简退避）。
  * 同时，将 refs 送入路由，根据意图细分为 `CORE`、`BOUNDARY`、`BLOCKED`、`UNUSED` 四种等级，剔除旁路噪声，防止把简单问答写宽。

#### 第五步：并发切面 Agent 仿真生成
* **核心方法**：[PipelineWorkflow.run_parallel_answers](file:///d:/REN/qa/core/pipeline_workflow.py#L527) 调度并发；[PipelineWorkflow.answer_single_facet](file:///d:/REN/qa/core/pipeline_workflow.py#L365) 单视角深度问答。
* **枝叶流转**：
  * 使用 `asyncio.Semaphore` 限制最大并发量。
  * 调用高级大模型（model_pool = `"premium"`），强制大模型按照 `FacetQAOutput` 强类型 schema 输出包含 evidences、reasoning_chains、answer_body 的回答。
  * **质量防卫线**：调用 `strategies/quality_gate/answer_guard.py` 的 `check_answer_quality` 检查输出质量，不合格则打回并重试。若重复失败则启动普通文本 + 推理心流（call_llm_with_reasoning）的降级容错方案。

#### 第五步半：视角冗余度检测与去重过滤 (Redundancy Filter)
* **核心类/方法**：
  * 冗余过滤调度：在 [PipelineWorkflow.generate_single_round](file:///d:/REN/qa/core/pipeline_workflow.py#L780) 中调用冗余过滤器。
  * 冗余过滤策略：[LLMRedundancyFilterStrategy.filter_redundancy](file:///d:/REN/qa/strategies/redundancy_filter/llm_filter.py#L20)
* **枝叶流转**：
  * 并发切面回答生成后，将其送入策略去重过滤器。
  * 过滤器调用轻量级大模型（model_pool = `"lightweight"`），利用 `FACET_REDUNDANCY_DETECTOR_TEMPLATE` 指引模型识别内容重复或过度重叠的切面，并输出需要剔除的视角索引。
  * 最终仅保留非冗余的切面回答，传送至下一步进行综合总结。

#### 第六步：综合总结与多轮进化
* **核心方法**：[PipelineWorkflow.synthesize_answers](file:///d:/REN/qa/core/pipeline_workflow.py#L692) 总结；[PipelineWorkflow.generate_next_question](file:///d:/REN/qa/core/pipeline_workflow.py#L720) 下一轮生成。
* **枝叶流转**：
  * 剥离各切面回答中的 `<think>` 推理块，将干净的 `answer_body` 合并，调用小模型合成连贯的 `summary`。
  * 结合历史 `history`，由下一问生成器产生深入的下一个问题，开始下一轮生成，最后以 JSONL 追加写入到 `medical_qa_dataset.jsonl` 中。

---

## 3. 🛡️ 第二阶段：思维链提纯、学术自愈与质量门禁拦截流转详析

第二阶段的核心目标是：针对原始生成数据，离线剔除所有的工程残留字眼（如 refs、json 等），重新提纯思维链 CoT 使之完美契合最终回答事实边界，达到頂尖推理模型（如 DeepSeek-R1）微调的冷启动标准。

### 3.1 核心主线数据流
1. **启动清洗**：由外部运行脚本 [medicalqa_purifier.py](file:///d:/REN/qa/scripts/medicalqa_purifier.py) 开始任务分配。
2. **记录循环**：通过 `process_record` 方法遍历并定位行号，并发分发单行内的切面任务。
3. **单切面重写**：由 `process_planner` 异步任务调用 [PurificationEngine.purify_single_think](file:///d:/REN/qa/core/purification_engine.py#L67) 执行核心思维链净化重构。
4. **事务决策与写回**：对成功与局部成功的 Planner 执行最终合并，重新生成摘要并秒级合并写回原文件。

---

### 3.2 详细步骤与代码实现节点 (枝叶分展)

#### 第一步：安全网关与 RAG 脱敏解析 (FactPack Builder)
* **核心类/方法**：
  * 视角前置核验：[verify_facet_by_small_model](file:///d:/REN/qa/scripts/medicalqa_purifier.py#L42)
  * RAG 标签脱敏：[PurificationEngine.purify_single_think](file:///d:/REN/qa/core/purification_engine.py#L105-L124) 
* **枝叶流转**：
  * 使用小模型网关拦截不合格切面（如“数据不足无法规划”等拒答语），回滚或智能退避修补。
  * 解析 raw refs，剥离其前缀并重组为文献中性标识（如 `[文献_01]`），将 RAG 原始路径物理隔绝在 `provenance_map` 元数据中。

#### 第二步：回答正文重写 (Answer Body Rewriter)与防抖纠偏
* **核心方法**：
  * 重写方法：[rewrite_answer_body](file:///d:/REN/qa/scripts/medicalqa_purifier.py#L279)
  * 硬性除噪：[scrub_engineering_leakage](file:///d:/REN/qa/scripts/medicalqa_purifier.py#L208) 和 [scrub_unsupported_official_identifiers](file:///d:/REN/qa/scripts/medicalqa_purifier.py#L245)
* **枝叶流转**：
  * 先调用轻量大模型将原始 Answer Body 缩窄，剔除偏离主线的病生理推演。
  * 利用 `scrub_engineering_leakage` 强制正则过滤显式工程字眼。
  * 利用 `scrub_unsupported_official_identifiers` 比对 refs 事实，删除未在文献中出现的官方标准代号、注册文号，防止大模型捏造事实，输出作为 CoT 重写的对齐边界。

#### 第三步：CoT 探索性思考链重构 (CoT Purifier)
* **核心方法**：[PurificationEngine.purify_single_think](file:///d:/REN/qa/core/purification_engine.py#L67)
* **枝叶流转**：
  * 传入上一步的已缩窄 `purified_answer_body`（已解决参数空转 Bug）。
  * 动态计算证据等级（HIGH/LOW/NO EVIDENCE），对低证据题启用极简 simplify 模式阻断微观受体脑补。
  * 提示词施加一致性对齐红线，调用高级大模型输出格式高度规范的 exploratory CoT 文本。

#### 第四步：本地多层校验防御门禁 (Validators)
* **核心类/方法**：
  * 结构感知：[is_catastrophic_format_collapse](file:///d:/REN/qa/core/purification_helper.py#L239)
  * 语义洗白：[HealingService.heal_conversational_noise](file:///d:/REN/qa/services/healing_service.py#L24) （由 [HealingService](file:///d:/REN/qa/services/healing_service.py) 提供）
* **枝叶流转**：
  * **AST 结构拦截**：在本地利用 `ast.literal_eval` 与 JSON 正则检测是否残留格式，放行方括号基因型与文献代号，解除误杀。
  * **Trie-Tree/AC 禁词多文本联动校验**：在 [verify_purification.py](file:///d:/REN/qa/scripts/verify_purification.py) 中，同步扫描 `think`、`answer_body`、`summary` 三字段中是否存在泄露禁词。
  * **语义自愈**：利用 `Semantic Wash Map` 对残留代偿词（如实体信息 -> 相关文献记录）进行无损事实伪造的中性转译，正则修复孤立冒号。

#### 第五步：参考引导裁判打分 (LLM Judge)
* **核心类/方法**：[LLMJudgeStrategy.evaluate](file:///d:/REN/qa/strategies/quality_gate/llm_judge.py#L22)
* **枝叶流转**：
  * Judge LLM 接收 Q、planner、purified_think、以及脱敏后的 `Cleaned Facts` 原子事实包。
  * 依据三方裁判细则，若新增事实能被事实包支持，则判定为合规对齐并奖励分数；若不被支持，一票否决扣至 50 分以下，将打分打回并进入重试循环（Feedback Loop）。

#### 第六步：局部事务管理与并发写写回机制 (Transaction Manager)
* **核心控制流**：
  * 事务局部状态决策：在 [medicalqa_purifier.py](file:///d:/REN/qa/scripts/medicalqa_purifier.py#L806) 的局部事务控制分支。
  * 安全并发合并写回：在 [medicalqa_purifier.py](file:///d:/REN/qa/scripts/medicalqa_purifier.py#L969-L981) 处执行并发合并冲突防御。
* **枝叶流转**：
  * 判定 `line_status` 与 `planner_status`，某一切面失败不连坐整行，成功切面可作为 `partial_success` 局部持久化，失败切面隔离到 `purification_failures.jsonl` 中。
  * **Lost Update 并发保护**：在最终写入磁盘前，再次读取一次最新的数据集，与在提纯期间追加的语料行执行安全合并写回，防止并发覆盖。
