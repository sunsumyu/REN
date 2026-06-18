# hsa-agent 项目参考价值分析报告

本报告针对 `hsa-agent`（医疗审计智能体核心框架）的架构设计、安全控制、记忆机制及质量门控进行深度剖析，并对比当前 `qa`（医疗问答数据集生成流水线）项目，评估其参考与借鉴价值。

---

## Ⅰ. 核心架构与功能对比

| 维度 | hsa-agent (医疗审计智能体) | qa (医疗问答数据集生成流水线) |
| :--- | :--- | :--- |
| **主要定位** | 针对医保结算数据，通过 LLM 自动生成 SQL、执行查询、自愈纠错并产出专业合规审计报告。 | 基于医学知识图谱（KG）与外部检索，通过多切面（Facet）规划与并行生成，自动化产出高质量多轮医疗问答数据集。 |
| **工作流控制** | 基于 **LangGraph** 的有状态图（StateGraph）机制：`Planner -> Aligner -> SQLEXEC -> (Critic) -> Consolidator -> Reporter`。 | 基于 asyncio 协程并发的线性流水线模型：`计划切面 -> 切面预处理 -> 并行问答 -> 冗余过滤 -> 综合总结 -> 下一轮问题生成`。 |
| **数据校验** | **双阶段硬规则与语义校验**：<br>1. AST 级语法与安全过滤（sqlglot）<br>2. 物理与语义对撞（字段、实体、金额事实对齐） | **双重质检机制**：<br>1. 基础过滤（防拒答与提示词污染拦截）<br>2. 大模型裁判打分（LLM-as-Judge 综合质检） |
| **缓存与记忆** | **三层认知记忆与动作缓存**：<br>1. 内存工作记忆（RAM）<br>2. 基于 FAISS 的成功审计路径/经验库（Episodic）<br>3. 精准 SQL 语义缓存（Action Cache） | **持久化数据库双写**：<br>将质检通过的问答数据同步写入 `medical_qa_dataset.jsonl` 与 SQLite3 数据库 `qa_datasets.db`，包含缓存 `medical_cache.db`。 |

---

## Ⅱ. hsa-agent 对 qa 项目的参考价值

`hsa-agent` 在架构解耦、运行成本优化、物理防御及确定性校验等维度的实践对 `qa` 项目具有极高的借鉴意义。具体表现在以下四个核心方向：

### 1. 智能路由与多级模型分流 (Cost & Latency Optimization)
*   **hsa-agent 机制**：
    *   通过 `FastRouter`（基于关键词与组合模糊匹配）识别“已知规则/算法”，直接执行静态 SQL/算法，完全绕过 LLM。
    *   对于未知问题，通过 `_estimate_complexity` 根据问题长度及分析诉求（如包含“趋势、分布、同环比”等）自动判定复杂度，动态流转至 `LIGHT`（低成本）或 `HEAVY`（强智能）模型。
*   **qa 项目现状**：
    *   目前在 `pipeline_workflow.py` 中，模型 pool（`lightweight` 与 `premium`）是按阶段硬编码锁定的。例如，切面深度问答一律使用 `premium`，而切面规划、过滤、综合等一律使用 `lightweight`。
*   **借鉴价值（⭐️⭐️⭐️）**：
    *   **动态切面路由**：引入意图分类。对于定义清晰、内容简单的医疗实体（如“感冒药的使用禁忌”），不需要硬性拆分为 8 个切面并并发调用 `premium` 模型，可以直接降级使用单切面或完全使用 `lightweight` 快速生成，大幅降低 API 消耗和生成延迟。
    *   **模板化直通车**：对于高度标准化的医学主题，可设计“问答知识模版路由”，跳过 LLM 规划阶段，直接基于模版填充上下文并生成，提升生成速度。

### 2. 精准的确定性事实校验 (Booster & Fact-Checking)
*   **hsa-agent 机制**：
    *   `booster.py` 内置了严苛的确定性数据提取与逻辑校验，废除正则模糊匹配，若数据解析失败则直接抛出异常触发状态机熔断。
    *   `verify_evidence_grounding`（证据溯源校验）：自动提取 LLM 报告中声称的金额、标识符（ID等），在 ClickHouse 返回的原始数据池中进行硬比对。如果报告中胡编了原始数据里没有的数值，则直接拦截。
    *   `verify_semantic_alignment`（语义一致性校验）：检查报告中提及的业务关键词（如科室、性别、年龄段）是否在 SQL 对应的字段中出现。例如，SQL 没查性别字段，但报告里写“男性患者多发”，便会直接拦截。
*   **qa 项目现状**：
    *   在 `main.py` 中，质量网关一律依靠 `ComprehensiveJudgeMetrics`（LLM-as-Judge 裁判）打分判定（必须均分 $\ge 6.0$ 且无防拒答拦截词）。
*   **借鉴价值（⭐️⭐️⭐️⭐️⭐️）**：
    *   **硬规则基准事实比对**：在调用 LLM-as-Judge 裁判之前，可以用 Python 编写轻量级的确定性前置质检。例如，**比对生成回答中的核心医学名词、推荐药物或治疗方案，是否真实存在于 `refs`（检索出的文献与图谱上下文）中**。如果出现了上下文里未包含的药物名称，直接拦截并标记为“幻觉漂移”，无需耗费 Token 让 LLM 裁判去打分。
    *   **回答一致性与多样性监测**：通过语义向量比对同一批次中并行切面回答的重合度，防止不同切面输出高度相似/复读的内容。

### 3. 认知经验沉淀与语义缓存 (Episodic Memory & Action Cache)
*   **hsa-agent 机制**：
    *   `semantic_memory.py` 使用 FAISS 构建本地向量数据库，缓存成功的审计动作（问题-SQL-方法论）。
    *   当新问题进入时，进行 L0 字典匹配、L1 Redis 缓存查找与 L2 语义相似度匹配。相似度 $\ge 0.85$ 时，直接提取缓存数据。
    *   `Consolidator`（固化智能体）设定了门禁：只有**无报错且有数据返回**的路径，才被允许固化进长期记忆。
*   **qa 项目现状**：
    *   目前使用 `qa_datasets.db` 与 `medical_qa_dataset.jsonl` 作为最终数据集落地，暂未在多轮对话生成过程中引入演进式的“生成记忆/成功经验库”。
*   **借鉴价值（⭐️⭐️⭐️⭐️）**：
    *   **生成经验固化**：在批量生成医疗问答数据集时，系统常会遇到生成失败或质检不达标而被丢弃的样本（失败率在 `main.py` 中有统计）。可以建立一个**“优秀范式库”**与**“拦截黑名单库”**。
    *   质检得分极高（例如分流打分 $>9.0$）的样本自动进入 Episodic Memory 库。后续生成类似主题时，将其作为 Few-Shot 示例动态注入 Prompt，实现“越生越聪明，质量越往后越稳定”。
    *   如果某个问题模式导致了 LLM 产生幻觉或拒答，将其特征固化并注入负向提示词（Negative Prompts）。

### 4. 故障诊断与自愈循环 (Critic & Self-Healing Loop)
*   **hsa-agent 机制**：
    *   在有状态图（LangGraph）中加入 `CRITIC` 节点。如果 SQL 校验失败、SQLGuardian 报错，或者 QualityGate 评估响应信心值 $< 0.5$，不直接向用户报错，而是将错误日志挂载回 context，路由给 `reflection_agent.py` 进行诊断修复，直到满足质量门槛或达到最大重试次数。
*   **qa 项目现状**：
    *   `answer_single_facet` 采用传统的循环重试机制（`max_quality_attempts = 2`），若检测不通过，重试时只是简单重新调用 LLM。如果依然失败，则通过降级 prompt 重新调用。若最终仍不通过，则直接丢弃（`DROPPING this facet`）或在主流水线中丢弃该样本。
*   **借鉴价值（⭐️⭐️⭐️⭐️）**：
    *   **带反馈的生成自愈**：当切面问答质检失败时，将质检失败的详细原因（如 LLM 裁判给出的“Grounding 评分过低”或“隔离性差”）作为反馈信息（Feedback）包装进消息链中，再让 LLM 重新生成。这种“带诊断意见的重试”比“盲目重新生成”拥有高得多的自愈成功率，能大幅降低样本的丢弃率（Failed Ratio）。

---

## Ⅲ. 七维度评估指标参考价值分析

通过阅读 `hsa-agent` 的实际评测运行脚本 [run_7metrics_bench.py](file:///E:/chain/hsa-agent/scratch/run_7metrics_bench.py#L600-L641) 可以发现，其内置评测大模型（Judge LLM）实际执行的 **7 维度打分体系**与 README.md 中的宣传表述存在细微差异。其实际评估的 7 个核心维度是：**Success (成功度)**、**Recall (查全率/召回率)**、**Precision (查准率/精确度)**、**Faithfulness (事实忠实度)**、**Relevance (问题相关性)**、**Professionalism (法规专业度)**、**Interpretability (可解释性与证据链)**。

这一套来自工业级真实评测脚本的七维度指标，对 `qa` 项目当前使用的五维综合打分体系（Grounding, Isolation, Explainability, Professionalism, Relevance）具有极高的对齐与启发价值：

### 1. 指标对比与映射关系

| run_7metrics_bench 实际指标 | qa 医学问答指标 | 映射与补充参考价值 |
| :--- | :--- | :--- |
| **Success** (成功度)<br>*评估任务是否完整执行（如 SQL 运行成功、输出完整报告、包含画图要求等）。* | *（暂无直接对应指标，硬编码在 pipeline 外围格式判断中）* | **高参考价值（格式与任务完整性量化）**。<br>问答项目目前通过 `schema_ok`（检测 JSON 括号）判断格式是否异常，如果失败直接丢弃。<br>*借鉴点：可将其泛化为 Success（任务/格式合规率），量化评估模型是否输出了包含特定结构（如 `<think>`、`<facet>` 和特定 JSON 字段）的完整问答，将“格式判定”从外部硬拦截升级为评测维度的分数沉淀。* |
| **Recall** (查全率/召回率)<br>*审计关注是否漏掉可疑记录。对无发现报告强制审查是否包含扫描行数、时间范围等边界证据证明。* | *（暂无直接对应指标）* | **极高参考价值（无发现/拒答的严谨度控制）**。<br>在医疗问答中，如果外部检索无相关文献，模型容易触发“拒答”或直接输出“无法回答”。<br>*借鉴点：可引入 Recall 指标，当模型回答“无相关指南”或“无发现”时，审查其是否输出了合理的扫描证据链（例如：说明核查了哪些文献和图谱实体后未能找到依据，而不是单薄的一句‘我不知道’）。* |
| **Precision** (查准率/精确度)<br>*审查被标记的违规是否真实准确，是否存在胡乱推测。* | *（暂无直接对应指标，混在 Grounding 中）* | **中参考价值（提高事实审查分辨率）**。<br>在 `qa` 中，可将其与 Grounding 解耦。Grounding 关注“是否编造了 refs 之外的数据”；而 Precision 关注“在基于 refs 的前提下，医学推论、剂量匹配或药效陈述是否百分百精确无误”。这有助于区分“无心幻觉”和“逻辑推理错误”。 |
| **Faithfulness** (事实忠实度)<br>*每一项主张均可追溯到 actual SQL 输出。* | **Grounding** (事实忠实度)<br>*核对 refs 背景事实，严重扣分捏造实体。* | **完全等价**。均评估回答是否完全忠实于给定的物理事实（SQL 结果 vs. `refs` 文献背景）。 |
| **Relevance** (相关性)<br>*直接回答问题并提供具体行动建议。* | **Relevance** (相关性)<br>*回答直击核心诉求，过滤无意义套话。* | **完全等价**。均评估回答的相关性。 |
| **Professionalism** (专业度)<br>*是否正确引用法规文号（如医保发〔2021〕48号）、使用 ICD-10/DRG 术语。* | **Professionalism** (专业性)<br>*术语是否规范、整体用词是否像临床专家。* | **高参考价值（评估标准具象化）**。<br>`qa` 的 Professionalism 目前偏主观。<br>*借鉴点：可使指标刚性化，评估回答中是否包含具体的国家级诊疗规范、中国药典标准或明确的医学术语规范（如使用标准疾病名称而非通俗口语）。* |
| **Interpretability** (可解释性)<br>*拥有清晰的证据链，且对特定任务输出可视化辅助（如 Mermaid 拓扑图、ASCII 时序图、对比表）。* | **Explainability** (可解释性)<br>*答案有清晰的逻辑推导和证据来源引用（如：根据《XX说明书》）。* | **高参考价值（强化可视化推理）**。<br>`qa` 的 Explainability 仅关注文本逻辑。<br>*借鉴点：在生成复杂的联合用药禁忌、时序病程演变等高级问答时，可引入 Interpretability 维度，要求模型利用 ASCII 表格或时序步骤直观展示药代动力学过程或诊断递进逻辑。* |

---

## Ⅳ. 幻觉控制与废话拦截优化方案（基于 hsa-agent 的实践）

为了让 `qa` 项目的输出更加稳定、更少出现事实性幻觉，并且在生成端减少非 think 推理内容的客套废话与冗余表达，结合 `hsa-agent` 的工程实践，可采取以下两阶段优化策略：

### 1. 清洗前：生成端的刚性控制（减少废话与事实错误）

#### 策略 A：采用 GSSC 尾部锚定注入（缓解幻觉）
*   **hsa-agent 做法**：将所有 Schema 规范、字段 DDL、专家经验库（FAISS 召回内容）全部格式化并**物理挂载到消息链的末尾**（如 `<database_schema_context>`），并在系统提示词中警告模型“滚动至最尾部查看”。
*   **对 qa 的改进**：在 [pipeline_workflow.py](file:///e:/chain/QA/qa/core/pipeline_workflow.py) 组装 Prompt 时，**将 `refs` 参考文献和知识图谱数据作为独立标签块，强制放置在 User Message 的最后**。利用 LLM 的“近因偏差”（Recency Bias），使模型在生成时对文献事实保持最高关注度，避免中途遗忘或捏造剂量/药名。

#### 策略 B：引入动态临床刚性红线（防事实性错误）
*   **hsa-agent 做法**：针对特定关键词（如“牙科”、“透析”）注入场景化 SQL 刚性防漂移红线。
*   **对 qa 的改进**：根据输入问题的疾病或药物类别，动态注入医学知识红线。例如：
    ```text
    【⚠️ 刚性临床约束：儿科/妊娠期用药安全】
    1. 必须根据儿童年龄或体重详细说明剂量计算依据，严禁含糊其辞。
    2. 严禁推荐成人专用或该人群禁忌的药物（如喹诺酮类、四环素类），否则直接判定不合格。
    ```

#### 策略 C：系统提示词刚性样式脱敏（防前置客套废话）
*   在 `answer_single_facet` 的系统提示词中追加**样式红线**，消除“好的”、“针对您的问题解答如下”等前置过渡废话：
    ```text
    【输出格式硬红线（消除前导废话）】
    1. 在 </think> 标记之后，你必须直接、即刻输出专业解答，绝对禁止包含任何前置客套话、引言、自我介绍或过渡句。
    2. 常见禁用前导词包括但不限于：“针对您的问题...”、“好的，下面为您...”、“以下是为您的解答...”。
    3. 违反此项规定将直接触发系统质检拦截并重新生成。
    ```

---

### 2. 清洗与质检：后置的多级拦截与自愈（有效拦截并低成本清洗）

#### 策略 A：轻量级 Python 事实指针对碰撞（借鉴 `verify_evidence_grounding`）
*   **hsa-agent 做法**：提取报告中的金额和 ID，在 ClickHouse 原始数据中执行硬搜索，存在未命中实体则直接报错。
*   **对 qa 的改进**：编写前置校验函数，提取生成回答中的数字、单位（如 `500mg`、`每日3次`）和专有名词（药名），对 `refs` 字符串执行快速布尔查找：
    ```python
    # 伪代码：事实硬比对
    for keyword in extracted_clinical_entities:
        if keyword.lower() not in raw_refs_pool.lower():
            return False, f"检测到幻觉实体 '{keyword}' 未出现在参考资料中，拒绝学习。"
    ```
    此操作完全运行在 CPU 本地，零 Token 成本，能秒级拦截低级事实性错误，避免其流入 LLM 裁判。

#### 策略 B：多级过滤与智能客套话前置切除（降低废话拦截成本）
*   在 [answer_guard.py](file:///e:/chain/QA/qa/strategies/quality_gate/answer_guard.py) 的 `check_answer_quality` 中，我们可以采取**“先容错切除，后硬性拦截”**的双重机制：
    1.  **第一步：无损物理切除**。对于以“好的，分析如下：”开头的文本，不直接报错重试，而是利用正则自动将其剔除，保留后续的专业内容，节省重试 API 的 Token：
        ```python
        def sanitize_convo_filler(text: str) -> str:
            filler_pattern = r"^(好的[，、]?|您好[，！]?|以下是为您的解答[：:]?|针对您的问题分析如下[：:]?|针对您的问题，我给出的回答是[：:]?)\s*"
            return re.sub(filler_pattern, "", text.strip())
        ```
    2.  **第二步：对残留的严重污染执行拦截（反馈自愈）**。若切除后仍有“作为一个AI”等特征词，则拦截并带上精准诊断意见（Feedback Loop）返回给 Critic，使其进行针对性重写，而不是盲目重试。

---

## Ⅴ. 总结与落地建议

### 结论
`hsa-agent` 是一个高可靠、高防御性的**生产级应用系统**，其核心设计理念是**“用确定性规则（AST、物理对齐、硬指标提取）构筑红线，仅将推理交给 LLM，并通过智能路由与语义缓存将成本和幻觉降到最低”**。

对于以**生成多样化、高质量数据**为目标的 `qa` 项目，**不应盲目照搬其“完全跳过 LLM（Fast Route）”的防多样性机制**，但其在以下两个维度的工程设计极具落地价值：
1.  **确定性事实对比前置（借鉴 `verify_evidence_grounding`）**：在 LLM 裁判质检前，使用 Python 字符串/向量匹配，强制校验生成的回答是否飘出了 `refs` 参考范围，减少 LLM 裁判的调用频次，降低 Token 成本。
2.  **带诊断意见的自愈生成（借鉴 `CRITIC` 循环）**：在 QA 流水线重试时，将前一次失败的评测指标及原因注入 Prompt，使模型能够针对性修正，从而提升数据集生成的整体通过率。

---

## Ⅵ. 基于 qa 实际 codebase 的合理性验证与落地优先级

根据当前 `qa` 项目源码及相关诊断文件的静态审查，对上述分析报告进行合理性验证，并确立具体的落地执行路径：

### 1. 代码事实深度印证

*   **Judge "盲评" 事实确认**：[llm_judge.py:L22](file:///d:/REN/qa/strategies/quality_gate/llm_judge.py#L22) 的 `evaluate` 接口确实未接收任何 `refs` 上下文。这导致裁判在进行大模型打分时，缺乏客观的事实参考源，极易把真实的推理误判为幻觉。**此处分析完全合理，建议立即改造。**
*   **方括号硬拦截问题**：在 [purification_helper.py:L241](file:///d:/REN/qa/core/purification_helper.py#L241) 中，`invalid_chars` 确实包含了 `[` 和 `]` 字符：
    ```python
    invalid_chars = ['{', '}', '[', ']', '",', '我决定构建', '步骤1', '阶段一']
    ```
    这直接导致如 `[CYP2C9*3]`、`[1]` 等标准医学术语被格式网关强行拦截。**此处分析完全合理，应替换为结构感知型检测器。**
*   **重试反馈缺陷**：在生成阶段的 [pipeline_workflow.py:L414](file:///d:/REN/qa/core/pipeline_workflow.py#L414) 中，`answer_single_facet` 在重试时并未将失败诊断信息回灌入 `messages` 中，而提纯阶段在 [purification_engine.py:L285](file:///d:/REN/qa/core/purification_engine.py#L285) 已有 `feedback_prompt = feedback_msg` 的机制。**此处分析极其精准，生成重试自愈机制确实应当增强。**
*   **GSSC 污染验证**：[purification_engine.py:L116](file:///d:/REN/qa/core/purification_engine.py#L116) 的 `anchors` 拼装代码如下：
    ```python
    anchors.append(f"- [{src}] {clean_ctx}")
    ```
    这确实暴露了 `source`（即 `src`）的库名称或 RAG 标记，易导致噪声泄漏。**此处分析合理，必须对注入源头执行文献脱敏。**

### 2. 修正与纠偏点

*   **模型路由已具备部分治理**：文档指出“qa 目前模型分流完全硬编码”，但这在当前 codebase 中不完全准确。在 [pipeline_workflow.py:L640](file:///d:/REN/qa/core/pipeline_workflow.py#L640) 中已引入了 `EvidenceScopeRouter`、切面治理策略（如 `RedirectToSimpleStrategy`）和极简模式（`simplify`）的分流机制。因此，模型路由优化的优先级可以适度降低。
*   **Professionalism 评估刚性化去偏**：如果大模型在生成时被强行灌注了“必须引用具体药典/指南”的硬指标，但在原始 `refs` 事实包中并没有这些权威指南，模型为通过质检会去伪造文献出处。因此，必须对专业性（Professionalism）指标去刚性化，保持 `FactPack` 作为事实边界的唯一性，不允许无中生有的“编造式引用”。

### 3. 具体实施落地优先级路线

根据故障的严重程度与开发难度，制定如下三阶段落地计划：

```mermaid
graph TD
    A[第一步: 阻断误杀与脱敏 (基础止血)] --> B[Structure Gate 去除方括号硬拦截]
    A --> C[RAG Source 注入源头脱敏]
    B & C --> D[第二步: 架构演进与事实对齐]
    D --> E[Reference-Guided Judge 循证裁判]
    D --> F[Answer/Think 边界物理对齐与传参修复]
    E & F --> G[第三步: 进阶自愈与缓存]
    G --> H[生成阶段带诊断反馈的重试自愈]
    G --> I[语义缓存与动态 Few-shot 沉淀]
```

1.  **第一阶段：阻断误杀与脱敏（基础止血，立即实施）**
    *   **Structure Gate 升级**：修改 [purification_helper.py:L241](file:///d:/REN/qa/core/purification_helper.py#L241)，移除 `[` 和 `]` 的直接拦截，换成结构敏感检测，挽救被误杀的基因型等专业术语。
    *   **RAG Source 注入源头脱敏**：修改 [purification_engine.py:L116](file:///d:/REN/qa/core/purification_engine.py#L116)，将 `[{src}]` 脱敏为中性的文献序号（如 `[文献_01]`），防止工程标签进入模型。
2.  **第二阶段：架构演进与对齐（提升质量网关准确性）**
    *   **Reference-Guided Judge**：重构 [llm_judge.py:L22](file:///d:/REN/qa/strategies/quality_gate/llm_judge.py#L22)，把脱敏后的事实上下文 `CleanedFact` 作为 Ground Truth 输入给裁判模型。
    *   **Answer/Think 边界对齐**：修复 `purify_single_think` 中 `purified_answer` 空转的 Bug，将其作为事实边界硬性灌入 CoT 生成模型。
3.  **第三阶段：自愈与优化（降本增效）**
    *   **带诊断反馈的重试自愈**：在 [pipeline_workflow.py:L414](file:///d:/REN/qa/core/pipeline_workflow.py#L414) 中重构 `answer_single_facet`，在失败重试时，将错误扣分诊断回灌到消息链中。
    *   **语义缓存与 Few-shot 沉淀**：引入语义相似度，对高质量（打分 $>9.0$）的生成实例沉淀为动态示例，对低分失败模式进行拉黑规避。关于成功与失败经验双向沉淀的详细架构设计与代码蓝图，请参见 [episodic_and_failure_memory_design.md](file:///d:/REN/qa/docs/episodic_and_failure_memory_design.md)。

