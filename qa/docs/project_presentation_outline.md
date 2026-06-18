# 医疗 Think CoT QA 数据工程系统 - 演示大纲与核心逻辑

本文档按照 10 个核心维度重新组织了项目逻辑，特别适合用于项目汇报、技术评审（PPT 讲解）或快速梳理系统全貌。

## 1. 项目定位：医疗 Think CoT 数据工程系统
- **目标**：构建一个面向医疗推理大模型微调（SFT / Think CoT）的自动化数据生成与清洗工厂。
- **核心理念**：不是简单地“让大模型多写答案”，而是确保每条训练数据具备“真实的临床推理深度”，且绝对受控于“医学事实边界”。
- **产出物**：100% 可追溯、无严重幻觉、具有长思维链（CoT）的高质量医疗 QA 数据集。

## 2. 核心痛点：幻觉、浅问题、不可追踪失败
- **痛点一：问题太浅**。常规生成往往问“用法用量是什么”，缺乏 Think CoT 需要的复杂临床冲突场景。
  - *解决方案*：基于多实体图谱聚合与问题复杂度门控（`is_fact_retrieval_question()`）过滤单跳事实查询。
- **痛点二：幻觉与无证据外推**。模型极易在思维链中自行脑补药代动力学、分子机制或不相关的临床试验。
  - *解决方案*：引入严格的“证据契约（Evidence Contract）”，明确 Allowed Facts 和 Forbidden Expansions。
- **痛点三：失败不可追踪**。流水线报错后往往只留下一个 Failed，难以复盘定位。
  - *解决方案*：建设细粒度的单任务 Outcome 审计（`write_task_outcome()`）与拒绝报告生成体系。

## 3. 全局架构：Generation + Purification 双轨
项目整体分为两条相互独立的管线：
- **Generation Pipeline（生成管线）**：从知识图谱与医学文献出发，生成多视角、带 Think 的 QA 候选数据，并进行质量门打分和隔离。
- **Purification Pipeline（提纯管线）**：对生成并通过初筛的数据，进行离线“思维链净化”，剔除 JSON Schema、系统 prompt 等工程痕迹与废话，打磨为可直接用于模型训练的纯净文本。
- *代码落点*：`main.py`（生成管线入口）、`scripts/medicalqa_purifier.py`（提纯管线入口）。

## 4. 生成主流程：图谱 refs 到多视角 QA
- **图谱驱动**：通过 `GraphService.fetch_random_knowledge_graph()` 获取随机的疾病、药物及它们之间的临床关系实体。
- **三级证据检索**：基于图谱实体，触发本地 RAG（Tier 1）、外部 API（Tier 2）和受限搜索（Tier 3），组装成上下文 `context_list` 和证据池 `refs`。
- **问题生成**：利用 `context_list` 引导大模型提出包含临床决策冲突的复杂问题。
- **多视角切面（Facet）回答**：同一个问题并发多个 Agent，分别从“药代动力学”、“禁忌人群”、“临床决策”等不同切面进行独立回答生成。

## 5. 问题与 facet 治理：如何保证推理深度
- **问题门控**：拦截诸如“某药有哪些不良反应”这类纯说明书搬运问题，强迫模型生成带条件的推演题。
- **Facet 兼容性治理（Q-Facet Governance）**：避免模型强行套用不适合的分析框架。
  - *动作分类*：`KEEP`（保留）、`RENAME`（重命名为贴切切面）、`DROP`（偏题剪枝）、`REDIRECT_SIMPLE`（简单事实题启用极简推理，防止长篇大论的脑补）。
  - *代码落点*：`FacetGovernanceFilter.evaluate_compatibility()`。

## 6. 证据契约：如何限制无证据外推
此为控制大模型幻觉的“杀手锏”。
- **证据路由**：`EvidenceScopeRouter.route_references()` 将海量 `refs` 过滤并划分为 `CORE`、`BOUNDARY`、`BLOCKED`。
- **契约生成**：`build_evidence_contract()` 将有效证据固化为 `allowed facts`，并明确生成 `forbidden expansions`（例如：不得引入 refs 未提及的 CYP 代谢途径或换药方案）。
- **约束生成**：在生成回答时，将证据契约作为系统 Prompt 强力注入。

## 7. 后置硬拦截：模型不听 prompt 时如何处理
当大模型在生成中依然“放飞自我”违反契约时：
- **越界检测**：`detect_forbidden_expansion()` 在大模型输出后，使用小模型或正则逻辑对回答文本进行后置扫描。
- **异常隔离**：若某一样本在 Summary 阶段连续 3 次尝试均被检测到违反证据契约，系统将直接抛出 `SampleQuarantineException` 业务异常。
- **硬性阻断**：问题严重的数据绝不流入下游，直接落入隔离区（Quarantine）。

## 8. Quality Gate：如何评分、拒绝和定位失败
所有生成完毕的候选样本必须经过“综合质量门（Quality Gate）”的终审评判。
- **多维度评分**：使用法官模型（Judge LLM）从 `success`、`recall`（召回关键证据）、`precision`（表述精确）、`faithfulness`（忠实证据）、`isolation`（无工程污染）和 `complexity`（推理深度）等 9 个维度进行 0-10 分的独立打分。
- **失败定位**：通过 `evaluate_facets_for_rejected_sample()` 深入剖析是因为哪个切面的失败拖累了全局。
- *代码落点*：`ComprehensiveJudgeMetrics`、`format_dataset_for_quality_judge()`。

## 9. 审计日志：如何复盘每个 Task
建立完备的追溯机制，不放过任何一个失败样本的根因。
- **`generation_task_outcomes.jsonl`**：记录每个 Task 的最终状态（passed、quality_rejected、quarantined、exception）及耗时和原因。
- **`generation_audit.jsonl`**：记录 Facet 治理、证据契约建立、违规拦截等全链路中间状态。
- **`generation_rejections.md`**：为被质量门拒绝的样本生成人类可读的“拒稿报告”，方便开发人员迭代 Prompt。

## 10. 当前成果与下一步：从可用到企业级
- **当前状态**：已经跑通了由图谱驱动、具备严格证据约束、拥有独立质量门禁和完备审计体系的双轨（生成+提纯）数据工厂，基本实现了“可用”目标。
- **下一步演进**：
  - **证据细粒度绑定**：将 Answer 中的具体推理判断精准绑定到对应的 `fact_id`，实现句子级的防幻觉。
  - **提纯管线深度联动**：把证据契约的约束力进一步贯穿到离线 Purification 环节，让洗出的 CoT 数据不仅格式干净，且事实边界坚如磐石。
  - **可视化与数据看板**：基于现有的 JSONL 审计日志，构建实时监控 Dashboard，提升人工抽检与反馈闭环效率。

  ---

## 11. 推荐 PPT 讲解顺序

如果把本文档改成 PPT，建议 10 页讲清楚：

1. 项目定位：医疗 Think CoT 数据工程系统。
2. 核心痛点：幻觉、浅问题、不可追踪失败。
3. 全局架构：Generation + Purification 双轨。
4. 生成主流程：图谱 refs 到多视角 QA。
5. 问题与 facet 治理：如何保证推理深度。
6. 证据契约：如何限制无证据外推。
7. 后置硬拦截：模型不听 prompt 时如何处理。
8. Quality Gate：如何评分、拒绝和定位失败。
9. 审计日志：如何复盘每个 Task。
10. 当前成果与下一步：从可用到企业级。
