# 🩺 多厂商异构共识与混合双轨事实校验网关设计方案 (安全增强版)
## (Multi-Vendor Heterogeneous Consensus & Hybrid Fact-Checking Gate - Enhanced)

在高度严肃的医疗知识图谱与 CoT 提纯工程中，单纯依赖 Graph RAG（图数据库检索）作为对齐指标，会因数据库录入污染（Garbage In, Garbage Out）导致错误事实被强行包装。而单纯依赖大模型参数常识，又会引入难以受控的幻觉。

为了彻底解决该问题，本设计提出**混合双轨事实校验（RAG vs. 临床常识）与多厂商异构模型交叉审计网关**。

---

## 1. 📐 混合双轨审计架构图

```text
                  【输入数据 (Q, CoT, RAG Refs)】
                              │
                              ▼
                【第一阶段：混合事实网关（一审）】
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
          [无冲突：通过]               [检测到冲突！]
                │                           │
                ▼                           ▼
        进入常规提纯流          1. 登记冲突台账 (factual_conflicts_registry.jsonl)
                                2. 网关评估：是否触发异构审计？（高危判定/置信度/白名单）
                                │
                                ▼ [是] (执行二审)
               【第二阶段：多厂商盲审审计网关（二审）】
           (防止一审观点锚定：仅提供 Q, CoT claim, RAG, 权威依据)
                                │
                                ▼ (多维度裁决分类)
                      【二审分类裁判结果】
                                │
        ┌───────────────────┬───┴───────────────────┬───────────────────┐
        ▼                   ▼                       ▼                   ▼
   【RAG_ERROR】    【COT_HALLUCINATION】     【BOTH_ERROR】    【NEED_HUMAN_REVIEW】
        │                   │                       │                   │
        ▼                   ▼                       ▼                   ▼
  比对黄金白名单         打回重写并扣分           挂起人工审查        挂起人工审查
        │
  ┌─────┴─────────────────────────────────────┐
  ▼ [匹配黄金源成功：低风险]                    ▼ [匹配失败/高风险]
 自动物理热修复 (APPLIED)               生成提案挂起 (PENDING_APPROVAL)
```

---

## 2. 🛠️ 核心模块设计

### 2.1 一审：双轨事实校验网关 (Hybrid Fact-Checking Gate)
在提纯质检一审中，裁判模型不仅比对 RAG 引用的一致性，还通过自身的“医学共识”评估 RAG 的合理性：
- **冲突判定**：如果 RAG 引用与临床常识/公理违背（如“小便不利”与“小便自利”），输出 `conflict_detected: true`、冲突描述及修改前后的值，但不直接修改数据库。

### 2.2 冲突注册表 (Factual Conflicts Registry)
一审发现冲突后，写入机读冲突审计文件 `logs/factual_conflicts_registry.jsonl`：
```json
{
  "timestamp": "2026-06-05T18:45:00Z",
  "line_number": 245,
  "question": "甘草干姜茯苓白术汤的典型体征...",
  "target_ref": {
    "source": "refs:《图谱实体集:甘草干姜茯苓白术汤》",
    "context": "肾著之病，其人身体重...小便不利，饮食如常..."
  },
  "conflict_description": "图谱写为'小便不利'，常识为'小便自利'",
  "severity": "CRITICAL",
  "status": "PROPOSED"
}
```

### 2.3 二审：盲审隔离与多厂商异构会审 (Blind Auditing & Multi-Vendor Consensus)
为了防止一审判决的“锚定效应”污染二审，审计网关对二审模型采取**“隔离盲审（Blind Auditing）”**设计：
*   **输入隔离**：二审模型仅接收原始问题、RAG 参考内容、CoT 声称的事实，以及辅助文献。**一审裁判的评分理由和倾向性结论作为附录，不作为主 Prompt 输入**。
*   **模型异构路由**：主模型为 DeepSeek，审计模型强制选用 GLM-5/GPT-4o，实现不同训练集背景的模型互补。
*   **多维度细粒度裁决分类（Taxonomy Expansion）**：
    二审模型不得采用简单的二分类，而必须输出以下六种状态之一：
    - `RAG_ERROR`：RAG 数据源有误，CoT 逻辑正确。
    - `COT_HALLUCINATION`：RAG 正确，CoT 纯属凭空捏造。
    - `BOTH_ERROR`：RAG 与 CoT 均存在严重的医学/事实逻辑错误。
    - `INSUFFICIENT_EVIDENCE`：当前提供的学术材料不足以佐证谁是谁非。
    - `AMBIGUOUS`：医学理论上存在争议，没有绝对对错。
    - `NEED_HUMAN_REVIEW`：复杂临床边界，必须由执业医师介入。

#### 二审审计结构化输出格式
```json
{
  "verdict": "RAG_ERROR" | "COT_HALLUCINATION" | "BOTH_ERROR" | "INSUFFICIENT_EVIDENCE" | "AMBIGUOUS" | "NEED_HUMAN_REVIEW",
  "confidence": 0.98,                  // 裁决置信度 (0.0 - 1.0)
  "supported_by": ["《金匮要略》原文"],  // 二审模型的常识或文献佐证
  "conflicting_span": "小便不利",        // 发生冲突的具体原文片段
  "corrected_fact": "小便自利",         // 建议纠正后的正确值
  "requires_human_review": false,      // 是否需要人工复审标记
  "reason": "详细的临床药理学审视及决策逻辑"
}
```

### 2.4 审计控制与防抖机制（Latency & Cost Control）
异构审计不应该对所有提纯样本触发，以控制 API 成本和响应延迟：
- **触发门禁**：仅在以下情况启动二审网关：
  1. 一审显式检测出 `conflict_detected == true`；
  2. 医学严谨度评分（`medical_rigor_score`）低于阈值但纯净度高（表明模型在极力反驳 RAG 事实）；
  3. 涉及特殊高风险药物或临床毒性病症。

---

## 3. 🛡️ 安全隔离热修复规则（Safety Rules）

1. **绝对禁止纯模型决策直接写库**：
   - 即使二审模型裁决为 `RAG_ERROR`，提案仍默认保存为 `PENDING_APPROVAL`。
2. **黄金标准库精确对齐**：
   - 系统挂载一个只读的“黄金标准白名单”（如：金匮原版原著字符集）。
   - 只有当 `corrected_fact`（拟纠正值）在只读黄金库中存在**100% 精确字符匹配**时，该提案才被允许自动物理应用（`status = 'APPLIED'` 并自动写库）。
   - 如果白名单无此词条，则强制降级为 `PENDING_APPROVAL` 并触发 Webhook 推送给专家审批，严防大模型捏造事实。
3. **缓存重建与失活**：
   - 当提案被 `APPLIED`（无论是自动还是人工批准）并写入 `local_rag.db` 后，系统必须同步清除 `LocalRAGService` 的内存 `metadata_store` 缓存并重构受影响行的 FTS5 虚拟索引，确保后续数据流转读到的是最新的订正值。
