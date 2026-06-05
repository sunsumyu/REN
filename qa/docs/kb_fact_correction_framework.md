# 🩺 企业级知识库事实主动纠错与溯源追踪框架设计规格书
## (Knowledge Base Fact Auto-Correction & Provenance Tracking Framework)

在 `Medical QA Facet Synthesis & CoT Purification` 工业级流水线中，图谱或关系数据库（如 `qa_datasets.db`、`local_rag.db`）作为生成和提纯的“刚性事实锚点”，一旦在源头发生数据录入偏差（例如，将金匮经典条文“小便自利”误录为“小便不利”），该错误会通过对齐红线被提纯引擎强行放大，进而导致生成的高价值微调数据集产生逻辑畸变。

为了实现数据源头的本质净化，本规格书设计了一套**知识库事实主动纠错与热修复溯源框架**。该框架的核心是不妥协于错误数据，而是在提纯阶段主动识别、溯源、纠正数据库中的物理数据，并沉淀审计台账。

---

## 1. 🔍 总体架构图

```text
    【提纯引擎 PurificationEngine】
                │
                ▼ (提取 Active Refs 及文献实体 ID)
    【核监裁判 LLM Judge & Academic Validator】
                │
                ├─────── (检测到 Refs 事实与医学共识/经典白名单冲突？)
                │
                ▼ [是] (触发纠错程序)
    【生成订正提案 FactCorrectionProposal (Pydantic Model)】
                │
                ├─── 记录提案：ID, 错误内容, 修正内容, 溯源路径, 临床依据
                │
                ▼
    【数据库热修复管理器 DBHotpatchManager】 ───> 更新 【本地数据库 qa_datasets.db / local_rag.db】
                │
                ▼ (追加写入审计日志)
    【纠错溯源审计台账 logs/kb_correction_audit.jsonl】
```

---

## 2. 🛠️ 核心模块设计说明

### 2.1 核监网关 (Academic Validator & LLM Judge)
在提纯校验阶段，利用高级大模型作为“事实合规官（Fact Compliance Officer）”。它同时接收主问题 $Q$、切面名称、以及带 `entity_id` / `relationship_id` 的 RAG 原始参考信息。
通过学术共识库和内置中医经典知识（或白名单匹配），对 Refs 数据进行审计。若发现事实偏离，不仅给提纯打回，同时抛出 **`FactCorrectionProposal`** 结构化数据。

#### 订正提案数据结构 (Pydantic Schema)
```python
class FactCorrectionProposal(BaseModel):
    db_type: str                  # 目标数据库 (如 "qa_datasets.db" 或 "local_rag.db")
    table_name: str               # 目标表名 (如 "entities" 或 "relationships")
    target_id: str                # 物理 ID (对应图数据库或 SQLite 中的主键 ID，如 "60704540018180096")
    field_name: str               # 发生错误的字段 (如 "description", "relationship")
    original_value: str           # 原始错误值 (如 "小便不利")
    corrected_value: str          # 拟订正的正确值 (如 "小便自利")
    clinical_evidence: str        # 纠错的医学文献/金匮原文出处及推导理由
    confidence_score: float       # 置信度 (0.0 - 1.0)
```

---

### 2.2 数据库热修复管理器 (DBHotpatchManager)
`DBHotpatchManager` 负责执行具体的物理订正写回。为了防止并发写入导致 SQLite 锁死，该模块采用排他写锁，针对指定的主键 `target_id` 进行热修复 UPDATE 操作。

#### 核心数据库写入操作
```python
import sqlite3
import asyncio

class DBHotpatchManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = asyncio.Lock()

    async def execute_patch(self, proposal: FactCorrectionProposal) -> bool:
        async with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 安全的参数化查询，对指定 ID 字段执行订正更新
                query = f"""
                    UPDATE {proposal.table_name}
                    SET {proposal.field_name} = ?
                    WHERE id = ?
                """
                cursor.execute(query, (proposal.corrected_value, proposal.target_id))
                conn.commit()
                rows_affected = cursor.rowcount
                conn.close()
                
                return rows_affected > 0
            except Exception as e:
                logger.error(f"Failed to hotpatch DB table {proposal.table_name} for ID {proposal.target_id}: {e}")
                return False
```

---

### 2.3 溯源审计台账 (Graveyard & Audit Trail)
每当热修复成功执行后，系统必须向统一审计文件 `logs/kb_correction_audit.jsonl` 追加一条机读日志，并向 `logs/kb_correction_audit.md` 写入一份人类可读的月度/日度账本，包含修改痕迹：

#### 审计日志格式 (JSONL Entry)
```json
{
  "timestamp": "2026-06-05T17:15:30Z",
  "proposal_id": "corr_8923748234",
  "target_id": "60704540018180096",
  "db_type": "qa_datasets.db",
  "table_name": "entities",
  "field": "description",
  "before": "腰部冷痛沉重、饮食如常、口不渴、小便不利、舌淡苔白、脉沉迟或沉缓。",
  "after": "腰部冷痛沉重、饮食如常、口不渴、小便自利、舌淡苔白、脉沉迟或沉缓。",
  "evidence": "《金匮要略·脏腑经络先后病脉证》原文记载：'肾著之病，其人身体重，腰中冷，如坐水中...反不渴，小便自利，饮食如常，病属下焦'。此处误作'不利'，属严重医理错误，现予纠正以保真典型辨证体征。",
  "operator_agent": "Agent-Antigravity-v3.5"
}
```

---

### 2.4 物理版本撤销表与双向回滚机制 (Versioning & Undo Table)
为了确保任何自动或人工的纠错均能 100% 无损撤回与复原，系统在 `qa_datasets.db`（或指定 RAG 数据库）中建立一张**物理版本控制表 `kb_correction_history`**。

#### 物理表结构 (SQLite DDL)
```sql
CREATE TABLE IF NOT EXISTS kb_correction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT UNIQUE NOT NULL,      -- 纠错提案 ID
    db_type TEXT NOT NULL,                  -- 目标数据库名称
    table_name TEXT NOT NULL,               -- 目标表名
    target_id TEXT NOT NULL,                -- 变更记录物理 ID / 主键
    field_name TEXT NOT NULL,               -- 变更字段
    before_value TEXT,                      -- 修正前的原始值 (Pre-Image，重要！)
    after_value TEXT,                       -- 修正后的新值 (Post-Image)
    evidence TEXT,                          -- 纠错理由 / 临床依据
    applied_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'APPLIED'           -- 状态: APPLIED (已应用), ROLLEDBACK (已撤销)
);
```

#### 双向一键回滚物理实现 (Rollback Execution API)
若在二次评估中触发熔断，或者人工审计发现误判，系统或管理员可触发 `rollback_patch`，根据历史记录表中的 `before_value` 逆向将数据覆盖改回：

```python
async def rollback_patch(db_path: str, proposal_id: str) -> bool:
    """
    基于物理版本撤销表，一键回滚指定提案的修改，无损还原数据。
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. 检索提案对应的原始值与目标定位
        cursor.execute(
            "SELECT table_name, field_name, target_id, before_value, status FROM kb_correction_history WHERE proposal_id = ?",
            (proposal_id,)
        )
        row = cursor.fetchone()
        if not row:
            logger.error(f"Rollback failed: Proposal ID {proposal_id} not found in history.")
            return False
            
        table_name, field_name, target_id, before_value, status = row
        if status == 'ROLLEDBACK':
            logger.warning(f"Proposal ID {proposal_id} has already been rolled back.")
            return True
            
        # 2. 逆向 UPDATE，利用 before_value 还原数据
        rollback_query = f"UPDATE {table_name} SET {field_name} = ? WHERE id = ?"
        cursor.execute(rollback_query, (before_value, target_id))
        
        # 3. 更新历史表状态为 ROLLEDBACK
        cursor.execute(
            "UPDATE kb_correction_history SET status = 'ROLLEDBACK' WHERE proposal_id = ?",
            (proposal_id,)
        )
        conn.commit()
        conn.close()
        
        logger.info(f"🎉 Successfully rolled back proposal {proposal_id}. Restored ID {target_id}.{field_name} to pre-image.")
        return True
    except Exception as e:
        logger.critical(f"Failed to execute rollback for proposal {proposal_id}: {e}")
        return False
```

---


## 3. 🔄 提纯链路自愈闭环流程 (Step-by-Step)

1. **RAG 注入**：提纯任务启动，增量读取原始数据，调用 `GraphService` 提取实体 `60704540018180096`（甘草干姜茯苓白术汤）的属性。
2. **纠错网关拦截**：`PurificationEngine` 发起对齐校验。大模型合规官发现其 `description` 存在“小便不利”，依据金匮白名单比对失败，生成 `FactCorrectionProposal`，并将本次提纯任务打回并挂起。
3. **数据库热修复**：系统触发 `DBHotpatchManager` 对 `qa_datasets.db` 进行物理修复，将 `小便不利` 纠正为 `小便自利`，同时写入纠错溯源台账 `logs/kb_correction_audit.jsonl`。
4. **事务回滚与重试**：释放该提纯行的临时缓冲区，将该行标记为 `RETRY_WITH_PATCHED_DB`，并在几秒钟后**重新从数据库发起 RAG 检索**。
5. **重载提纯通过**：第二次运行时，RAG 检索到的数据已是正确的 `小便自利`，CoT 重写和回答正文重写基于正确事实展开，裁判评分通过，高熵纯净数据落盘，自愈闭环完成。

---

## 4. 📈 方案落地收益

*   **数据源质变**：不仅仅净化了微调数据集，更连带清洗了底层的 RAG 向量库与图数据库，实现了**一次纠错，全网受益**。
*   **规避幻觉污染**：防止模型为了强行对齐错误事实，而在 CoT 内部进行诡辩性的因果逻辑推导，确保医学机理 100% 正确。
*   **企业级可审计性**：所有修改有迹可循，随时接受领域专家的合规性抽样审查。

---

## 5. 🛡️ 安全防御与信任域闸门机制 (Safety Gates & Trust Domains)

为了防止大模型将“正确数据误判为错误”或者引入二次幻觉篡改知识库，系统设立了四层刚性安全防御机制，以确保每一次物理修改的绝对正确性：

### 5.1 黄金标准权威白名单校验 (Golden Source Alignment)
- 并非所有纠错提案都能被直接接纳。系统内部挂载了一个只读的“黄金标准共识库”（例如：包含完全结构化、经人工校对的中医方剂原著文本及权威化学药品说明书白名单）。
- 当大模型提出 `FactCorrectionProposal` 后，系统会提取提案中的 `corrected_value`（如 `"小便自利"`) 并在黄金只读库中执行**精确字符匹配**。
- **硬性拦截**：如果拟修改的正确值在黄金标准库中无法匹配，系统会立刻拒绝执行自动 Patch，防止大模型编造新的事实。

### 5.2 多模型隔离交叉会审 (Consensus Cross-Validation)
- 提案在被执行前，会被发送至独立运行、使用不同基座模型（如 `GPT-4o` 与 `DeepSeek-R1` 互斥隔离）的“会审裁判组”。
- 第二方与第三方模型将以独立视角对“纠错合理性与临床证据”进行二次评估。
- **一票否决制**：只有当所有评估模型都给出 `CONFIRMED` 决策，且综合置信度得分 $Confidence\_Score > 0.95$ 时，提案才会获得执行权，任何存在分歧的纠错均会被自动熔断并挂起。

### 5.3 灰度信任域与人机协同网关 (Human-in-the-Loop Gateway)
- 按照风险级别对热修复权限进行灰度分区：
  - **Auto-Apply 域（低风险）**：仅限于原书字词录入错误且在权威黄金库中拥有 100% 精确匹配支持的提案（例如，古书拼写排版错误）。
  - **Human-Approval 域（常规及高风险）**：对于涉及复杂病机分析、药物配伍剂量、用药禁忌等深层临床逻辑的纠错提案，系统**严禁自动修改数据库**。提案将自动推送到人工审核控制台，以 Webhook 形式通知医学专家（Human Link），只有经过执业医师手动一键批准后，物理 UPDATE 才会生效。

### 5.4 原子化事务回滚备份 (Pre-Image Backup & Rollback)
- 在 `DBHotpatchManager` 对数据库执行 UPDATE 写入前的最后一毫秒，系统会自动将目标行的数据导出并存储在 `logs/rollback_backups/` 下，作为“前镜像备份（Pre-Image Backup）”。
- **熔断回滚**：如果热修复写入后，该行在下一次提纯运行时依然无法通过裁判的质量门槛（如打分反而下降，或触发二次冲突），系统将触发原子化 rollback 事务，自动还原数据库目标行至前备份状态，并将该样本强制隔离（Quarantine）转为专家介入处理。

