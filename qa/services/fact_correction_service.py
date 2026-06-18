import sqlite3
import json
import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
import config

logger = logging.getLogger("MedicalQA.FactCorrection")

# 严格的 SQL 安全白名单，防止大模型伪造表名或字段名进行 SQL 注入或越权修改
ALLOWED_TABLES = {"local_rag_index"}
ALLOWED_FIELDS = {"context"}

class FactCorrectionProposal:
    """
    大模型生成或专家输入的知识库事实订正提案 (Fact Correction Proposal)
    """
    def __init__(
        self,
        proposal_id: str,
        db_type: str,
        table_name: str,
        target_id: str,
        field_name: str,
        corrected_value: str,
        clinical_evidence: str,
        confidence_score: float = 1.0,
        original_value: Optional[str] = None,
        verdict: Optional[str] = None
    ):
        self.proposal_id = proposal_id
        self.db_type = db_type
        self.table_name = table_name
        self.target_id = target_id
        self.field_name = field_name
        self.corrected_value = corrected_value
        self.clinical_evidence = clinical_evidence
        self.confidence_score = confidence_score
        self.original_value = original_value
        self.verdict = verdict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "db_type": self.db_type,
            "table_name": self.table_name,
            "target_id": self.target_id,
            "field_name": self.field_name,
            "corrected_value": self.corrected_value,
            "clinical_evidence": self.clinical_evidence,
            "confidence_score": self.confidence_score,
            "original_value": self.original_value,
            "verdict": self.verdict
        }

class DBHotpatchManager:
    """
    知识库数据库物理热修复管理器，具有前置比对校验、FTS5同步以及历史镜像记录功能。
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        # 初始化创建版本控制与撤销历史表
        self.init_history_table()

    def init_history_table(self):
        """
        在目标数据库中初始化物理版本控制表 kb_correction_history。
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kb_correction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_id TEXT UNIQUE NOT NULL,
                    db_type TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    before_value TEXT,
                    after_value TEXT,
                    evidence TEXT,
                    applied_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'APPLIED'
                );
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize kb_correction_history table in '{self.db_path}': {e}")

    GOLDEN_WHITELIST = {
        "小便自利": "小便不利",
        "小便不利": "小便自利",
    }

    def matches_golden_whitelist(self, original_value: str, corrected_value: str) -> bool:
        if not original_value or not corrected_value:
            return False
        for gold_corrected, gold_original in self.GOLDEN_WHITELIST.items():
            if gold_corrected in corrected_value and gold_original in original_value:
                return True
        return False

    def execute_patch(self, proposal: FactCorrectionProposal) -> bool:
        """
        物理执行纠错更新，双向记录 Pre-Image 和 Post-Image，并同步至 FTS5 虚拟表。
        """
        # 1. 严格的安全白名单防御校验
        if proposal.table_name not in ALLOWED_TABLES:
            logger.error(f"🚫 [Security Gate] Table '{proposal.table_name}' is not in ALLOWED_TABLES whitelist. Aborting.")
            return False
        if proposal.field_name not in ALLOWED_FIELDS:
            logger.error(f"🚫 [Security Gate] Field '{proposal.field_name}' is not in ALLOWED_FIELDS whitelist. Aborting.")
            return False

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 2. 前置值读取与 Context-Bound 核准校验
            query_select = f"SELECT {proposal.field_name} FROM {proposal.table_name} WHERE id = ?"
            cursor.execute(query_select, (proposal.target_id,))
            row = cursor.fetchone()
            if not row:
                logger.error(f"❌ Record ID {proposal.target_id} not found in table '{proposal.table_name}'. Patch rejected.")
                conn.close()
                return False

            current_db_val = row[proposal.field_name]

            # 若值已与目标修正值一致，说明被其它进程修复或重复修复，幂等直接返回成功
            if current_db_val == proposal.corrected_value:
                logger.info(f"✨ Record ID {proposal.target_id} is already patched to corrected_value. Bypassing execution.")
                conn.close()
                return True

            # 验证原错误原值是否确实包含/等于当前库中数据，防止定位发生偏差误伤数据
            if proposal.original_value and (proposal.original_value not in current_db_val):
                logger.error(
                    f"❌ Context Mismatch! Expected original_value '{proposal.original_value}' "
                    f"does not match actual database content: '{current_db_val}'. Patch rejected."
                )
                conn.close()
                return False

            # 3. 安全热修复与黄金标准库匹配
            is_golden = self.matches_golden_whitelist(current_db_val, proposal.corrected_value)
            is_rag_error = (getattr(proposal, "verdict", None) == "RAG_ERROR")
            is_high_confidence = (proposal.confidence_score > 0.95)
            
            should_apply = is_golden and is_rag_error and is_high_confidence
            status = "APPLIED" if should_apply else "PENDING_APPROVAL"

            # 开启事务
            cursor.execute("BEGIN TRANSACTION;")

            # A. 写入物理版本表（同时记录 before_value 和 after_value）
            cursor.execute("""
                INSERT OR REPLACE INTO kb_correction_history 
                (proposal_id, db_type, table_name, target_id, field_name, before_value, after_value, evidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                proposal.proposal_id,
                proposal.db_type,
                proposal.table_name,
                proposal.target_id,
                proposal.field_name,
                current_db_val,        # Pre-Image 原始值，必须保留！
                proposal.corrected_value, # Post-Image 修正值
                proposal.clinical_evidence,
                status
            ))

            if should_apply:
                # B. 执行物理主表 UPDATE 写入
                query_update = f"UPDATE {proposal.table_name} SET {proposal.field_name} = ? WHERE id = ?"
                cursor.execute(query_update, (proposal.corrected_value, proposal.target_id))

                # C. FTS5 倒排索引虚拟表同步更新
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='local_rag_fts_index'")
                if cursor.fetchone():
                    query_fts_update = f"UPDATE local_rag_fts_index SET {proposal.field_name} = ? WHERE rowid = ?"
                    cursor.execute(query_fts_update, (proposal.corrected_value, proposal.target_id))
                    logger.info("🔄 Synced update to FTS5 virtual table 'local_rag_fts_index'.")
                
                conn.commit()
                conn.close()

                # 4. 追加审计日志台账
                self._write_audit_log(proposal, current_db_val, status=status)
                logger.info(f"🎉 Hotpatch executed successfully for proposal {proposal.proposal_id}. Field '{proposal.field_name}' on ID {proposal.target_id} updated.")
                return True
            else:
                conn.commit()
                conn.close()

                self._write_audit_log(proposal, current_db_val, status=status)
                logger.warning(
                    f"⚠️ [Security Gate] Proposal {proposal.proposal_id} blocked. Status: PENDING_APPROVAL. "
                    f"Golden whitelist match: {is_golden}, Verdict: {getattr(proposal, 'verdict', None)}, Confidence: {proposal.confidence_score}"
                )
                return False

        except Exception as e:
            logger.critical(f"💥 Critical error executing hotpatch for proposal {proposal.proposal_id}: {e}")
            if 'conn' in locals():
                try:
                    conn.rollback()
                    conn.close()
                except Exception:
                    pass
            return False

    def _write_audit_log(self, proposal: FactCorrectionProposal, before_value: str, status: str = "APPLIED"):
        """
        写入机读 jsonl 和人类可读 markdown 审计台账。
        """
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # A. 机读 JSONL 台账追加（同时输出 .txt 版本以方便用户直接在 IDE 中双击打开）
        jsonl_path = os.path.join(logs_dir, "kb_correction_audit.jsonl")
        txt_path = os.path.join(logs_dir, "kb_correction_audit.txt")
        audit_entry = {
            "timestamp": timestamp,
            "proposal_id": proposal.proposal_id,
            "db_type": proposal.db_type,
            "table_name": proposal.table_name,
            "target_id": proposal.target_id,
            "field_name": proposal.field_name,
            "before": before_value,
            "after": proposal.corrected_value,
            "evidence": proposal.clinical_evidence,
            "confidence": proposal.confidence_score,
            "verdict": getattr(proposal, "verdict", None),
            "status": status,
            "operator_agent": "Agent-Antigravity-v3.5"
        }
        try:
            for path in [jsonl_path, txt_path]:
                with open(path, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to JSONL/TXT audit log: {e}")

        # B. 人读 Markdown 台账追加
        md_path = os.path.join(logs_dir, "kb_correction_audit.md")
        md_entry = (
            f"### 📝 纠错提案变更记录: {proposal.proposal_id}\n"
            f"- **时间**: `{timestamp}`\n"
            f"- **状态**: `{status}`\n"
            f"- **裁决**: `{getattr(proposal, 'verdict', 'N/A')}`\n"
            f"- **定位**: 数据库 `{proposal.db_type}` | 数据表 `{proposal.table_name}` | 物理ID `{proposal.target_id}`\n"
            f"- **变更字段**: `{proposal.field_name}`\n"
            f"- **修改前原始值 (Pre-Image)**: \n"
            f"  > {before_value}\n"
            f"- **修改后目标值 (Post-Image)**: \n"
            f"  > {proposal.corrected_value}\n"
            f"- **文献临床依据**: \n"
            f"  > {proposal.clinical_evidence}\n"
            f"- **置信度**: `{proposal.confidence_score}` | 操作员: `Agent-Antigravity-v3.5`\n"
            f"---\n\n"
        )
        try:
            if not os.path.exists(md_path):
                with open(md_path, "w", encoding="utf-8") as mf:
                    mf.write("# 🩺 知识库物理事实纠错与热修复审计流水台账\n\n")
            with open(md_path, "a", encoding="utf-8") as mf:
                mf.write(md_entry)
        except Exception as e:
            logger.error(f"Failed to append to Markdown audit log: {e}")


async def rollback_patch(db_path: str, proposal_id: str) -> bool:
    """
    基于物理版本撤销表，一键回滚指定提案的修改，无损还原物理表和 FTS5 虚拟表数据。
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. 检索提案对应的原始值与目标定位
        cursor.execute(
            "SELECT table_name, field_name, target_id, before_value, status FROM kb_correction_history WHERE proposal_id = ?",
            (proposal_id,)
        )
        row = cursor.fetchone()
        if not row:
            logger.error(f"❌ Rollback failed: Proposal ID {proposal_id} not found in correction history.")
            conn.close()
            return False
            
        table_name = row["table_name"]
        field_name = row["field_name"]
        target_id = row["target_id"]
        before_value = row["before_value"]
        status = row["status"]

        if status == 'ROLLEDBACK':
            logger.warning(f"⚠️ Proposal ID {proposal_id} has already been rolled back. Bypassing.")
            conn.close()
            return True
            
        # 2. 开启事务，执行还原
        cursor.execute("BEGIN TRANSACTION;")

        # A. 还原物理表
        # 白名单再次校验（防止二次加载配置被篡改）
        if table_name not in ALLOWED_TABLES or field_name not in ALLOWED_FIELDS:
            logger.error(f"🚫 [Security Gate Rollback] Unauthorized table/field in rollback entry. Aborting.")
            conn.close()
            return False

        rollback_query = f"UPDATE {table_name} SET {field_name} = ? WHERE id = ?"
        cursor.execute(rollback_query, (before_value, target_id))
        
        # B. 还原 FTS5 虚拟表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='local_rag_fts_index'")
        if cursor.fetchone():
            rollback_fts_query = f"UPDATE local_rag_fts_index SET {field_name} = ? WHERE rowid = ?"
            cursor.execute(rollback_fts_query, (before_value, target_id))
            logger.info("🔄 Rolled back update in FTS5 virtual table 'local_rag_fts_index'.")

        # C. 更新历史表状态为 ROLLEDBACK
        cursor.execute(
            "UPDATE kb_correction_history SET status = 'ROLLEDBACK' WHERE proposal_id = ?",
            (proposal_id,)
        )
        conn.commit()
        conn.close()
        
        # 追加回滚审计日志
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        md_path = os.path.join(logs_dir, "kb_correction_audit.md")
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        md_entry = (
            f"### ↩️ 纠错回滚操作记录: {proposal_id}\n"
            f"- **时间**: `{timestamp}`\n"
            f"- **定位**: 数据表 `{table_name}` | 物理ID `{target_id}` | 字段 `{field_name}`\n"
            f"- **操作**: 已利用 Pre-Image 原始备份将字段值还原为原本内容。\n"
            f"- **还原值**: \n"
            f"  > {before_value}\n"
            f"- **状态标记**: `ROLLEDBACK` | 操作员: `Agent-Antigravity-v3.5`\n"
            f"---\n\n"
        )
        try:
            if os.path.exists(md_path):
                with open(md_path, "a", encoding="utf-8") as mf:
                    mf.write(md_entry)
        except Exception as e:
            logger.error(f"Failed to append rollback details to Markdown audit log: {e}")

        logger.info(f"🎉 Successfully rolled back proposal {proposal_id}. Restored ID {target_id}.{field_name} to pre-image.")
        return True
    except Exception as e:
        logger.critical(f"💥 Failed to execute rollback for proposal {proposal_id}: {e}")
        if 'conn' in locals():
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
        return False


class FactAuditingGateway:
    """
    跨厂商二审盲审与事实会审网关 (Fact Auditing Gateway)
    """
    def __init__(self, llm_service):
        self.llm_service = llm_service

    async def audit_conflict(
        self,
        q: str,
        planner: str,
        purified_think: str,
        refs: List[Dict[str, Any]],
        first_stage_conflict_description: str = "",
        first_stage_conflict_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        进行防锚定的跨厂商二审盲审。
        """
        # 1. 跨厂商模型选择
        primary_model = config.MODEL_POOL_PREMIUM.lower()
        audit_model = config.AUDIT_MODEL
        
        # 强制异构校验
        if "deepseek" in primary_model and "deepseek" in audit_model.lower():
            audit_model = "glm-5.1"
        elif "glm" in primary_model and "glm" in audit_model.lower():
            audit_model = "deepseek-v4-pro"
            
        logger.info(f"🛡️ [FactAuditingGateway] Routing to heterogeneous audit model: '{audit_model}' (Primary: '{primary_model}')")

        # 2. 格式化参考资料与文献事实（排除第一阶段的主观判决，仅提供客观事实）
        refs_text = ""
        if refs:
            for idx, r in enumerate(refs, start=1):
                if isinstance(r, dict):
                    ctx = r.get('context', 'N/A')
                    clean_ctx = ctx.replace("【互联网权威医疗站快讯】:", "").replace("【互联网权威医疗数据通报】:", "").strip()
                    refs_text += f"- fact_{idx:03d}: {clean_ctx}\n"

        # 一审观点只作为 appendix 传入，防止二审模型被一审结论先入为主
        appendix_str = f"冲突描述: {first_stage_conflict_description}\n"
        if first_stage_conflict_details:
            appendix_str += f"冲突细节: {json.dumps(first_stage_conflict_details, ensure_ascii=False)}"
        else:
            appendix_str += "冲突细节: 无"

        prompt = f"""问题: {q}
切面视角: {planner}

原始 RAG 数据库参考内容:
\"\"\"
{refs_text}
\"\"\"

思维链（CoT）中声称的事实/推理:
\"\"\"
{purified_think}
\"\"\"

---
【附录 - 一审检测报告】（注意：此内容仅供参考，请你独立自主做出客观判定，不要被一审结论先入为主）：
{appendix_str}

请根据原始问题、RAG 参考内容与思维链声明事实进行独立比对，给出你的最终裁决。"""

        system_prompt = """您是一位顶级循证医学异构审计专家（Audit LLM）。您的任务是对医学问答中的事实冲突进行独立、客观的审计，并裁定冲突的本质原因。
请秉持完全独立、不受任何人偏见干扰的盲审态度，对给出的原始问题、RAG 数据库参考内容、CoT 声明事实进行对比审查。

### 📋 审计裁决分类准则 (Verdict Taxonomy)：
1. RAG_ERROR: 确证原始 RAG 数据库内容包含医学事实错误、不准确或过时的陈述（例如，图谱或说明书记载错误，或发生字面拼写错词，如将“小便不利”写错为“小便自利”），而思维链（CoT）或用户提出的修正才是真正符合临床常识与循证医学的。
2. COT_HALLUCINATION: 原始 RAG 数据库内容是绝对正确、无误且有扎实文献支持的，而思维链（CoT）凭空编造、捏造了不存在的机制、受体通路、用药禁忌或学术数据。
3. BOTH_ERROR: RAG 数据库内容和思维链（CoT）都存在不同程度的医学事实错误。
4. INSUFFICIENT_EVIDENCE: 根据当前的 RAG 数据库内容及医学常识，证据不够充分，无法支持做出明确的是非判断。
5. AMBIGUOUS: 争议性内容，医学界对此尚无统一的临床共识，存在不同学术流派的合理解释。
6. NEED_HUMAN_REVIEW: 案情极其复杂，或涉及重大安全红线（如物理致命性用药剂量偏差、严重配伍禁忌冲突），系统模型无法做出高置信度裁定，必须物理阻断交由人类医学专家委员会进行人工审阅。

### 📤 输出格式要求：
- 您必须且只能输出符合以下 JSON Schema 的规范 JSON 串，绝对不要包裹在 markdown 标记或 ``` 块中，不要有任何额外文字：
{
  "verdict": "RAG_ERROR" | "COT_HALLUCINATION" | "BOTH_ERROR" | "INSUFFICIENT_EVIDENCE" | "AMBIGUOUS" | "NEED_HUMAN_REVIEW",
  "confidence": 0.0-1.0之间的浮点数,
  "supported_by": "支撑正确事实的文献、指南或常识依据来源",
  "conflicting_span": "冲突的具体错词、句子或片段",
  "corrected_fact": "确证后的正确事实表述（若为 RAG_ERROR，请在此给出具体应写入数据库的修正后文本）",
  "requires_human_review": bool, // 当置信度低或状态为 NEED_HUMAN_REVIEW 或属于高危红线时必须设为 true
  "reason": "严谨细致的审计理由与事实推导过程"
}
"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.llm_service.call_llm(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model_pool=audit_model,
                    stage=f"二审跨厂商盲审 - {planner}"
                )
                
                # 尝试解析 JSON 块
                import core.purification_helper as purifier_module
                json_str = purifier_module.extract_json_block(response)
                result = json.loads(json_str)
                
                required_keys = ["verdict", "confidence", "supported_by", "conflicting_span", "corrected_fact", "requires_human_review", "reason"]
                for k in required_keys:
                    if k not in result:
                        if k == "verdict":
                            result[k] = "NEED_HUMAN_REVIEW"
                        elif k == "confidence":
                            result[k] = 0.5
                        elif k == "requires_human_review":
                            result[k] = True
                        else:
                            result[k] = ""
                            
                # 规范化 verdict 值
                valid_verdicts = {"RAG_ERROR", "COT_HALLUCINATION", "BOTH_ERROR", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS", "NEED_HUMAN_REVIEW"}
                if result["verdict"] not in valid_verdicts:
                    result["verdict"] = "NEED_HUMAN_REVIEW"
                    
                # 强制要求 confidence 为 float
                try:
                    result["confidence"] = float(result["confidence"])
                except Exception:
                    result["confidence"] = 0.5
                    
                return result
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} for heterogeneous audit failed: {e}")
                if attempt == max_retries - 1:
                    break
                await asyncio.sleep(1.0)
                
        # 兜底返回
        return {
            "verdict": "NEED_HUMAN_REVIEW",
            "confidence": 0.0,
            "supported_by": "System Error Fallback",
            "conflicting_span": "",
            "corrected_fact": "",
            "requires_human_review": True,
            "reason": f"Auditing service exception or parsing failed after {max_retries} retries."
        }
