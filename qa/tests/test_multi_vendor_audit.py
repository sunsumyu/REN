# -*- coding: utf-8 -*-
import os
import sys
import sqlite3
import json
import asyncio
import time
import random
from typing import Dict, Any, List, Tuple

# 添加根路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from core.purification_engine import PurificationEngine
from services.fact_correction_service import DBHotpatchManager, FactCorrectionProposal
from strategies.quality_gate.llm_judge import LLMJudgeStrategy
from retrieval.local_rag import LocalRAGService

TEST_DB_PATH = os.path.join(current_dir, "test_audit_temp.db")

class MockLLMService:
    def __init__(self):
        self.call_count = 0

    async def init_supported_models(self, force=False):
        pass

    async def call_llm(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "", max_tokens: int = None) -> str:
        self.call_count += 1
        
        # 1. 模拟一审裁判打分评估
        if "三维质检" in stage or "JUDGE_SYSTEM_PROMPT" in system_prompt or "三维评分" in prompt:
            if "trigger_conflict_approved" in prompt:
                return json.dumps({
                    "semantic_purity_score": 90,
                    "medical_rigor_score": 90,
                    "logical_depth_score": 90,
                    "factual_errors": [],
                    "conflict_detected": True,
                    "conflict_description": "原始事实为小便不利，但CoT推导成了小便自利",
                    "conflict_details": {
                        "field_name": "context",
                        "original_value": "小便不利",
                        "corrected_value": "小便自利",
                        "evidence": "甘草干姜茯苓白术汤"
                    },
                    "reason": "first-stage judge flagged conflict (approved case)",
                    "improvement_suggestions": "none"
                })
            elif "trigger_conflict_blocked" in prompt:
                return json.dumps({
                    "semantic_purity_score": 90,
                    "medical_rigor_score": 90,
                    "logical_depth_score": 90,
                    "factual_errors": [],
                    "conflict_detected": True,
                    "conflict_description": "原始事实为小便不利，但CoT推导成了大便不利",
                    "conflict_details": {
                        "field_name": "context",
                        "original_value": "小便不利",
                        "corrected_value": "大便不利",
                        "evidence": "甘草干姜茯苓白术汤"
                    },
                    "reason": "first-stage judge flagged conflict (blocked case)",
                    "improvement_suggestions": "none"
                })
            else:
                # 修复后重试时的裁判，返回通过分数，且无冲突
                return json.dumps({
                    "semantic_purity_score": 95,
                    "medical_rigor_score": 95,
                    "logical_depth_score": 95,
                    "factual_errors": [],
                    "conflict_detected": False,
                    "conflict_description": "",
                    "conflict_details": None,
                    "reason": "passed after retry",
                    "improvement_suggestions": "none"
                })

        # 2. 模拟二审跨厂商盲审
        elif "二审跨厂商盲审" in stage:
            if "大便不利" in prompt:
                # 触发阻断：二审裁决为 RAG_ERROR，置信度高，但由于大便不利不在黄金白名单中，写库时被安全阻断
                return json.dumps({
                    "verdict": "RAG_ERROR",
                    "confidence": 0.99,
                    "supported_by": "《伤寒杂病论》",
                    "conflicting_span": "小便不利",
                    "corrected_fact": "大便不利",
                    "requires_human_review": False,
                    "reason": "Test audit suggests RAG has error (blocked path)"
                })
            else:
                # 触发自动 Patch：二审裁决为 RAG_ERROR，置信度高，且小便自利在白名单中
                return json.dumps({
                    "verdict": "RAG_ERROR",
                    "confidence": 0.99,
                    "supported_by": "《金匮要略》",
                    "conflicting_span": "小便不利",
                    "corrected_fact": "小便自利",
                    "requires_human_review": False,
                    "reason": "Test audit confirms RAG has error (approved path)"
                })

        # 3. 模拟思维链重写提纯
        elif "思维链重写提纯" in stage:
            if "trigger_conflict_approved" in prompt:
                return "甘草干姜茯苓白术汤具有健脾祛湿之功，临床表现为小便自利。"
            elif "trigger_conflict_blocked" in prompt:
                return "甘草干姜茯苓白术汤具有健脾祛湿之功，临床表现为大便不利。"
            else:
                return "甘草干姜茯苓白术汤具有健脾祛湿之功，临床表现为小便自利。"
        
        return "mocked generic response"


class MockHealingService:
    async def heal_conversational_noise(self, text, line_num=None):
        return text
    async def verify_and_repair_academic_entities(self, text, q, planner, line_num=None):
        return text


def setup_test_db():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
        
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE local_rag_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT,
            source TEXT,
            context TEXT,
            category TEXT
        );
    """)
    cursor.execute("""
        CREATE VIRTUAL TABLE local_rag_fts_index USING fts5(
            source,
            context,
            entity_name,
            category UNINDEXED,
            tokenize="unicode61"
        );
    """)
    
    erroneous_context = "甘草干姜茯苓白术汤（又名肾著汤），主治肾著之病，其人身体重，腰中冷，腹痛，小便不利，饮食如常。"
    cursor.execute("""
        INSERT INTO local_rag_index (entity_name, source, context, category)
        VALUES ('甘草干姜茯苓白术汤', '《金匮要略》', ?, '方剂主治');
    """, (erroneous_context,))
    cursor.execute("""
        INSERT INTO local_rag_fts_index (rowid, entity_name, source, context, category)
        VALUES (1, '甘草干姜茯苓白术汤', '《金匮要略》', ?, '方剂主治');
    """, (erroneous_context,))
    
    conn.commit()
    conn.close()


async def run_tests():
    print("==========================================================")
    print("      多厂商异构共识与物理安全更新门禁 自动化集成测试")
    print("==========================================================\n")
    
    # ----------------------------------------------------
    # 测试用例 1: 黄金白名单匹配成功 -> 物理写库 -> 提纯自愈重试通过
    # ----------------------------------------------------
    print("[Test 1] 运行符合黄金白名单的自动 Patch 路径...")
    setup_test_db()
    
    llm_service = MockLLMService()
    healing_service = MockHealingService()
    evaluator = LLMJudgeStrategy(llm_service)
    engine = PurificationEngine(llm_service, healing_service, evaluator)
    
    # 手动设置环境变量指向我们的测试 DB，让 clear_all_caches 重新加载时能够读取测试 DB
    os.environ["LOCAL_RAG_SQLITE_DB_PATH"] = TEST_DB_PATH
    
    # 注册一个测试用的 LocalRAGService 实例以验证全局缓存清除
    rag_service = LocalRAGService(workspace_dir="")
    
    q = "甘草干姜茯苓白术汤的小便情况是怎样的？"
    planner = "药理机制"
    raw_think = "首先，甘草干姜茯苓白术汤治肾著，小便不利。其次，小便自利。"
    purified_answer = "小便自利。"
    
    refs = [{
        "source": "《金匮要略》",
        "context": "甘草干姜茯苓白术汤（又名肾著汤），主治肾著之病，其人身体重，腰中冷，腹痛，小便不利，饮食如常。"
    }]
    
    # 模拟输入带有 trigger_conflict_approved 标志
    purified_think, scores = await engine.purify_single_think(
        q="trigger_conflict_approved: " + q,
        planner=planner,
        raw_think=raw_think,
        purified_answer=purified_answer,
        line_num=1,
        refs=refs,
        simplify=False
    )
    
    # 验证数据库是否已被修改
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT context FROM local_rag_index WHERE id = 1")
    db_ctx = cursor.fetchone()[0]
    print(f"   - 数据库最新 context: {db_ctx}")
    assert "小便自利" in db_ctx, "FAILED: 物理表 local_rag_index 事实数据未更正为 小便自利！"
    assert "小便不利" not in db_ctx, "FAILED: 原始错误数据 小便不利 未被替换！"
    
    # 验证 FTS5 虚拟表是否同步修改
    cursor.execute("SELECT context FROM local_rag_fts_index WHERE rowid = 1")
    fts_ctx = cursor.fetchone()[0]
    assert "小便自利" in fts_ctx, "FAILED: FTS5 虚拟表数据未同步！"
    
    # 验证版本历史记录状态为 APPLIED
    cursor.execute("SELECT status, before_value, after_value FROM kb_correction_history WHERE target_id = '1'")
    hist = cursor.fetchone()
    assert hist is not None, "FAILED: kb_correction_history 表未写入历史！"
    assert hist[0] == "APPLIED", f"FAILED: 状态应为 APPLIED，实际为: {hist[0]}"
    assert "小便不利" in hist[1], "FAILED: before_value Pre-Image 记录有误！"
    assert "小便自利" in hist[2], "FAILED: after_value Post-Image 记录有误！"
    
    conn.close()
    
    # 验证提纯通过
    assert scores.get("is_passed") is True, "FAILED: 自愈重试应当通过提纯质检！"
    print("   - [OK] 黄金白名单 Patch 写入成功，物理表/FTS5同步完成，提纯顺利通过。")
    
    # ----------------------------------------------------
    # 测试用例 2: 非黄金白名单拦截 -> 标记为 PENDING_APPROVAL -> 拦截写库
    # ----------------------------------------------------
    print("\n[Test 2] 运行非黄金白名单的安全拦截路径...")
    setup_test_db()
    
    # 重新注册一个新的 LocalRAGService 实例
    rag_service.close()
    rag_service = LocalRAGService(workspace_dir="")
    
    # 模拟输入带有 trigger_conflict_blocked 标志 (会让 model 生成大便不利)
    purified_think, scores = await engine.purify_single_think(
        q="trigger_conflict_blocked: " + q,
        planner=planner,
        raw_think=raw_think,
        purified_answer=purified_answer,
        line_num=2,
        refs=refs,
        simplify=False
    )
    
    # 验证数据库是否未被修改
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT context FROM local_rag_index WHERE id = 1")
    db_ctx_blocked = cursor.fetchone()[0]
    print(f"   - 数据库最新 context (阻断后): {db_ctx_blocked}")
    assert "小便不利" in db_ctx_blocked, "FAILED: 物理表被意外修改，防线崩溃！"
    assert "大便不利" not in db_ctx_blocked, "FAILED: 物理表被写入了非白名单事实！"
    
    # 验证历史记录状态为 PENDING_APPROVAL
    cursor.execute("SELECT status FROM kb_correction_history WHERE status = 'PENDING_APPROVAL'")
    hist_blocked = cursor.fetchone()
    assert hist_blocked is not None, "FAILED: kb_correction_history 表中没有记录 PENDING_APPROVAL 历史！"
    
    conn.close()
    rag_service.close()
    
    # 验证提纯拦截未通过
    assert scores.get("is_passed") is False, "FAILED: 未通过安全门禁的行应当在提纯中被置为 is_passed=False"
    assert scores.get("requires_human_review") is True, "FAILED: 应该标记为需要人工审核！"
    print("   - [OK] 非白名单修改请求已被成功拦截，主表未受任何污染，状态正确记录为 PENDING_APPROVAL。")
    
    # 清理测试 DB
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
        
    print("\n==========================================================")
    print("       多厂商共识与会审安全拦截测试 100% 成功！")
    print("==========================================================")

if __name__ == "__main__":
    asyncio.run(run_tests())
