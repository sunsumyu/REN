# -*- coding: utf-8 -*-
import os
import sys
import sqlite3
import shutil
import asyncio

# 添加当前和父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from services.fact_correction_service import DBHotpatchManager, FactCorrectionProposal, rollback_patch

TEST_DB_PATH = os.path.join(current_dir, "test_temp.db")

def setup_test_db():
    # 确保清理旧的测试数据库
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
        
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    
    # 1. 创建物理表 local_rag_index
    cursor.execute("""
        CREATE TABLE local_rag_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT,
            source TEXT,
            context TEXT,
            category TEXT
        );
    """)
    
    # 2. 创建 FTS5 虚拟表 local_rag_fts_index (隐式包含 rowid)
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE local_rag_fts_index USING fts5(
                source,
                context,
                entity_name,
                category UNINDEXED,
                tokenize="unicode61"
            );
        """)
        fts_supported = True
    except sqlite3.OperationalError:
        print("  - [Warning] FTS5 not supported in host SQLite. Virtual table setup skipped.")
        fts_supported = False
        
    # 3. 插入初始错误记录 (小便不利)
    erroneous_context = "医疗实体【甘草干姜茯苓白术汤】（类型: 中药）：其人身体重，腰中冷，如坐水中，腹痛，小便不利，饮食如常，病属下焦。"
    cursor.execute("""
        INSERT INTO local_rag_index (entity_name, source, context, category)
        VALUES ('甘草干姜茯苓白术汤', '《金匮要略》', ?, '方剂功效');
    """, (erroneous_context,))
    
    if fts_supported:
        cursor.execute("""
            INSERT INTO local_rag_fts_index (entity_name, source, context, category)
            VALUES ('甘草干姜茯苓白术汤', '《金匮要略》', ?, '方剂功效');
        """, (erroneous_context,))
        
    conn.commit()
    conn.close()
    return fts_supported

def teardown_test_db():
    # 为了方便用户在测试完成后打开并检查日志与数据库内容，这里不再物理删除它们。
    print("  - [Info] Preserving test database and audit logs for user inspection.")
    pass

async def main():
    print("==========================================================")
    print("      知识库物理事实纠错与热修复回滚 自动化单元测试")
    print("==========================================================\n")
    
    fts_supported = setup_test_db()
    manager = DBHotpatchManager(TEST_DB_PATH)
    
    try:
        # ----------------------------------------------------
        # 测试 1: 安全白名单拦截拦截 (Security Gate Whitelist)
        # ----------------------------------------------------
        print("[Test 1] 验证越权表名/字段名 SQL 白名单过滤...")
        
        # 尝试恶意修改不存在的表 (越权操作)
        prop_bad_table = FactCorrectionProposal(
            proposal_id="prop_bad_table",
            db_type="local_rag.db",
            table_name="datasets", # 越权表名
            target_id="1",
            field_name="context",
            corrected_value="malicious text",
            clinical_evidence="test"
        )
        res1 = manager.execute_patch(prop_bad_table)
        assert res1 is False, "FAILED: 应该拦截非 local_rag_index 的表名更新！"
        
        # 尝试恶意修改不可更新字段 (防 SQL 注入/越权)
        prop_bad_field = FactCorrectionProposal(
            proposal_id="prop_bad_field",
            db_type="local_rag.db",
            table_name="local_rag_index",
            target_id="1",
            field_name="category", # 不在白名单中的字段
            corrected_value="malicious text",
            clinical_evidence="test"
        )
        res2 = manager.execute_patch(prop_bad_field)
        assert res2 is False, "FAILED: 应该拦截非 context 字段的更新！"
        print("  - [OK] 安全白名单防御成功，恶意越权修改提案已全部物理阻断。")
        
        # ----------------------------------------------------
        # 测试 2: 目标数据原值 Context-Bound 比对校验
        # ----------------------------------------------------
        print("\n[Test 2] 验证前置值 Context-Bound 对齐校验...")
        
        # 错配的原错误值：我们希望是“大便不利”，但实际数据库中是“小便不利”
        prop_mismatch = FactCorrectionProposal(
            proposal_id="prop_mismatch",
            db_type="local_rag.db",
            table_name="local_rag_index",
            target_id="1",
            field_name="context",
            corrected_value="小便自利",
            clinical_evidence="test",
            original_value="大便不利" # 与实际不匹配的错误值
        )
        res3 = manager.execute_patch(prop_mismatch)
        assert res3 is False, "FAILED: 原值对齐失败的提案应当被拦截拒绝！"
        print("  - [OK] 前置校验成功，检测出原值不一致，避免了错位误伤修改。")
        
        # ----------------------------------------------------
        # 测试 3: 正常纠错物理 Patch 及两镜像（Pre/Post Image）记录
        # ----------------------------------------------------
        print("\n[Test 3] 验证正常热修复执行及双镜像历史记录...")
        
        correct_context = "医疗实体【甘草干姜茯苓白术汤】（类型: 中药）：其人身体重，腰中冷，如坐水中，腹痛，小便自利，饮食如常，病属下焦。"
        
        prop_correct = FactCorrectionProposal(
            proposal_id="prop_correct_001",
            db_type="local_rag.db",
            table_name="local_rag_index",
            target_id="1",
            field_name="context",
            corrected_value=correct_context,
            clinical_evidence="《金匮要略·脏腑经络先后病脉证》原文记载：'肾著之病...反不渴，小便自利，饮食如常'，此处误作'不利'，属严重医理错误，现予纠正以保真典型辨证体征。",
            original_value="小便不利" # 精确指明要更正的原文片段
        )
        res4 = manager.execute_patch(prop_correct)
        assert res4 is True, "FAILED: 正常的纠错提案执行失败！"
        
        # A. 校验物理表是否已经被修改为 corrected_value
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT context FROM local_rag_index WHERE id = 1")
        row = cursor.fetchone()
        assert "小便自利" in row[0], "FAILED: 物理表 local_rag_index 事实数据未更正为 小便自利！"
        
        # B. 校验 FTS5 虚拟表是否同步修改
        if fts_supported:
            cursor.execute("SELECT context FROM local_rag_fts_index WHERE rowid = 1")
            fts_row = cursor.fetchone()
            assert "小便自利" in fts_row[0], "FAILED: FTS5 虚拟表 context 未同步更新为 小便自利！"
            
        # C. 校验版本控制表 kb_correction_history 是否记录了 before_value (Pre-Image) 和 after_value (Post-Image)
        cursor.execute("SELECT before_value, after_value, status FROM kb_correction_history WHERE proposal_id = 'prop_correct_001'")
        hist_row = cursor.fetchone()
        assert hist_row is not None, "FAILED: kb_correction_history 中未找到对应的历史记录！"
        assert "小便不利" in hist_row[0], f"FAILED: before_value 记录错误！实际为: {hist_row[0]}"
        assert "小便自利" in hist_row[1], f"FAILED: after_value 记录错误！实际为: {hist_row[1]}"
        assert hist_row[2] == 'APPLIED', f"FAILED: 初始状态应为 APPLIED！实际为: {hist_row[2]}"
        
        conn.close()
        print("  - [OK] 热修复执行成功。物理表、FTS5表同步更新。修改前/后值均被完整备份存入 kb_correction_history。")
        
        # ----------------------------------------------------
        # 测试 4: 幂等性自检
        # ----------------------------------------------------
        print("\n[Test 4] 验证补丁的幂等性执行...")
        res_idempotent = manager.execute_patch(prop_correct)
        assert res_idempotent is True, "FAILED: 幂等执行返回了失败！"
        print("  - [OK] 幂等检测通过，对于已修改过的数据自动绕过物理写入并返回成功。")
        
        # ----------------------------------------------------
        # 测试 5: 一键原子化回滚 (Rollback with Pre-Image)
        # ----------------------------------------------------
        print("\n[Test 5] 验证一键事务回滚还原 (Rollback via Pre-Image)...")
        res_rollback = await rollback_patch(TEST_DB_PATH, "prop_correct_001")
        assert res_rollback is True, "FAILED: 执行回滚操作失败！"
        
        # A. 校验物理表是否还原为 原始错误值 (小便不利)
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT context FROM local_rag_index WHERE id = 1")
        row_restored = cursor.fetchone()
        assert "小便不利" in row_restored[0], "FAILED: 物理表未能恢复为修改前原始值！"
        
        # B. 校验 FTS5 虚拟表是否还原
        if fts_supported:
            cursor.execute("SELECT context FROM local_rag_fts_index WHERE rowid = 1")
            fts_restored = cursor.fetchone()
            assert "小便不利" in fts_restored[0], "FAILED: FTS5 虚拟表未能还原！"
            
        # C. 校验版本控制历史中状态是否已置为 ROLLEDBACK
        cursor.execute("SELECT status FROM kb_correction_history WHERE proposal_id = 'prop_correct_001'")
        status_row = cursor.fetchone()
        assert status_row[0] == 'ROLLEDBACK', f"FAILED: 历史表中的 status 应更改为 ROLLEDBACK，实际为: {status_row[0]}"
        
        conn.close()
        print("  - [OK] 一键原子化回滚成功！物理数据与检索索引完美无损还原，回滚台账状态正确。")
        
        print("\n==========================================================")
        print("       所有测试单元 100% 验证成功！")
        print("==========================================================")
        
    except AssertionError as ae:
        print(f"\n❌ [Assertion Error] {ae}")
        teardown_test_db()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 [Unexpected Crash] {e}")
        teardown_test_db()
        sys.exit(1)
        
    teardown_test_db()

if __name__ == "__main__":
    asyncio.run(main())
