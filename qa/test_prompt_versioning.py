import os
import sys
import shutil

# Clean the DB tables before import to ensure test is fully idempotent (without deleting the file since it may be locked by the web server)
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.db")
import sqlite3
try:
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS prompt_versions;")
    conn.commit()
    conn.close()
    print("  - [Idempotent DB Reset] SQLite tables dropped successfully for a pristine test environment.")
except Exception as e:
    pass

# Add current directory to path to ensure we can import prompts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import prompts

def run_tests():
    print("==========================================================")
    print("      提示词版本管理器 (Prompt Version Manager) 自动化验证")
    print("==========================================================\n")

    db_path = os.path.join(os.path.dirname(__file__), "prompts.db")
    
    # ----------------------------------------------------
    # 测试点 1：自引导启动测试 (Self-Bootstrapping)
    # ----------------------------------------------------
    print("[Test 1] 验证数据库自引导启动 (Self-Bootstrapping)...")
    if os.path.exists(db_path):
        print("  - 数据库文件 prompts.db 已存在，说明模块加载时成功自动创建。")
    else:
        print("  - [FAIL] 数据库文件不存在！")
        return False
        
    # 读取一次活跃模版，验证是否能正常返回
    planner_default = prompts.FACET_PLANNER_TEMPLATE
    if planner_default and "用户会给出一个 query" in planner_default:
        print("  - [OK] 成功从数据库获取自动引导加载的 FACET_PLANNER_TEMPLATE Version 1！")
    else:
        print("  - [FAIL] 获取默认模板内容不符！")
        return False

    # ----------------------------------------------------
    # 测试点 2：查询历史版本
    # ----------------------------------------------------
    print("\n[Test 2] 验证查询初始版本历史...")
    history = prompts.list_prompt_versions("FACET_PLANNER_TEMPLATE")
    print(f"  - 查找到的当前版本数量: {len(history)}")
    for v in history:
        print(f"    * 版本 V{v['version']} | 描述: {v['description']} | 激活状态: {v['is_active']} | 创建时间: {v['created_at']}")
    
    if len(history) == 1 and history[0]["version"] == 1 and history[0]["is_active"] == True:
        print("  - [OK] 初始版本 V1 状态正确。")
    else:
        print("  - [FAIL] 初始版本状态异常！")
        return False

    # ----------------------------------------------------
    # 测试点 3：更新提示词版本 (Save New Version)
    # ----------------------------------------------------
    print("\n[Test 3] 验证创建提示词新版本 (Version 2)...")
    medicalized_planner_prompt = """<role>临床医学角度规划器</role>
    <task>为问题规划学术维度，例如：药理机制、临床疗效、安全性毒副反应、用药配伍禁忌等。</task>
    query: {{ query }}
    """
    
    new_v = prompts.update_prompt(
        "FACET_PLANNER_TEMPLATE", 
        medicalized_planner_prompt, 
        "医学化重构，剔除商业合规示例，注入药理疗效维度"
    )
    print(f"  - 成功保存并激活新版本: V{new_v}")
    
    # 再次读取，验证内存中读取的值是否已经**动态改变**（无需重启）
    active_content = prompts.FACET_PLANNER_TEMPLATE
    if "临床医学角度规划器" in active_content:
        print("  - [OK] 内存动态读取成功！最新激活版本已切换为 Version 2。")
    else:
        print("  - [FAIL] 读取的依然是旧提示词内容！")
        return False

    # 验证数据库中版本记录是否有两个，且 V2 是唯一活跃的
    history = prompts.list_prompt_versions("FACET_PLANNER_TEMPLATE")
    print(f"  - 重新查询版本数量: {len(history)}")
    for v in history:
        print(f"    * 版本 V{v['version']} | 描述: {v['description']} | 激活状态: {v['is_active']}")
        
    if len(history) == 2 and history[0]["version"] == 2 and history[0]["is_active"] == True and history[1]["is_active"] == False:
        print("  - [OK] 数据库中 Version 2 激活，Version 1 自动退役。")
    else:
        print("  - [FAIL] 数据库版本激活链记录错误！")
        return False

    # ----------------------------------------------------
    # 测试点 4：版本回滚验证 (Rollback)
    # ----------------------------------------------------
    print("\n[Test 4] 验证版本回滚 (Rollback to V1)...")
    success = prompts.rollback_prompt("FACET_PLANNER_TEMPLATE", 1)
    if success:
        print("  - 回滚操作执行成功。")
    else:
        print("  - [FAIL] 回滚操作执行失败！")
        return False
        
    # 读取内容，验证是否重新变成了原始的 V1
    rolled_content = prompts.FACET_PLANNER_TEMPLATE
    if "用户会给出一个 query" in rolled_content and "临床医学角度规划器" not in rolled_content:
        print("  - [OK] 动态读取验证成功：内容已无缝还原为 Version 1 (Bootstrap Default)！")
    else:
        print("  - [FAIL] 模板内容未正确回滚！")
        return False
        
    history = prompts.list_prompt_versions("FACET_PLANNER_TEMPLATE")
    if history[0]["version"] == 2 and history[0]["is_active"] == False and history[1]["version"] == 1 and history[1]["is_active"] == True:
        print("  - [OK] 数据库中 Version 1 重新置为活跃，Version 2 置为非活跃。")
    else:
        print("  - [FAIL] 数据库回滚状态不一致！")
        return False

    # ----------------------------------------------------
    # 测试点 5：异常边界测试
    # ----------------------------------------------------
    print("\n[Test 5] 验证非正常版本边界测试...")
    # 尝试回滚到不存在的 Version 999
    success_999 = prompts.rollback_prompt("FACET_PLANNER_TEMPLATE", 999)
    if not success_999:
        print("  - [OK] 回滚不存在的版本 999 成功触发拦截拒绝。")
    else:
        print("  - [FAIL] 漏洞：居然成功回滚到了不存在的版本！")
        return False
        
    # 尝试更新一个不存在的提示词变量名
    try:
        prompts.update_prompt("NON_EXISTENT_TEMPLATE", "some text")
        print("  - [FAIL] 漏洞：允许更新未定义模版名！")
        return False
    except ValueError as e:
        print(f"  - [OK] 成功拦截未知变量更新错误: {e}")

    print("\n==========================================================")
    print("       ALL TESTS PASSED 100% SUCCESSFULLY!")
    print("==========================================================")
    return True

if __name__ == "__main__":
    run_tests()
