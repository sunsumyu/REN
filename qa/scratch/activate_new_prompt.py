# -*- coding: utf-8 -*-
"""
一键把 prompts.py 中的 _BOOTSTRAP_QUESTION_CREATOR_TEMPLATE 新版本
强制写入 PromptManager 数据库并激活。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import prompts

new_content = prompts._BOOTSTRAP_QUESTION_CREATOR_TEMPLATE

version_id = prompts.update_prompt(
    "QUESTION_CREATOR_TEMPLATE",
    new_content,
    description="架构升级：允许通用医学常识补全临床场景，禁止捏造药物具体事实；新增乌鸡增乳胶囊示例"
)
print(f"SUCCESS: QUESTION_CREATOR_TEMPLATE 已更新到版本 {version_id}")

# 验证读取
live = prompts.QUESTION_CREATOR_TEMPLATE
if "允许通用医学常识补全临床场景" in live:
    print("✅ 验证通过：数据库中已激活新版本提示词")
else:
    print("❌ 验证失败：数据库版本未更新，请检查")
