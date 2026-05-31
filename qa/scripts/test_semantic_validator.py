# -*- coding: utf-8 -*-
"""
小模型语义安全校验网关单元测试脚本 (Test Semantic Validator)。
传入典型合法视角与非法占位符变体，测试小模型（deepseek-v4-flash）的语义识别准确率。
"""

import sys
import asyncio
from pathlib import Path

# 将相关路径加入 sys.path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(parent_dir))

from api_client import APIClient
from scripts.medicalqa_purifier import verify_facet_by_small_model

# 针对 Windows 环境配置标准输出
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

async def test_semantic_validator():
    print("[TEST] 正在初始化 API 客户端与小模型语义网关...")
    client = APIClient()
    
    # 定义测试用例 (输入, 期望结果)
    test_cases = [
        # 非法占位符变体 (应该判定为 False / INVALID)
        ("请提供具体的医疗问题以便规划视角", False),
        ("由于信息不足，无法为该问题规划视角", False),
        ("请补充临床病历和症状", False),
        ("请输入需要分析的具体药物成分", False),
        ("无法规划，问题过于简单", False),
        
        # 合法临床/药理视角 (应该判定为 True / VALID)
        ("药代动力学", True),
        ("分子机制", True),
        ("特殊人群用药安全", True),
        ("不良反应与配伍禁忌", True),
        ("古籍收采", True),
        ("浆细胞", True)
    ]
    
    passed_count = 0
    print("-----------------------------------------")
    print(f"🚀 开始执行 {len(test_cases)} 个视角用例测试...")
    print("-----------------------------------------")
    
    for idx, (facet, expected) in enumerate(test_cases):
        try:
            is_valid = await verify_facet_by_small_model(client, facet)
            status = "PASS" if is_valid == expected else "FAIL"
            if status == "PASS":
                passed_count += 1
            print(f"[{status}] 用例 {idx+1}: 视角 = '{facet}' | 网关输出 = {'VALID' if is_valid else 'INVALID'} | 期望 = {'VALID' if expected else 'INVALID'}")
        except Exception as e:
            print(f"[FAIL] 用例 {idx+1}: 发生异常: {e}")
            
    print("-----------------------------------------")
    success_rate = (passed_count / len(test_cases)) * 100
    print(f"📊 测试统计报告: 成功数 = {passed_count}/{len(test_cases)} | 准确率 = {success_rate:.2f}%")
    print("-----------------------------------------")
    
    await client.close()
    
    if passed_count == len(test_cases):
        print("[SUCCESS] 语义网关全部测试用例通过！泛化判定能力符合预期。")
    else:
        print("[WARN] 仍有部分边缘用例未能完全对齐，可进一步优化系统提示词。")

if __name__ == "__main__":
    asyncio.run(test_semantic_validator())
