# -*- coding: utf-8 -*-
"""
诊断与测试脚本：专门用于验证 deepseek-v4-pro (及带 bsp- 前缀的实际模型 ID) 的可用性、网络连通性以及 Token 额度。
"""

import os
import sys
import json
import time
import httpx

# 动态加载项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config

def test_models_availability():
    print("=============================================================")
    print("🔍 阶段一：动态获取网关所支持的全部模型列表...")
    print("=============================================================")
    
    headers = {
        "Content-Type": "application/json"
    }
    api_key = config.LLM_API_KEY.strip() if config.LLM_API_KEY else ""
    if not api_key:
        print("❌ 错误：未在环境变量或 .env 中检测到有效的 LLM_API_KEY！")
        return
        
    if api_key.startswith("Bearer "):
        headers["Authorization"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
        
    url_models = config.LLM_API_URL.replace("/chat/completions", "/models")
    
    supported_models = []
    try:
        response = httpx.get(url_models, headers=headers, timeout=10.0)
        print(f"请求状态码: {response.status_code}")
        if response.status_code == 200:
            res_data = response.json()
            if isinstance(res_data, dict) and "data" in res_data:
                print("🎉 获取成功！网关当前账号可用的所有模型 ID 清单如下:")
                for item in res_data["data"]:
                    m_id = item.get("id")
                    print(f"  - {m_id}")
                    supported_models.append(m_id)
            else:
                print("⚠️ 网关返回的 JSON 结构中不包含 'data' 字段:")
                print(json.dumps(res_data, ensure_ascii=False, indent=2))
        else:
            print(f"❌ 获取模型失败: HTTP {response.status_code}")
            print(response.text)
            return
    except Exception as e:
        print(f"❌ 请求网关异常: {e}")
        return

    print("\n=============================================================")
    print("🧪 阶段二：测试 deepseek-v4-pro 与 bsp-deepseek-v4-pro 的连通性...")
    print("=============================================================")
    
    # 自动确定要测试的模型候选
    candidates = []
    # 1. 测试原生名字
    candidates.append("deepseek-v4-pro")
    
    # 2. 如果网关支持带前缀的名字，加入测试
    for m in supported_models:
        if "pro" in m and m != "deepseek-v4-pro":
            candidates.append(m)
            
    # 3. 如果没找到带 pro 的，测试网关返回的第一个模型
    if len(candidates) <= 1 and supported_models:
        candidates.append(supported_models[0])

    for model_name in set(candidates):
        print(f"\n👉 [测试模型]: '{model_name}'")
        data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "你好，请回答两个字：‘收到’。"}
            ],
            "temperature": 0.3
        }
        
        start_time = time.time()
        try:
            res = httpx.post(config.LLM_API_URL, headers=headers, json=data, timeout=30.0)
            elapsed = time.time() - start_time
            print(f"  - 请求状态码: {res.status_code} (耗时: {elapsed:.2f} 秒)")
            
            if res.status_code == 200:
                res_json = res.json()
                if "error" in res_json:
                    err = res_json.get("error", {})
                    print(f"  - ❌ 网关内部业务报错: {err.get('message')} (错误码: {err.get('code')})")
                else:
                    content = res_json["choices"][0]["message"]["content"]
                    usage = res_json.get("usage", {})
                    print(f"  - ✅ 调用成功！")
                    print(f"  - 🤖 回答内容: '{content.strip()}'")
                    print(f"  - 📊 Token 消耗: 输入 {usage.get('prompt_tokens', 0)}，输出 {usage.get('completion_tokens', 0)}，总计 {usage.get('total_tokens', 0)}")
            else:
                print(f"  - ❌ HTTP 调用失败 (HTTP {res.status_code}): {res.text}")
        except httpx.TimeoutException:
            print(f"  - ❌ 调用超时 (超过 30 秒)！表明该模型在大并发或网关响应极慢。")
        except Exception as e:
            print(f"  - ❌ 调用出现异常: {e}")

if __name__ == "__main__":
    test_models_availability()
