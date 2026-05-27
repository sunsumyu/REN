import httpx
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
import config

print("="*60)
print("🔍 批处理测试 - 寻找可用的大语言模型 POST 接口")
print("="*60)

api_key = config.LLM_API_KEY.strip() if config.LLM_API_KEY else ""
headers = {
    "Authorization": f"Bearer {api_key}" if api_key else "Bearer dummy",
    "Content-Type": "application/json"
}

# 极简的测试数据负载
data_payload = {
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 5
}

# 候选测试列表
candidates = [
    ("1. 默认内网 (HTTPS, 启用SSL验证)", "https://volley.inner.yzint.cn/v1/chat/completions", True, None),
    ("2. 默认内网 (HTTPS, 禁用SSL验证 verify=False)", "https://volley.inner.yzint.cn/v1/chat/completions", False, None),
    ("3. 默认内网 (HTTP协议, 绕过SSL)", "http://volley.inner.yzint.cn/v1/chat/completions", True, None),
    ("4. 潜在外网备份1 (HTTPS)", "https://volley.yzint.cn/v1/chat/completions", True, None),
    ("5. 潜在外网备份1 (加上 /api 前缀)", "https://volley.yzint.cn/api/v1/chat/completions", True, None),
    ("6. 潜在外网备份2 (HTTPS)", "https://ai.yzint.cn/v1/chat/completions", True, None),
    ("7. 潜在外网备份2 (加上 /api 前缀)", "https://ai.yzint.cn/api/v1/chat/completions", True, None)
]

for label, url, verify_ssl, proxy in candidates:
    print(f"\n👉 正在测试 [{label}]:")
    print(f"   URL: {url}")
    
    # 强制不使用环境变量里的代理，以防系统代理干扰
    client_kwargs = {
        "headers": headers,
        "timeout": 5.0,
        "verify": verify_ssl
    }
    
    try:
        with httpx.Client(**client_kwargs) as client:
            response = client.post(url, json=data_payload)
            print(f"   - 响应状态码: {response.status_code}")
            if response.status_code == 200:
                print("   - 🎉 成功！！！此接口能够正常完成大模型 POST 问答！")
                print(f"   - 返回预览: {response.text[:120]}")
            else:
                print(f"   - ❌ 失败 (状态码 {response.status_code}): {response.text[:200].strip()}")
    except httpx.ConnectTimeout:
        print("   - ❌ 连接超时 (Timeout)")
    except httpx.ConnectError as e:
        print(f"   - ❌ 连接错误 (ConnectError): {e}")
    except Exception as e:
        print(f"   - ❌ 其他异常: {e}")

print("\n" + "="*60)
print("💡 结论建议:")
print("根据上面测试输出为 200 的项，修改 config.py 中的 LLM_API_URL 即可！")
print("="*60)
