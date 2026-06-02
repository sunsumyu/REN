import httpx
import os
import sys
import socket

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)
import config

print("="*60)
print("🔍 医疗数据集生成系统 - 网络与 VPN 诊断工具")
print("="*60)

# 1. 检查 API Key
api_key = config.LLM_API_KEY.strip() if config.LLM_API_KEY else ""
if not api_key:
    print("❌ 警告: 未检测到 LLM_API_KEY 环境变量！请确保在 .env 文件中填写。")
else:
    print(f"✅ 检测到 API Key: {api_key[:10]}...{api_key[-10:] if len(api_key)>20 else ''}")

# 2. DNS 解析测试
domains = [
    "ai.yzint.cn",
    "volley.inner.yzint.cn",
    "volley.yzint.cn"
]

print("\n📡 步骤 1: DNS 域名解析测试 (检查是否在 VPN 内网)...")
for domain in domains:
    try:
        ip = socket.gethostbyname(domain)
        print(f"  - {domain:25} => 解析成功! IP: {ip}")
    except socket.gaierror:
        print(f"  - {domain:25} => ❌ 解析失败! (提示: 如果 volley.inner.yzint.cn 解析失败，说明您的公司 VPN 已断开，请重新连接 VPN！)")

# 3. HTTP 接口连通性测试
headers = {
    "Authorization": f"Bearer {api_key}" if api_key else "Bearer dummy"
}

urls_to_test = [
    ("默认内网地址 (需要VPN)", "https://volley.inner.yzint.cn/v1/models"),
    ("潜在外网备份1", "https://volley.yzint.cn/v1/models"),
    ("潜在外网备份2", "https://ai.yzint.cn/v1/models"),
    ("知识图谱API (测试外网连通性)", f"{config.GRAPH_API_URL}?count=1")
]

print("\n🚀 步骤 2: HTTP 接口连通性与权限测试...")
for label, url in urls_to_test:
    print(f"\n👉 测试 [{label}]: {url}")
    try:
        # 3秒超时防止卡死
        res = httpx.get(url, headers=headers, timeout=3.0)
        print(f"  - 响应状态码: {res.status_code}")
        if res.status_code == 200:
            print("  - 🎉 连接成功! 该域名可正常使用。")
            if "models" in url:
                try:
                    data = res.json()
                    models = [item.get('id') for item in data.get("data", []) if item.get('id')]
                    print(f"  - 🎉 获取成功！共检测到 {len(models)} 个支持的模型:")
                    for m in models:
                        print(f"    * {m}")
                except Exception as e:
                    print(f"  - ❌ 解析模型列表与JSON失败: {e}")
        else:
            print(f"  - 连接失败: 状态码 {res.status_code}, 返回: {res.text[:100]}")
    except httpx.ConnectTimeout:
        print("  - ❌ 连接超时! (提示: 这通常是由于未连接 VPN 导致的网络阻断)")
    except httpx.ConnectError as e:
        print(f"  - ❌ 连接错误! 错误信息: {e}")
    except Exception as e:
        print(f"  - ❌ 其他异常: {e}")

print("\n"+"="*60)
print("💡 诊断建议:")
print("1. 如果 volley.inner.yzint.cn 解析失败或连接超时，请【检查并连接您的公司 VPN】。")
print("2. 内网域名 volley.inner.yzint.cn 只有在 VPN 正常连接时才能被解析和访问。")
print("3. 如果您重新连上了 VPN，请重新运行数据生成脚本。")
print("="*60)
