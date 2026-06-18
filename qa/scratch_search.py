import httpx
import json

def search_entity():
    keyword = "甘草干姜茯苓白术汤"
    base_url = "https://ai.yzint.cn/api/knowledge/v1/graph/entity"
    
    # 我们并不知道远端搜索接口的准确路径，所以暴力穷举几种业界常见的接口风格
    get_urls = [
        f"{base_url}/search?name={keyword}",
        f"{base_url}/search?keyword={keyword}",
        f"{base_url}/query?name={keyword}",
        f"{base_url}?name={keyword}",
    ]
    
    post_endpoints = [
        f"{base_url}/search",
        f"{base_url}/query",
        f"{base_url}/list",
        f"{base_url}/find"
    ]
    
    post_payloads = [
        {"name": keyword},
        {"keyword": keyword},
        {"entityName": keyword}
    ]
    
    print(f"[*] 正在尝试从图数据库按名称搜索实体: 【{keyword}】")
    
    with httpx.Client(verify=False) as client:
        # 1. 尝试 GET 请求
        print("\n--- 尝试 GET 搜索接口 ---")
        for url in get_urls:
            print(f"[->] GET {url}")
            try:
                resp = client.get(url, timeout=3.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and (data.get("success") or "data" in data) and data.get("code") != 404:
                        print("\n✅ 疑似成功获取搜索结果:")
                        print(json.dumps(data, ensure_ascii=False, indent=2)[:1000])
                        return
            except Exception:
                pass
                
        # 2. 尝试 POST 请求
        print("\n--- 尝试 POST 搜索接口 ---")
        for url in post_endpoints:
            for payload in post_payloads:
                print(f"[->] POST {url} | Payload: {payload}")
                try:
                    resp = client.post(url, json=payload, timeout=3.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, dict) and data.get("code") == 0 and "data" in data:
                            print("\n✅ 成功获取 POST 搜索结果:")
                            print(json.dumps(data, ensure_ascii=False, indent=2))
                            return
                except Exception:
                    pass
                    
    print("\n[-] 穷举失败。未能找到支持按名称搜索的开放图数据库接口。")

if __name__ == "__main__":
    search_entity()
