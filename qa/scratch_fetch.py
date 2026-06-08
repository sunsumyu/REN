import httpx
import json

def fetch_entity():
    entity_id = "60704540018180096"
    url = "https://ai.yzint.cn/api/knowledge/v1/graph/entity/getIds"
    payload = {"entityIds": [entity_id]}
    
    print(f"[*] 正在尝试从图数据库 POST 拉取实体: {entity_id}")
    print(f"[->] 请求 URL: {url}")
    print(f"[->] Payload: {json.dumps(payload)}")
    
    with httpx.Client(verify=False) as client:
        try:
            resp = client.post(url, json=payload, timeout=5.0)
            print(f"[<-] 状态码: {resp.status_code}")
            
            if resp.status_code == 200:
                data = resp.json()
                print("\n✅ 成功获取实体数据 (截断输出前1000字符):")
                formatted_json = json.dumps(data, ensure_ascii=False, indent=2)
                print(formatted_json[:1000])
                
                # 检查是否包含“小便不利”
                if "小便不利" in formatted_json:
                    print("\n⚠️ 发现关键字：【小便不利】确实存在于图数据库的返回值中！")
                else:
                    print("\n🟢 未发现关键字：图数据库返回值中不包含【小便不利】。")
            else:
                print(f"[-] 接口报错，返回内容: {resp.text}")
        except Exception as e:
            print(f"[!] 发生错误: {e}")

if __name__ == "__main__":
    fetch_entity()
