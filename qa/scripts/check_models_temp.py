import httpx
import os
import sys

# Load env from workspace dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import config

headers = {
    "Authorization": f"Bearer {config.LLM_API_KEY.strip()}" if config.LLM_API_KEY else "Bearer dummy"
}
url = "https://volley.yzint.cn/api/v1/models"
print(f"Fetching from {url}...")
try:
    res = httpx.get(url, headers=headers, timeout=10.0)
    print("Status Code:", res.status_code)
    if res.status_code == 200:
        models_data = res.json()
        print("Success! Supported models list:")
        if isinstance(models_data, dict) and "data" in models_data:
            for item in models_data["data"]:
                print(f"- {item.get('id')}")
        else:
            print(models_data)
    else:
        print("Failed to fetch models:", res.text)
except Exception as e:
    print("Error:", e)
