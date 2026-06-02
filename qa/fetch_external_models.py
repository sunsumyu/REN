import requests
import json
import os

url = "https://volley.yzint.cn/v1/models"
headers = {
    "Authorization": "Bearer sk-47eb1dc812fee85e58f0f5227d3930c2de019f2b"
}

try:
    print(f"Fetching models from {url}...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    models = []
    if "data" in data:
        for model in data["data"]:
            models.append(model.get("id"))
    
    output_file = os.path.join(os.path.dirname(__file__), "external_models.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully saved {len(models)} models to {output_file}")
    
except Exception as e:
    print(f"Error fetching models: {e}")
