# -*- coding: utf-8 -*-
import os
import sys
import json
import httpx

# Load config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config

def inspect():
    headers = {
        "Content-Type": "application/json"
    }
    api_key = config.LLM_API_KEY.strip() if config.LLM_API_KEY else ""
    if not api_key:
        print("❌ LLM_API_KEY is empty!")
        return
        
    if api_key.startswith("Bearer "):
        headers["Authorization"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
        
    url = config.LLM_API_URL.replace("/chat/completions", "/models")
    print(f"Querying models from: {url}")
    
    try:
        response = httpx.get(url, headers=headers, timeout=10.0)
        output = {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        output = {
            "error": str(e)
        }
        
    out_path = os.path.join(current_dir, "models_output.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        
    print(f"🎉 Inspection completed! Results saved to: {out_path}")

if __name__ == "__main__":
    inspect()
