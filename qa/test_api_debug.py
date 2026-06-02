import httpx
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load env from workspace
workspace_dir = Path("D:/REN/qa")
load_dotenv(workspace_dir / ".env")

api_key = os.getenv("LLM_API_KEY", "").strip()
api_url = os.getenv("LLM_API_URL", "https://volley.yzint.cn/api/v1/chat/completions")

headers = {
    "Content-Type": "application/json"
}
if api_key:
    if api_key.startswith("Bearer "):
        headers["Authorization"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
else:
    headers["Authorization"] = "Bearer dummy"

async def test():
    # 1. Test /models endpoint
    models_url = api_url.replace("/chat/completions", "/models")
    print(f"Testing GET {models_url}")
    print(f"Headers: {headers}")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(models_url, headers=headers, timeout=10.0)
            print(f"Status: {resp.status_code}")
            print(f"Response headers: {dict(resp.headers)}")
            print(f"Response text (first 500 chars):")
            print(repr(resp.text[:500]))
        except Exception as e:
            print(f"Error: {e}")

        # 2. Test /chat/completions endpoint
        print("\nTesting POST /chat/completions")
        data = {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2
        }
        try:
            resp = await client.post(api_url, headers=headers, json=data, timeout=10.0)
            print(f"Status: {resp.status_code}")
            print(f"Response headers: {dict(resp.headers)}")
            print(f"Response text (first 500 chars):")
            print(repr(resp.text[:500]))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
