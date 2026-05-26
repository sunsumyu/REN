import asyncio
import httpx
import time
import sys
from pydantic import BaseModel, Field
from typing import List

# Import config from workspace
sys.path.append(r"d:\REN\qa")
import config

# Test schema for structured output check
class TestPlan(BaseModel):
    items: List[str] = Field(description="A list of 3 medical topics")

async def send_single_request(client: httpx.AsyncClient, req_id: int, model: str, is_structured: bool = False) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.LLM_API_KEY.strip()}" if config.LLM_API_KEY else "Bearer dummy"
    }
    
    messages = [{"role": "user", "content": f"Output 3 medical topics about liver care. Please return in JSON format. Request ID: {req_id}."}]
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }
    
    if is_structured:
        is_openai = model.lower().startswith("gpt")
        if is_openai:
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "TestPlan",
                    "strict": True,
                    "schema": TestPlan.model_json_schema()
                }
            }
        else:
            data["response_format"] = {
                "type": "json_object"
            }
        
    start_time = time.time()
    try:
        response = await client.post(config.LLM_API_URL, headers=headers, json=data, timeout=30.0)
        elapsed = time.time() - start_time
        status = response.status_code
        
        if status == 200:
            res_json = response.json()
            if "error" in res_json and res_json["error"]:
                err = res_json["error"]
                return {
                    "req_id": req_id,
                    "success": False,
                    "elapsed": elapsed,
                    "error_msg": err.get("message"),
                    "error_code": err.get("code"),
                    "status_code": 200
                }
            return {
                "req_id": req_id,
                "success": True,
                "elapsed": elapsed,
                "status_code": 200
            }
        else:
            return {
                "req_id": req_id,
                "success": False,
                "elapsed": elapsed,
                "error_msg": response.text,
                "status_code": status
            }
    except Exception as e:
        return {
            "req_id": req_id,
            "success": False,
            "elapsed": time.time() - start_time,
            "error_msg": str(e),
            "status_code": -1
        }

async def test_concurrency(model: str, concurrency_levels: List[int]):
    print("\n" + "="*60)
    print(f"🚀 STAGE 1: Concurrency Level Stress Test for model: {model}")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        for level in concurrency_levels:
            print(f"\nTesting Concurrency Level: {level} parallel requests...")
            tasks = [send_single_request(client, i, model) for i in range(level)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_elapsed = time.time() - start_time
            
            success_count = sum(1 for r in results if r["success"])
            failures = [r for r in results if not r["success"]]
            
            print(f"Completed in {total_elapsed:.2f} seconds.")
            print(f"Success Rate: {success_count} / {level} ({success_count/level*100:.1f}%)")
            
            if failures:
                print("First failure details:")
                print(f"  - Status Code: {failures[0]['status_code']}")
                print(f"  - Error Code: {failures[0].get('error_code')}")
                print(f"  - Error Msg: {failures[0].get('error_msg')}")
            else:
                print("✅ All requests succeeded perfectly at this concurrency level!")

async def test_cooling_time(model: str, delays: List[float]):
    print("\n" + "="*60)
    print(f"🚀 STAGE 2: Sequential Cooling Time (Interval Delay) Test for model: {model}")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        for delay in delays:
            print(f"\nTesting Delay Interval: {delay:.2f} seconds...")
            success_count = 0
            failures = []
            
            for i in range(3):
                if i > 0 and delay > 0:
                    await asyncio.sleep(delay)
                res = await send_single_request(client, i, model)
                if res["success"]:
                    success_count += 1
                else:
                    failures.append(res)
                    break
            
            if failures:
                print(f"❌ Rate limited after {success_count} successful requests with delay {delay}s.")
                print(f"  - Error Code: {failures[0].get('error_code')}")
                print(f"  - Error Msg: {failures[0].get('error_msg')}")
            else:
                print(f"✅ Succeeded perfectly! Delay of {delay}s is sufficient for sequential calls.")
                break

async def test_structured_compatibility(model: str):
    print("\n" + "="*60)
    print(f"🚀 STAGE 3: Structured Output Compatibility Test for model: {model}")
    print("="*60)
    
    async with httpx.AsyncClient() as client:
        is_openai = model.lower().startswith("gpt")
        format_name = "strict JSON Schema" if is_openai else "standard JSON Mode"
        print(f"Calling structured output with {format_name} on {model}...")
        res = await send_single_request(client, 999, model, is_structured=True)
        if res["success"]:
            print(f"✅ Success! Model '{model}' supports {format_name} structured outputs flawlessly.")
            print(f"   Elapsed Time: {res['elapsed']:.2f} seconds.")
        else:
            print(f"❌ Failed! Structured Output format ({format_name}) not compatible or timed out on '{model}':")
            print(f"   Error: {res.get('error_msg')} (Code: {res.get('error_code')})")

async def main():
    model = "gpt-4o"
    print(f"Starting Stress Test for '{model}' on gateway...")
    await test_concurrency(model, [10, 30, 50, 100])
    await test_cooling_time(model, [0.0, 0.5, 1.0, 2.0, 3.0])
    await test_structured_compatibility(model)

if __name__ == "__main__":
    asyncio.run(main())
