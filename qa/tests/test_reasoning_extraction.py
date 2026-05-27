import asyncio
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from api_client import APIClient
from models import FacetPlan

async def test_reasoning():
    client = APIClient()
    try:
        print("Initializing supported models...")
        await client.init_supported_models()
        print("Supported models:", client.supported_models)
        
        # Test call_llm_with_reasoning
        prompt = "用一句话回答：为什么说感冒时需要多喝水？请写出你完整的思考过程。"
        print(f"\n--- Testing call_llm_with_reasoning ---")
        content, reasoning = await client.call_llm_with_reasoning(prompt, model_pool="premium")
        print(f"Content:\n{content}")
        print(f"\nReasoning Content:\n{reasoning}")
        
        # Test call_llm_structured
        print(f"\n--- Testing call_llm_structured ---")
        messages = [{"role": "user", "content": "为医疗问题 '阿司匹林在抗血小板治疗中的药理作用是什么？' 规划 3 个医学视角"}]
        result = await client.call_llm_structured(messages, FacetPlan, model_pool="lightweight")
        print(f"Parsed Result Facets: {result.facets}")
        reasoning_content = getattr(result, "_reasoning_content", None)
        print(f"\nAttached Reasoning Content:\n{reasoning_content}")
        
    except Exception as e:
        print("Error occurred during test:", e)
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_reasoning())
