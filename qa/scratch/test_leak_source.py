# -*- coding: utf-8 -*-
import asyncio
import sys
import json
import re
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from api_client import APIClient
from services.healing_service import HealingService
from core.purification_prompts import get_purify_system_prompt
from core.purification_helper import pre_strip_engineering_noise

async def main():
    client = APIClient()
    
    # Load raw dataset
    dataset_path = Path(__file__).resolve().parent.parent / "medical_qa_dataset.jsonl"
    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Line 249 is index 248
    line_data = json.loads(lines[248])
    q = line_data["Q"]
    planners = line_data["planners"]
    refs = line_data.get("refs", [])
    
    # Find "用药注意事项" facet
    p_data = None
    for p in planners:
        if p["planner"] == "用药注意事项":
            p_data = p
            break
            
    if not p_data:
        print("❌ 用药注意事项 facet not found in line 249!")
        return
        
    raw_answer = p_data["answer"]
    
    # Extract think block
    think_match = re.match(r"^\s*<think>([\s\S]*?)</think>([\s\S]*)$", raw_answer)
    if not think_match:
        print("❌ No think block found in raw answer!")
        return
    raw_think = think_match.group(1).strip()
    
    # Strip RAG and pmid noise
    facet_match = re.match(r"^\s*(<facet\s*=\s*[^>]+>)\s*([\s\S]*)$", raw_think)
    actual_raw_think = facet_match.group(2).strip() if facet_match else raw_think
    stripped_think = pre_strip_engineering_noise(actual_raw_think)
    
    system_prompt = get_purify_system_prompt("用药注意事项")
    
    prompt = f"""主问题 Q:
\"\"\"
{q}
\"\"\"

待参考医学事实文献 refs:
\"\"\"
{json.dumps(refs, ensure_ascii=False)}
\"\"\"

原始思维链 (CoT) 内容:
\"\"\"
{stripped_think}
\"\"\"

请严格按照净化重写指南，仅输出重构后的纯净思维链本身。"""

    print("Step 1: Calling Premium Model (deepseek-v4-pro) for main rewrite...")
    purified_premium = await client.llm_service.call_llm(
        prompt, 
        system_prompt=system_prompt, 
        model_pool="premium", 
        stage="Test-Premium-Rewrite",
        max_tokens=3072
    )
    purified_premium = purified_premium.replace("<think>", "").replace("</think>", "").strip()
    if purified_premium.startswith("```"):
        purified_premium = "\n".join(purified_premium.splitlines()[1:])
    if purified_premium.endswith("```"):
        purified_premium = "\n".join(purified_premium.splitlines()[:-1])
    purified_premium = purified_premium.strip()
    
    print("\n[DEBUG] PREMIUM MODEL OUTPUT START:")
    print(purified_premium[:800])
    print("[DEBUG] PREMIUM MODEL OUTPUT END\n")
    
    print("Step 2: Calling Healing Service (Lightweight: deepseek-v4-flash) for noise removal...")
    healing_service = HealingService(client.llm_service)
    healed_lightweight = await healing_service.heal_conversational_noise(purified_premium, line_num=249)
    
    print("\n[DEBUG] HEALED LIGHTWEIGHT OUTPUT START:")
    print(healed_lightweight[:800])
    print("[DEBUG] HEALED LIGHTWEIGHT OUTPUT END\n")

if __name__ == "__main__":
    asyncio.run(main())
