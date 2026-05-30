import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from api_client import APIClient
from scripts.medicalqa_purifier import smooth_planner_term, _SMOOTHED_PLANNER_CACHE

async def main():
    client = APIClient()
    
    test_terms = ["古籍收采", "包装形式", "指标偶联监测", "古籍收采"]
    
    print("🚀 Starting Paraphraser Verification Test...\n")
    
    for i, term in enumerate(test_terms):
        print(f"--- Test {i+1}: '{term}' ---")
        start_cache_len = len(_SMOOTHED_PLANNER_CACHE)
        
        # Test smoothing
        smoothed = await smooth_planner_term(client, term)
        
        end_cache_len = len(_SMOOTHED_PLANNER_CACHE)
        print(f"Result: '{smoothed}'")
        
        # Check cache state
        is_hit = start_cache_len == end_cache_len
        print(f"Cache state: {'HIT (0ms)' if is_hit else 'MISS (API Called)'}")
        print(f"Current Cache Keys: {list(_SMOOTHED_PLANNER_CACHE.keys())}\n")

if __name__ == "__main__":
    asyncio.run(main())
