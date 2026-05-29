import json

def inspect():
    results = []
    
    # Raw dataset
    try:
        with open('d:/REN/qa/medical_qa_dataset_raw.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            data = json.loads(line)
            if "商陆皂苷辛" in data.get('Q', ''):
                planners = [p['planner'] for p in data.get('planners', [])]
                results.append(f"RAW Line {i+1} planners ({len(planners)}): {planners}")
                break
    except Exception as e:
        results.append(f"Error reading raw: {e}")
        
    # Purified dataset
    try:
        with open('d:/REN/qa/medical_qa_dataset.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            data = json.loads(line)
            if "商陆皂苷辛" in data.get('Q', ''):
                planners = [p['planner'] for p in data.get('planners', [])]
                results.append(f"PURIFIED Line {i+1} planners ({len(planners)}): {planners}")
                break
    except Exception as e:
        results.append(f"Error reading purified: {e}")
        
    out_content = "\n".join(results)
    print(out_content)
    with open('d:/REN/qa/inspect_q24_results.txt', 'w', encoding='utf-8') as f:
        f.write(out_content)

if __name__ == "__main__":
    inspect()
