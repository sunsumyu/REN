import json
import re
import os

def clean_reasoning_content(think_text):
    """
    清洗思维链 <think> 中的微调无关提示词、大模型元指令和 JSON Schema 相关工程噪声。
    """
    patterns_to_remove = [
        # JSON Schema 及结构化提取指令
        r"我们被要求输出一个JSON对象，遵循给定的schema.*?\n",
        r"需要基于提供的refs，严格遵循JSON schema输出。?\n",
        r"Q是“.*?”，并且只使用提供的refs。?\n",
        r"整理evidences列表：.*?\n",
        r"reasoning_chains:.*?\n",
        r"现在输出JSON。确保没有markdown包裹。直接输出JSON字符串。?\n",
        r"按照schema，输出一个JSON对象，包含.*?\n",
        r"整理evidences：.*?\n",
        r"分析refs中与.*?相关的条目。?\n",
        r"我们将提取这些作为证据。忽略无关的.*?。?\n",
        r"让我们构建JSON。?\n",
        r"根据证据，直接提取即可。?\n",
        r"在回答中不能出现任何安全免责声明。?\n",
        r"注意refs中包含关于.*?的关联，以及一些无关的.*?内容，必须仅基于refs。?\n",
        r"refs中相关：.*?\n",
        r"由于视角是“.*?”，回答应强调.*?。?\n",
        r"因无配伍禁忌的明确数据，只能说.*?。?\n",
        r"回答正文应分点，如：.*?\n",
        r"最终输出JSON。?\n",
        r"我们被要求以“.*?”视角，生成一个关于“.*?”的结构化回答。?\n",
        r"遵循输出结构：\n\{\n\s+\"sub_questions\":[\s\S]*?\}\n",
        r"现在生成JSON。?\n",
        r"构建子问题：从.*?角度拆分.*?\n"
    ]
    
    cleaned = think_text
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    # 移除空行，保持排版紧凑
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines)

def process_dataset(input_file, output_sft_file):
    if not os.path.exists(input_file):
        print(f"Error: 找不到输入的数据集文件 {input_file}")
        return
        
    sft_records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                q = data.get("Q", "")
                planners = data.get("planners", [])
                summary = data.get("summary", "")
                
                # 1. 生成多视角微调样本 (切面专业问答)
                for p in planners:
                    facet = p.get("planner", "")
                    raw_answer = p.get("answer", "")
                    
                    # 剥离 <think> 标签并清洗工程指令
                    think_match = re.match(r"^<think>([\s\S]*?)</think>\s*([\s\S]*)$", raw_answer)
                    if think_match:
                        think_content = think_match.group(1).strip()
                        answer_body = think_match.group(2).strip()
                        
                        cleaned_think = clean_reasoning_content(think_content)
                        formatted_answer = f"<think>\n{cleaned_think}\n</think>\n{answer_body}"
                    else:
                        formatted_answer = raw_answer
                    
                    sft_records.append({
                        "instruction": f"你是一个顶级临床医学分析专家，请针对以下医学提问，从【{facet}】视角给出深入的专业循证分析：",
                        "input": q,
                        "output": formatted_answer
                    })
                    
                # 2. 生成多角度总结微调样本 (全局循证综合)
                sft_records.append({
                    "instruction": "你是一个顶级临床医学分析专家，请针对以下医学提问，整合多重视角进行全面、权威的医学循证解答与总结分析：",
                    "input": q,
                    "output": summary
                })
                
            except Exception as e:
                print(f"解析第 {i+1} 行数据时发生异常，已跳过: {e}")
                
    with open(output_sft_file, 'w', encoding='utf-8') as out_f:
        json.dump(sft_records, out_f, ensure_ascii=False, indent=2)
        
    print(f"🎉 转换完成！共生成 {len(sft_records)} 条 SFT 微调样本。")
    print(f"💾 已保存至: {output_sft_file}")

if __name__ == "__main__":
    # 默认路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "medical_qa_dataset.jsonl")
    output_path = os.path.join(base_dir, "medical_qa_sft_cleaned.json")
    process_dataset(input_path, output_path)
