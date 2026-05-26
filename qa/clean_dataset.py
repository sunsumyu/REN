import json
import re
import os
import shutil

def clean_think_text(think_text):
    """
    精细化清洗思维链中的元指令、JSON Schema 指令以及非医学推理的垃圾信息。
    """
    # 1. 移除非医学推理的特定多行模式（如JSON模板展示）
    think_text = re.sub(r"遵循输出结构：\s*\{\s*\"sub_questions\":[\s\S]*?\}\s*", "", think_text, flags=re.IGNORECASE)
    
    lines = think_text.splitlines()
    cleaned_lines = []
    
    # 2. 逐行过滤垃圾元指令与工程噪声
    garbage_keywords = [
        "我们被要求",
        "遵循JSON schema",
        "严格遵循JSON",
        "遵循给定的schema",
        "Q是“",
        "只使用提供的refs",
        "关键证据来自refs",
        "其他refs无关",
        "排除无关的",
        "忽略无关的",
        "整理evidences",
        "整理证据",
        "提取证据",
        "reasoning_chains",
        "final_conclusion_summary",
        "answer_body",
        "sub_questions",
        "evidences",
        "让我们构建JSON",
        "现在生成JSON",
        "现在输出JSON",
        "最终输出JSON",
        "开始构建JSON",
        "按照schema",
        "拆解子问题",
        "构建子问题",
        "回答正文要",
        "避免工具痕迹",
        "确保引用证据",
        "location可写",
        "根据facet",
        "根据提供的refs",
        "从refs看",
        "系统可能误关联",
        "所有输出必须为",
        "没有围栏",
        "无可靠参考文献",
        "回答正文应分点",
        "因为说明书有",
        "我们用更权威",
        "这里有无关证据",
        "步骤包括识别问题",
        "推理链：步骤",
        "证据：提取refs",
        "注意refs中",
        "我们只选取与Q",
        "分析refs中",
        "这将作为证据",
        "根据证据，直接提取",
        "这些都应纳入回答",
        "没有明确数据",
        "目前系统内未提供",
        "仅有的证据集中在",
        "但这并不直接等价",
        "首先，理解问题",
        "首先，需要拆解",
        "首先，需要从",
        "我们被要求以",
        "视角是“",
        "组织主线与强调重点"
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
            
        # 过滤结构化JSON的残余符号（比如单独的括号、逗号等）
        if stripped in ["{", "}", "[", "]", '"sub_questions": [', '"evidences": [', '"reasoning_chains": [', '"final_conclusion_summary": "",', '"answer_body": ""']:
            continue
            
        # 过滤包含垃圾关键字的行
        is_garbage = False
        for kw in garbage_keywords:
            if kw in stripped:
                is_garbage = True
                break
                
        if is_garbage:
            continue
            
        # 保留真正的临床推理与证据内容
        cleaned_lines.append(line)
        
    # 3. 重新组装并清理多余的换行
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text) # 压缩连续空行
    return cleaned_text.strip()

def run_cleanup():
    dataset_path = r"d:\REN\qa\medical_qa_dataset.jsonl"
    backup_path = r"d:\REN\qa\medical_qa_dataset_raw.jsonl"
    
    if not os.path.exists(dataset_path):
        print(f"❌ 未找到数据集文件: {dataset_path}")
        return
        
    # 1. 自动备份原始文件
    if not os.path.exists(backup_path):
        print(f"📦 正在备份原始数据集到: {backup_path}")
        shutil.copyfile(dataset_path, backup_path)
    else:
        print(f"ℹ️ 原始备份已存在: {backup_path}")
        
    # 2. 读取并清洗
    print("🧹 正在进行废话与工程噪声清洗...")
    cleaned_count = 0
    total_planners = 0
    
    out_lines = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                out_lines.append(line)
                continue
                
            try:
                data = json.loads(line)
                planners = data.get("planners", [])
                
                for p in planners:
                    total_planners += 1
                    raw_answer = p.get("answer", "")
                    
                    # 匹配 <think> 块，允许首尾有空白字符
                    think_match = re.match(r"^\s*<think>([\s\S]*?)</think>([\s\S]*)$", raw_answer)
                    if think_match:
                        think_content = think_match.group(1)
                        answer_body = think_match.group(2)
                        
                        cleaned_think = clean_think_text(think_content)
                        
                        # 仅在有变化时更新并计数
                        if cleaned_think != think_content.strip():
                            cleaned_count += 1
                            
                        # 组装回去
                        p["answer"] = f"<think>\n{cleaned_think}\n</think>\n{answer_body.strip()}"
                
                out_lines.append(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"⚠️ 解析第 {i+1} 行时出错: {e}")
                out_lines.append(line)
                
    # 3. 写回原文件
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
        
    print(f"✨ 清洗完成！")
    print(f"📊 统计数据:")
    print(f"  - 总共处理切面 (Planners): {total_planners}")
    print(f"  - 成功净化思维链 (Think blocks): {cleaned_count}")
    print(f"💾 数据已写回: {dataset_path}")

if __name__ == "__main__":
    run_cleanup()
