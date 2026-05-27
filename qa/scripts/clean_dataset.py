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
    
    # 2. 预编译的高抗干扰正则模式，匹配同义词与变形垃圾行
    garbage_patterns = [
        # 2.1 工程指令与格式类（JSON, schema, prompts, metadata）
        re.compile(r"(?:我们)?被要求(?:以|输出)?", re.I),
        re.compile(r"遵循.*(?:JSON|schema|Schema)", re.I),
        re.compile(r"输出.{0,3}JSON", re.I),
        re.compile(r"构建.{0,3}JSON", re.I),
        re.compile(r"格式约束|元指令|字段拼装|围栏|不含 markdown|禁止任何免责声明|所有输出必须为", re.I),
        re.compile(r"Q是“|确保输出JSON|按照schema", re.I),
        re.compile(r"facet|<facet|>\s*$", re.I),
        
        # 2.2 工具痕迹与免责避让（references, tool terms, warnings）
        re.compile(r"(?:避免|去除|移处|移除).{0,5}工具痕迹", re.I),
        re.compile(r"没有围栏|避让免责声明|医疗免责|无可靠参考文献", re.I),
        re.compile(r"(?:绝对)?不能出现.*工具", re.I),
        
        # 2.3 检索与忽略纠结（refs document filtering）
        re.compile(r"(?:应当|可以|需要|应当)?忽略.{0,10}无关", re.I),
        re.compile(r"检查.{0,10}refs", re.I),
        re.compile(r"排除.{0,10}无关", re.I),
        re.compile(r"与Q无关|与问题无关|与此问题无关|与本题无关|其他refs无关", re.I),
        re.compile(r"系统可能误关联|这与问题不直接相关|与本品无关|与贝美格无关", re.I),
        re.compile(r"忽略那些关于|忽略其他|只使用提供的refs|关键证据来自refs", re.I),
        re.compile(r"\brefs\b|相关refs|提供refs", re.I),
        
        # 2.4 中间步骤与思维树结构（planning markers）
        re.compile(r"推理链：|证据提取：|拆解(?:子)?问题|构建子问题|整理证据|提取证据|图谱关系：|图谱显示|知识关联", re.I),
        re.compile(r"final_conclusion_summary|answer_body|sub_questions|evidences|reasoning_chains", re.I),
        re.compile(r"step_id:\s*[A-Z]\d+|logic:\s*", re.I),
        re.compile(r"首先(?:，)?(?:需要)?(?:从|理解|分析|拆解)", re.I),
        re.compile(r"切面(?:是|视角)|视角(?:是|为)", re.I),
        re.compile(r"根据证据，直接提取|这亦属包装安全|回答正文", re.I),
        re.compile(r"这些都应纳入回答|根据facet|根据提供的refs|从refs看", re.I),
        re.compile(r"没有明确数据|目前系统内未提供|仅有的证据集中在|我们只选取与", re.I),
        re.compile(r"组织主线与强调重点|将作为证据|分析refs中|注意refs中|步骤包括识别问题" , re.I),
        re.compile(r"^\s*\d+[\.、\s].*？\s*$", re.I), # Numbered questions ending with a question mark
        re.compile(r"\b[P|p]\d+\b|\b[P|p]\d+[:：]", re.I), # step labels like P1, P2
        re.compile(r"我们将|我们来|我们需|我们可以|我们可以选择|我们可以通过|我们来整理", re.I), # first-person planning
        re.compile(r"考虑是否|没有提及|无相关|是否包含|是否需要|是否表述|是否提及|严谨回答", re.I) # self-debating scaffolding
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
            
        # 过滤结构化JSON的残余符号（比如单独的括号、逗号等）
        if stripped in ["{", "}", "[", "]", '"sub_questions": [', '"evidences": [', '"reasoning_chains": [', '"final_conclusion_summary": "",', '"answer_body": ""']:
            continue
            
        # 过滤包含垃圾模式的行
        is_garbage = False
        for pattern in garbage_patterns:
            if pattern.search(stripped):
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
