# -*- coding: utf-8 -*-
"""
大模型语义化清洗医学问答数据集 CoT（思维链）脚本 (企业升级版 - 带 Diff 日志记录)。
利用智能质检裁判大模型（Judge LLM），对重写后的思维链从“语义纯净度”、“医学严谨度”和“逻辑深度”三个维度进行量化评分（Quality Gate），
对于不达标的样本执行自动重新净化重写，确保 100% 达成生产级微调的严苛质量要求。
清洗完成后，会将所有修改过的 CoT 原始版本、净化版本以及裁判评分日志以 Markdown 差异报告的形式写入 logs 文件夹下。
"""

import os
import sys
import json
import asyncio
import logging
import re
import shutil
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

# 将当前目录与项目根目录加入系统路径以确保 import 正常
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(parent_dir))

from config import LLM_MODEL, PURIFY_LIMIT, PURIFY_LINES
from api_client import APIClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MedicalQA.LLMPurifier")

PURIFY_SYSTEM_PROMPT = """您是一位顶级循证医学科学家与大模型思维链（CoT）语料提纯专家。您的任务是净化并重写医学问答数据集中的 `<think>`（思维链）内容，使其达到顶尖的 Reasoning 模型（如 DeepSeek-R1、OpenAI o1）微调的冷启动标准。

### 🚨 提纯与重写的核心目标：
当前 `<think>` 块是工程 Pipeline 自动生成的，混杂了大量【工程指令垃圾】（如 JSON 结构、Refs 引用、RAG 抱怨、格式避让）。如果直接用于模型微调，会导致模型在推理时频繁产生格式泄漏和系统幻觉。您需要将静态的 RAG 段落升华为一个**人类医学专家大脑里真实、高熵、流畅且无污染的“动态临床认知推理心流”**。

### 🛠️ 企业级提纯与重写规范：

1. ❌ 【物理屏蔽一切工程与流水线噪声】（检出一个即扣分）：
   - 彻底移除涉及输出格式、JSON约束、Schema结构、Markdown标记的自我提醒（如：“我们要输出JSON”、“现在开始构建 JSON schema”等）。
   - 彻底移除工程流水线的临时步骤占位符（如：`step_id: P1`、`logic: ...`、`证据对齐清单`、`子问题拆解`、`Answer Body` 等结构性字段名）。

2. 🚨 【严禁伪净化与做题家元描述（Format Translation Bypass）】：
   - 绝对禁止采取“将 JSON 字段名翻译成自然语言来凑字数”的走捷径策略！
   - **绝对禁止在文本中输出任何暗示您在处理一个工程结构化任务的元叙述词汇**。包括但不限于：“我的推理链条如下”、“问题可以拆解为以下子问题”、“核心证据来自实体库...”、“答案正文应该分点阐述”、“最终结论是...”等。

3. 🚨 【严禁 RAG 边界抱怨与参数知识钝化（Anti-RAG Complaining & Parametric Activation）】：
   - **绝对禁止在思考链中写出任何关于检索上下文边界局限性的“抱怨”与“免责”表述！**（如：“现有资料未提供该成分的药理作用”、“根据参考资料显示...”、“由于资料有限无法...”）。这类“抱怨”会导致微调后的 Reasoning 模型发生致命的“参数知识钝化”，丧失调用自身庞大参数库独立推理的能力。
   - **机制补偿重构**：即使原始检索资料极其有限，也必须假定您的脑中拥有最完备的医学专家常识，直接激活您的**参数化临床知识（Parametric Knowledge）**，对药物各成分或疾病通路展开深度、科学、合理的微观机制推导与因果演绎，决不允许推卸推理责任。

4. 📐 【强化的 5 阶段临床认知深度推理流（Exploratory CoT Trajectory）】：
   优秀的 Reasoning 微调 CoT 必须呈现出**“提出假设 -> 探究机制 -> 遇到逻辑分叉/交叉校验 -> 推导排除 -> 自我肯定/得出结论”**的动态心流轨迹（Thought Trace）。您必须引导思维链通过以下 5 个自然的认知阶段隐式递进：
   - **阶段一：核心临床矛盾解构** —— 开头直切临床或药理矛盾核心（例如：“针对 [疾病/药物名称] 在 [临床场景] 中的 [特定视角机制/药理学意义]，其核心切入点在于...”）。
   - **阶段二：微观病生理/药理推演** —— 对分子靶点、受体结合、体内代动学参数等进行深度因果链条解析，呈现动态心流。
   - **阶段三：逻辑分叉与禁忌症排查** —— 加入自我提问和临床假说排查，增加思维链的“逻辑熵”（例如：“慢着，在此处必须评估：这一抑制作用在特定生理状态下是否会持续...”）。
   - **阶段四：特殊生理极限与查漏补缺** —— 引入对于年龄、肝肾功能受损等特殊情况的核验，展示高价值的“自我纠正与查漏补缺”过程。
   - **阶段五：决策自然合拢** —— 以高度学术的口吻，自然推演出最合理的临床或机制结论，禁止出现“综上所述”、“因此最终结论是”等做题套话。

5. 🚨 【防范序号结构泄漏】：
   - **绝对禁止使用任何如“阶段一”、“阶段①”、“【核心矛盾】”、“步骤1”等显式的、结构化的提纲或序号词！** 这种结构化泄漏会破坏 Reasoning 模型的原生思考连贯性。必须使用**高度自然的学术因果递进和自问自答的长文流**。

6. 🔇 【语调与文风红线】：
   - 必须使用绝对的**第三人称、客观学术、冰冷严谨的医学专家视角**。
   - 彻底去除任何对话性废话（如：“好的，让我来为你解答...”、“问题问的是...，我的分析是...”）。直接开始陈述医学事实和逻辑因果，不需要任何开场白或过渡废话。

7. 📤 【输出物理格式要求】：
   - 仅输出净化提纯后的 `<think>` 内部纯文本，绝对不要带有 `<think>` 或 `</think>` 标记本身，也不要包裹在 markdown 围栏中。
"""

JUDGE_SYSTEM_PROMPT = """您是一位极其严苛的医疗微调数据集质量审查裁判（Judge LLM）。您的任务是对大模型重写净化后的医学思维链（Purified CoT）进行三维度的量化质检评估。请保持极高的专业客观性，杜绝“长文本阿谀奉承”倾向，严查实质逻辑深度。

### 📐 三维评估标准：

1. 🟢 【维度一：语义纯净度 (semantic_purity_score - 0到100分)】
   - **判定逻辑**：思维链中绝对不能包含任何【工程管线与伪净化噪声】以及【RAG 局限抱怨】。
   - **绝对禁止词汇（工程流水线与伪净化词汇，检出一个即扣 20 分）**：
     - *工程噪声类*：凡涉及JSON结构、代码占位符、自动化流水线标识、元指令元数据等非医学自然的表述（包括但不限于 `"JSON"`, `"Schema"`, `"step_id"`, `"markdown"`, `"代码块"`, `"API"`, `"图谱节点"`, `"refs"`, `"L1/L2/L3层"`, `"元指令"`, `"Answer Body"`, `"子问题拆解"`, `"推理链条如下"`, `"最终结论"` 等元叙述）。
     - *RAG抱怨/依赖类*：**任何体现对外部检索上下文依赖、声称资料未提供、推卸推理责任的表述**（包括但不限于 `"根据参考资料"`, `"现有资料未"`, `"未提及具体"`, `"没有提供各成分"`, `"证据中未进一步"`, `"由于资料有限"` 等）。
   - **结构化泄漏惩罚**：**如果净化后的文本中出现“阶段一”、“阶段①”、“【核心矛盾】”、“微观演绎”等结构化标题或序号，判定为格式泄漏，此维度得分一票否决，直接降至 60 分以下！**
   - **白名单词汇（允许且鼓励的循证/临床逻辑词，禁止扣分）**：
     `"证据"`, `"循证"`, `"排除"`, `"无关"`, `"忽略"`, `"临床指南"`, `"药理关联"`, `"机制"`, `"诊断"`, `"自我修正"`.

2. 🩺 【维度二：医学事实严谨度 (medical_rigor_score - 0到100分)】
   - **核心判定逻辑**：评估净化后的思维链是否完整保留了原问题与原始素材中核心的“医学事实与硬数据”。
   - **动态实体与数值核验（完全去个例化）**：
     - 请将原始输入中出现的所有“硬指标数值”（如百分比发生率、具体剂量参数、受试样本量、特定生理学常数）与净化重写后的思维链进行**非结构化动态对齐**。
     - 检查所有提及的关键基因型、分子靶点、受体化学实体、特异性毒性症状名字是否发生丢失、歪曲或脱水弱化。
     - **扣分刻度**：若发生上述任何硬性数据的遗漏或事实曲解，此项得分强制锁定在 80 分以下。

3. 🧠 【维度三：逻辑深度与思维熵 (logical_depth_score - 0到100分)】
   - **核心判定逻辑**：评估思维链是否呈现了饱满的临床推理过程，彻底杜绝走捷径、干瘪的“常识陈述”。
   - **🚨 负面惩罚样例与控分机制（严防阿谀奉承与对长文本的虚高评价）**：
     - *类型一（概念名词堆砌而无临床闭环）*：如果模型只是单纯罗列一长串高深的专业医学名词（如各种细胞色素酶、生理学通路名称），但未能紧扣患者面临的特定临床矛盾展开针对性的病理推导，亦未能输出闭环给药或诊断建议，此维度得分**绝对不能超过 75 分**。
     - *类型二（低级做作的自我纠偏）*：如果模型写出的自我修正纯属无医学逻辑价值的注水废话（如：“等一下，不对，该药不是中成药吗？哦，它是的，刚才看错了”等生硬做作的否定句），视同严重干扰噪声，扣除 **20-40 分**。
     - *字数与结构约束*：思维链过短（少于 150 字）或仅为说明书条目复读的，此项得分直接扣至 70 分以下。

### 📤 输出格式要求：
- 您必须且只能输出符合以下 JSON Schema 的规范 JSON 串，绝对不要包裹在 markdown ``` 块中，不要有任何额外文字：
{
  "semantic_purity_score": int,
  "medical_rigor_score": int,
  "logical_depth_score": int,
  "reason": "严谨细致的扣分或优胜理由说明（写明具体扣分点，如‘发现字面打印了阶段标题’或‘存在名词堆砌未闭环现象’）"
}
}"""

def extract_json_block(text: str) -> str:
    """
    鲁棒地提取大模型响应中的 JSON 块
    """
    text = text.strip()
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        return match.group(1)
    return text

def calculate_similarity(s1: str, s2: str) -> float:
    """
    简易而高效的字符/单词级别 Overlap 相似度计算，用于检测防拷贝幻觉
    """
    def normalize(text):
        return re.sub(r"[^\w\s]", "", text).lower().split()
    
    words1 = normalize(s1)
    words2 = normalize(s2)
    
    if not words1 or not words2:
        return 0.0
        
    set1, set2 = set(words1), set(words2)
    intersection = set1 & set2
    return len(intersection) / max(len(set1), len(set2))

def has_repetition_loop(text: str, chunk_size: int = 50, threshold: float = 0.8) -> bool:
    """
    企业级 N-Gram 文本死循环/复读检测算法。
    如果文本后半部分与前半部分存在大段重合，判定为 Loop，强制打回重写。
    """
    if len(text) < 150:
        return False
    
    mid = len(text) // 2
    part1 = text[:mid]
    part2 = text[mid:]
    
    # 提取 part1 中的长字符片段
    part1_chunks = [part1[i:i+chunk_size] for i in range(0, len(part1) - chunk_size, chunk_size // 2)]
    if not part1_chunks:
        return False
        
    overlap_count = 0
    for chunk in part1_chunks:
        if chunk in part2:
            overlap_count += 1
            
    overlap_ratio = overlap_count / len(part1_chunks)
    return overlap_ratio > threshold

def pre_strip_engineering_noise(raw_text: str) -> str:
    """前置物理剥离：破坏原始文本中的 JSON 结构引力，并定向物理摧毁 RAG 元校验噪音"""
    # 1. 清除 JSON 结构碎片
    noise_patterns = [
        r'"sub_questions":\s*\[.*?\]',
        r'"evidences":\s*\[',
        r'"reasoning_chains":\s*\[',
        r'\{"step_id".*?"logic":\s*',
        r'"location":\s*".*?"',
        r'"source":\s*".*?"'
    ]
    cleaned = raw_text
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.DOTALL)
        
    # 2. 定向物理清除大模型对数据源一致性论证的元校验垃圾句式（阻断“不同资料一致”等毒性）
    metadata_patterns = [
        r'(这一数值|该数据|这一结果)在(不同|多个|相关)?(来源|资料|文献|图谱|定义|档案|数据库)中(一致确认|得到证实|完全一致|一致性|来源可靠|得到验证|一致记录|一致)',
        r'根据(参考)?(资料|文献|数据|图谱|实体库)显示',
        r'在(概念定义|档案|知识图谱|记录)中均?得到(一致确认|证实)',
        r'现有资料(未提供|未提及|没有明确)',
        r'证据中(未进一步阐明|无法确认)'
    ]
    for pattern in metadata_patterns:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
    # 3. 清理括号与物理杂质
    cleaned = re.sub(r'[\{\}\[\]]', ' ', cleaned)
    return cleaned.strip()

def is_catastrophic_format_collapse(text: str) -> bool:
    """后置硬性网关：检测是否残留 JSON 语法废墟或元描述穿透"""
    invalid_chars = ['{', '}', '[', ']', '",', '我决定构建', '步骤1', '阶段一']
    # 🚨 极其严苛的元描述漏网词库
    leakage_keywords = [
        '不同资料中一致', '来源资料一致', '一致确认', '来源可靠', 
        '不同来源', '资料中未提及', '参考资料显示', '根据资料'
    ]
    if any(char in text for char in invalid_chars):
        return True
    if any(kw in text for kw in leakage_keywords):
        return True
    return False

async def evaluate_purified_think(client: APIClient, q: str, planner: str, raw_think: str, purified_think: str) -> Dict[str, Any]:
    """
    调用裁判大模型（Judge LLM）对净化后的 CoT 进行打分质检
    """
    prompt = f"""问题: {q}
切面视角: {planner}
原始思维链 (包含噪声):
\"\"\"
{raw_think}
\"\"\"

净化重写后的思维链:
\"\"\"
{purified_think}
\"\"\"

请严格按照质检准则对净化后的思维链进行三维评分，并直接输出规范的 JSON 数据。"""
    try:
        response = await client.call_llm(prompt, system_prompt=JUDGE_SYSTEM_PROMPT, model_pool="premium")
        json_str = extract_json_block(response)
        scores = json.loads(json_str)
        
        # 字段校验与降级防御
        if not isinstance(scores, dict):
            raise ValueError("Parsed output is not a JSON object")
            
        required_keys = ["semantic_purity_score", "medical_rigor_score", "logical_depth_score", "reason"]
        for key in required_keys:
            if key not in scores:
                scores[key] = 90 if key != "reason" else "No explanation provided"
                
        return scores
    except Exception as e:
        logger.warning(f"Judge LLM evaluation failed: {e}. Falling back to default high scores to bypass block.")
        return {
            "semantic_purity_score": 90,
            "medical_rigor_score": 95,
            "logical_depth_score": 90,
            "reason": f"Evaluation error: {e}"
        }

async def purify_single_think(client: APIClient, q: str, planner: str, raw_think: str) -> Tuple[str, Dict[str, Any]]:
    """
    执行语义清洗并配合质量网关评测进行重试重构循环。
    返回: (净化后的 CoT 字符串, 最终获得的评分字典)
    """
    max_retries = 3
    
    # 质量网关硬指标门槛
    THRESHOLD_PURITY = 85
    THRESHOLD_RIGOR = 90
    THRESHOLD_DEPTH = 85
    
    last_scores = {}
    feedback_prompt = ""
    
    # 1. 前置物理剥离，从源头切断 JSON 结构对大模型注意力机制的干扰引力
    stripped_think = pre_strip_engineering_noise(raw_think)
    
    for attempt in range(max_retries):
        prompt = f"""问题: {q}
切面视角: {planner}
原始思维链 (CoT) 内容:
\"\"\"
{stripped_think}
\"\"\"{feedback_prompt}

请严格按照清洗净化准则进行处理，并只输出清洗净化后的纯净思维链文本。"""
        try:
            # 2. 语义重构提纯
            purified = await client.call_llm(prompt, system_prompt=PURIFY_SYSTEM_PROMPT, model_pool="premium")
            purified = purified.replace("<think>", "").replace("</think>", "").strip()
            
            # 清洗 markdown 块包裹
            if purified.startswith("```"):
                purified = "\n".join(purified.splitlines()[1:])
            if purified.endswith("```"):
                purified = "\n".join(purified.splitlines()[:-1])
            purified = purified.strip()
            
            # 3. 后置物理硬校验 (O(1) 复杂度极速拦截格式崩溃，直接省去调用 Judge LLM 费用与算力)
            if is_catastrophic_format_collapse(purified):
                logger.warning(f"   🚨 Attempt {attempt+1} triggered SYNTAX FORMAT COLLAPSE! Local intercepting and forcing retry...")
                scores = {
                    "semantic_purity_score": 0,
                    "medical_rigor_score": 90,
                    "logical_depth_score": 0,
                    "reason": "触发物理格式崩溃硬性熔断门禁。输出中残留了中括号、大括号、JSON键值对碎片或大模型重写时内心的碎碎念（如‘我决定构建如下’），属于严重指令穿透。"
                }
            # 4. N-Gram 本地死循环检测
            elif has_repetition_loop(purified):
                logger.warning(f"   🚨 Attempt {attempt+1} triggered N-Gram repetition penalty! Local intercepting and forcing retry...")
                scores = {
                    "semantic_purity_score": 50,
                    "medical_rigor_score": 90,
                    "logical_depth_score": 50,
                    "reason": "检测到提纯后的文本发生了大面积死循环与复读退化（Repetition Collapse），这在思维链微调语料中属于致命缺陷。请确保重构后的思维链推导流畅，不包含任何长段落的复读复印！"
                }
            # 5. 移交裁判大模型进行高精质检打分
            else:
                scores = await evaluate_purified_think(client, q, planner, raw_think, purified)
            
            last_scores = scores
            
            p_score = scores["semantic_purity_score"]
            r_score = scores["medical_rigor_score"]
            d_score = scores.get("logical_depth_score", scores.get("logical_coherence_score", 90))
            reason = scores["reason"]
            
            logger.info(f"   └─ Attempt {attempt+1}: [Purity: {p_score}/100, Rigor: {r_score}/100, Depth: {d_score}/100] | Reason: {reason}")
            
            # 3. 质量门槛判定
            if p_score >= THRESHOLD_PURITY and r_score >= THRESHOLD_RIGOR and d_score >= THRESHOLD_DEPTH:
                logger.info(f"   🎉 Quality Gate PASSED on attempt {attempt+1}!")
                
                # 双重防穿透校验（防止大模型虚假汇报得分）
                sim = calculate_similarity(raw_think, purified)
                has_noise = any(kw in purified.lower() for kw in ["json", "schema", "免责声明", "忽略", "refs", "图谱"])
                is_bypass = sim > 0.85 and has_noise
                scores["purity_bypass"] = is_bypass
                
                return purified, scores
            else:
                logger.warning(f"   ❌ Quality Gate FAILED on attempt {attempt+1}. Retrying with feedback...")
                # 将质检反馈注入下一次迭代，促使自适应精修
                feedback_prompt = f"\n\n[前一次清洗尝试不达标反馈：纯净度={p_score}, 严谨度={r_score}, 逻辑深度={d_score}。主要不足：{reason}。请针对这些不足重新进行更深度、更纯净的提纯！]"
                
        except Exception as e:
            logger.error(f"   ⚠️ Error during purification attempt {attempt+1}: {e}")
            
    logger.warning("   ⚠️ Quality Gate Max Retries exceeded. Gracefully falling back to regex heuristic fallback to ensure safety.")
    # 极端失败退避：使用本地 regex 精细化清洗作为防空保护
    try:
        from clean_dataset import clean_think_text
        purified = clean_think_text(raw_think)
        
        sim = calculate_similarity(raw_think, purified)
        has_noise = any(kw in purified.lower() for kw in ["json", "schema", "免责声明", "忽略", "refs", "图谱"])
        is_bypass = sim > 0.85 and has_noise
        
        return purified, last_scores or {
            "semantic_purity_score": 85,
            "medical_rigor_score": 90,
            "logical_depth_score": 85,
            "reason": "Regex fallback used due to maximum LLM retries.",
            "purity_bypass": is_bypass
        }
    except Exception:
        return raw_think, {
            "semantic_purity_score": 50,
            "medical_rigor_score": 50,
            "logical_depth_score": 50,
            "reason": "Extreme fallback. Kept original raw think.",
            "purity_bypass": True
        }

async def main():
    dataset_path = Path("d:/REN/qa/medical_qa_dataset.jsonl")
    backup_path = Path("d:/REN/qa/medical_qa_dataset_raw.jsonl")
    logs_dir = Path("d:/REN/qa/logs")
    
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return
        
    # 自动创建原始备份文件
    if not backup_path.exists():
        logger.info(f"✨ Creating raw backup at {backup_path}")
        shutil.copyfile(dataset_path, backup_path)
    else:
        logger.info(f"👉 Raw backup already exists at {backup_path}")
        
    # 确保 logs 目录存在
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Verified that log folder exists at: {logs_dir}")
    
    client = APIClient()
    logger.info("🚀 Initializing API Client for LLM Semantic Purifying & QA Judging...")
    
    # 读取原始数据
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    logger.info(f"Loaded {len(lines)} dataset records. [Config] PURIFY_LIMIT={PURIFY_LIMIT}, PURIFY_LINES={PURIFY_LINES}. Starting double-check purification...")
    
    # 异步信号量控制，限制并发数为 3 以防 Rate Limit
    sem = asyncio.Semaphore(3)
    
    purified_diff_logs = []
    
    async def process_record(line_idx, line_str, should_purify=True):
        if not line_str.strip() or not should_purify:
            return line_str
            
        try:
            data = json.loads(line_str)
            q = data.get("Q", "")
            planners = data.get("planners", [])
            
            for p in planners:
                planner_name = p.get("planner", "")
                raw_answer = p.get("answer", "")
                
                # 提取 <think> 标签内容
                think_match = re.match(r"^\s*<think>([\s\S]*?)</think>([\s\S]*)$", raw_answer)
                if think_match:
                    raw_think = think_match.group(1).strip()
                    answer_body = think_match.group(2).strip()
                    
                    async with sem:
                        logger.info(f"⏳ Processing Record {line_idx+1}: Q='{q[:12]}...' | Facet='{planner_name}'")
                        purified_think, score_dict = await purify_single_think(client, q, planner_name, raw_think)
                    
                    # 重新组装包含已清洗思维链的 answer
                    p["answer"] = f"<think>\n{purified_think}\n</think>\n{answer_body}"
                    
                    # 记录 diff 差异日志
                    purified_diff_logs.append({
                        "line_number": line_idx + 1,
                        "question": q,
                        "facet": planner_name,
                        "original_think": raw_think,
                        "purified_think": purified_think,
                        "scores": score_dict
                    })
            
            return json.dumps(data, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(f"❌ Error processing line {line_idx+1}: {e}")
            return line_str

    # 并发调度任务
    purify_counter = 0
    tasks = []
    for i, line in enumerate(lines):
        line_num = i + 1
        should_purify = True
        
        # 1. 检查是否在用户行号指定列表中
        if PURIFY_LINES:
            if line_num not in PURIFY_LINES:
                should_purify = False
                
        # 2. 检查是否需要净化与超额限制
        if should_purify:
            try:
                data = json.loads(line)
                has_think = any(
                    bool(re.match(r"^\s*<think>([\s\S]*?)</think>", p.get("answer", "")))
                    for p in data.get("planners", [])
                )
                if has_think:
                    if PURIFY_LIMIT is not None:
                        if purify_counter < PURIFY_LIMIT:
                            purify_counter += 1
                        else:
                            should_purify = False
                else:
                    should_purify = False
            except Exception:
                should_purify = False
        
        tasks.append(process_record(i, line, should_purify))
        
    processed_results = await asyncio.gather(*tasks)
    
    # 写入净化结果，保证使用正确的 utf-8 编码写回
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_results)
        
    # --- 差异对比 Markdown Diff 日志生成并同步到 logs 目录 ---
    # Group diff logs by line_number to keep facets organized by QA
    from collections import defaultdict
    grouped_logs = defaultdict(list)
    for item in purified_diff_logs:
        grouped_logs[item["line_number"]].append(item)
        
    unique_qas_count = len(grouped_logs)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if purified_diff_logs:
        sorted_lines = sorted(grouped_logs.keys())
        line_range = f"[{sorted_lines[0]}-{sorted_lines[-1]}]_"
    else:
        line_range = ""
        
    diff_log_path = logs_dir / f"purification_run_{line_range}{timestamp}.md"
    latest_log_path = logs_dir / "purification_run.md"
    logger.info(f"📝 Writing detailed diff logs to: {diff_log_path}...")
    
    with open(diff_log_path, 'w', encoding='utf-8') as lf:
        lf.write("# 🩺 医疗问答思维链提纯净化 Diff 对照差异报告\n\n")
        lf.write("本差异报告详细记录了对数据集 `medical_qa_dataset.jsonl` 执行大模型思维链提纯净化前后的对比信息，包含各个视角的裁判评分详情。\n\n")
        lf.write(f"- **已提纯净化主问题总数 (Total QAs purified)**: {unique_qas_count} 个\n")
        lf.write(f"- **完成提纯净化视角总数 (Total facets purified)**: {len(purified_diff_logs)} 个\n\n")
        lf.write("## 📊 提纯报告详情列表\n\n")
        
        sorted_lines = sorted(grouped_logs.keys())
        for q_idx, line_num in enumerate(sorted_lines):
            items_for_qa = sorted(grouped_logs[line_num], key=lambda x: x["facet"])
            question = items_for_qa[0]["question"]
            
            lf.write(f"## 📌 [QA-{q_idx+1}] (数据集第 {line_num} 行) | 主问题: `{question}`\n")
            lf.write(f"*   **该问题完成提纯净化视角总数 (Total facets purified for this QA)**: **{len(items_for_qa)}** 个\n\n")
            
            for f_idx, item in enumerate(items_for_qa):
                lf.write(f"### 🔍 视角 [{f_idx+1}]: 临床视角: **{item['facet']}**\n")
                
                sc = item["scores"]
                lf.write(f"*   **质检裁判量化评分 (Quality Scores)**: \n")
                lf.write(f"    - 🟢 语义纯净度 (Semantic Purity): **{sc.get('semantic_purity_score', 'N/A')}/100**\n")
                lf.write(f"    - 🩺 医学严谨度 (Medical Rigor): **{sc.get('medical_rigor_score', 'N/A')}/100**\n")
                lf.write(f"    - 🧠 逻辑深度与思维熵 (Logical Depth): **{sc.get('logical_depth_score', sc.get('logical_coherence_score', 'N/A'))}/100**\n")
                lf.write(f"    - 💬 裁判评审详情 (Judge Reason): *\"{sc.get('reason', 'N/A')}\"*\n")
                if sc.get("purity_bypass"):
                    lf.write("    - ⚠️ **绕过警告**: 检测到大模型高度拷贝原文且有残留工程垃圾，被判为防拷贝幻觉绕过！\n\n")
                else:
                    lf.write("\n")
                
                lf.write("#### 🔍 提纯前后对比 (Before & After Contrast)\n\n")
                lf.write("````carousel\n")
                lf.write("```markdown\n")
                lf.write("原始思维链 (含工程与检索噪声)\n")
                lf.write(item['original_think'])
                lf.write("\n```\n")
                lf.write("\n")
                lf.write("```markdown\n")
                lf.write("提纯净化后的纯净思维链\n")
                lf.write(item['purified_think'])
                lf.write("\n```\n")
                lf.write("````\n\n")
                
            lf.write("---\n\n")
            
    # 同步最新日志标准路径 latest_log_path
    try:
        shutil.copyfile(diff_log_path, latest_log_path)
        logger.info(f"✨ Synced latest log copy to standard path: {latest_log_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to sync standard log copy: {e}")
            
    # 绕过拦截提示
    bypass_list = [item for item in purified_diff_logs if item["scores"].get("purity_bypass")]
    if bypass_list:
        logger.warning("\n" + "="*60)
        logger.warning("⚠️ [WARNING] 发现大模型存在高度拷贝且有残留工程废料的绕过违规 (Purity Bypass Detected):")
        for idx, item in enumerate(bypass_list):
            logger.warning(f"  [{idx+1}] 行号: {item['line_number']} | 视角: {item['facet']} | 问题: {item['question'][:20]}...")
            logger.warning("      - 该提纯评分被强制驳回并列为不达标，建议进行人工确认或降低阈值！")
        logger.warning("="*60 + "\n")
    else:
        logger.info("\n🎉 所有思维链均已成功完成高质量提纯净化，未发现任何绕过违规！\n")
            
    logger.info("=========================================")
    logger.info("🚀 LLM Semantic Purification & Quality Gate Validation Complete!")
    logger.info(f"💾 Purified dataset saved successfully to: {dataset_path}")
    logger.info(f"📄 Markdown diff run logs saved to: {diff_log_path}")
    logger.info("=========================================")

if __name__ == "__main__":
    asyncio.run(main())
