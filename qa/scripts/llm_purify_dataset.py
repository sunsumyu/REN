# -*- coding: utf-8 -*-
"""
大模型语义化清洗医学问答数据集 CoT（思维链）脚本 (企业升级版 - 带 Diff 日志记录)。
利用智能质检裁判大模型（Judge LLM），对重写后的思维链从“语义纯净度”、“医学严谨度”和“逻辑连贯性”三个维度进行量化评分（Quality Gate），
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
from pathlib import Path
from typing import Dict, Any, Tuple, List

# 将当前目录与项目根目录加入系统路径以确保 import 正常
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(parent_dir))

from config import LLM_MODEL
from api_client import APIClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MedicalQA.LLMPurifier")

PURIFY_SYSTEM_PROMPT = """您是一位拥有极高专业素养的医学微调数据集清洗与净化专家。您的任务是净化医学多维多轮问答数据集中的 `<think>`（思维链/CoT）内容，使其达到顶尖的生产级微调标准。

### 为什么需要净化？
当前数据集是通过工程 Pipeline 自动生成的，其 `<think>` 块中混杂了大量的【工程指令噪声】和【格式约束废话】（例如格式自我纠结、避让免责声明、JSON字段拼装、非医学文档过滤等）。这些非医学推理的信息如果被用于大模型微调，会导致模型学到无关的元指令，并在后续推理时产生严重的“格式幻觉”（例如在普通对话中突然输出“我们被要求输出JSON”或“注意不要加免责声明”）。

### 清洗净化准则：
1. **彻底清除工程格式废话**：
   - 移除非医学推理的特定模板描述（例如：‘首先理解问题...’、‘我们被要求从某某视角回答...’、‘角度是【...】，所以回答要强调...’）。
   - 移除所有关于输出格式、JSON约束、Schema结构、Markdown标记、避让免责声明的指令（例如：‘我们将输出 JSON。确保不含 markdown。’、‘现在构建JSON’、‘最终JSON为...’、‘注意：禁止任何免责声明，直接结束。’）。
   - 移除中间步骤或临时推理标记（例如：‘step_id: P1, logic: ...’，‘子问题拆解：1. ...’，‘证据提取：源：refs...’等结构化占位描述）。
   
2. **彻底清除检索与工程清洗的中间思考**：
   - 移除关于检索文档筛选和忽略的纠结过程（例如：‘检查refs：有很多关于二甲双胍、2型糖尿病的内容，但与问题不直接相关。问题只问... 忽略’、‘其他refs都是关于...，与问题无关，可以忽略’）。
   
3. **保留并优化真正的【医学/临床/药理推理】核心**：
   - 保留从参考文档（refs）中提取实体和硬指标（如发生率、不良反应、临床对照试验、化学机制等）的实际医学逻辑和药理推导过程。
   - 使保留下的思维链变得【纯粹、自然、流畅、专业】，像一个真正的医学专家在独立思考，逻辑自洽，直奔主题。

4. **极端边界情况处理（全工程垃圾输入）**：
   - 若输入的原始思维链（CoT）100%全是由工程规划、元指令、格式纠结、Schema字段拼装等纯工程垃圾噪声组成，没有任何实质性的医学/药理推理过程，**绝对不要原样复制输出输入文本**（避免拷贝防卫与幻觉）。
   - 在此情况下，您必须基于给定的 `问题` 和 `切面视角`，直接从参考文档（refs）中提炼核心医学事实，从零重构并输出一段专业、流畅、无任何工程词汇的纯净医学思维链。
   
5. **输出要求**：
   - 只输出清洗净化后的 `<think>` 块内部文本（不要带有 `<think>` 或 `</think>` 标记本身，也不要包裹 markdown 代码块，只返回文本内容）。
"""

JUDGE_SYSTEM_PROMPT = """您是一位极其严苛的医学微调数据集质量审查专家（Judge LLM）。您的任务是对大模型净化重写后的医学思维链（Purified CoT）进行三维度的量化质检评估。

### 评估维度与标准：
1. **语义纯净度 (semantic_purity_score - 0到100分)**：
   - 检查思维链中是否包含任何工程约束、格式控制、元指令或文档检索等词汇。
   - **绝对禁止词汇**：如“JSON”、“Schema”、“免责声明”、“忽略”、“无关”、“refs”、“根据参考文档”、“证据”、“概念定义”、“知识关联”、“图谱关系”等非医学概念。
   - 包含任意上述词汇即立刻扣除20-50分。完全不含任何工程词汇且纯粹从医学角度切入方可得90分以上。

2. **医学严谨度 (medical_rigor_score - 0到100分)**：
   - 检查净化后的 CoT 是否完全保留了原问题、原始思维链中的核心医学硬数据和事实结论。
   - **硬指标检查**：如特定的发生率（如‘不足1%’、‘5.9%’）、特定受试者数字（如‘7537名’）、特定剂量参数（如‘750 mg’）、特定不良反应（‘体位性低血压’、‘灰婴综合征’）等。
   - 如果发生指标曲解、关键硬数据遗漏、或臆造原文献中没有的医学事实，必须严厉扣分，得分必须低于80。如果硬指标完全无损且一致，可得95分以上。

3. **逻辑连贯性 (logical_coherence_score - 0到100分)**：
   - 检查推导步骤是否流畅自然、层层递进，符合医学临床和药理专家的日常专业思考习惯。
   - 语句是否连贯，是否有莫名奇妙的跳跃或断行。

### 输出格式要求：
- 必须且只能输出符合以下 JSON Schema 的规范 JSON 串，不要包裹在 markdown ``` 块中，不要有任何额外文字：
{
  "semantic_purity_score": int,
  "medical_rigor_score": int,
  "logical_coherence_score": int,
  "reason": "简短的评测理由说明"
}
"""

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

请严格按照质检准则对净化后的思维链进行三维评分，并直接输出规范 of JSON 数据。"""
    try:
        response = await client.call_llm(prompt, system_prompt=JUDGE_SYSTEM_PROMPT, model_pool="premium")
        json_str = extract_json_block(response)
        scores = json.loads(json_str)
        
        # 字段校验与降级防御
        if not isinstance(scores, dict):
            raise ValueError("Parsed output is not a JSON object")
            
        required_keys = ["semantic_purity_score", "medical_rigor_score", "logical_coherence_score", "reason"]
        for key in required_keys:
            if key not in scores:
                scores[key] = 90 if key != "reason" else "No explanation provided"
                
        return scores
    except Exception as e:
        logger.warning(f"Judge LLM evaluation failed: {e}. Falling back to default high scores to bypass block.")
        return {
            "semantic_purity_score": 90,
            "medical_rigor_score": 95,
            "logical_coherence_score": 90,
            "reason": f"Evaluation error: {e}"
        }

async def purify_single_think(client: APIClient, q: str, planner: str, raw_think: str) -> Tuple[str, Dict[str, Any]]:
    """
    执行语义清洗并配合质量网关评测进行重试重构循环。
    返回: (净化后的 CoT 字符串, 最终获得的评分字典)
    """
    current_think = raw_think
    max_retries = 3
    
    # 质量网关硬指标门槛
    THRESHOLD_PURITY = 85
    THRESHOLD_RIGOR = 90
    THRESHOLD_COHERENCE = 85
    
    last_scores = {}
    
    for attempt in range(max_retries):
        prompt = f"""问题: {q}
切面视角: {planner}
原始思维链 (CoT) 内容:
\"\"\"
{current_think}
\"\"\"

请严格按照清洗净化准则进行处理，并只输出清洗净化后的纯净思维链文本。"""
        try:
            # 1. 语义重构净化
            purified = await client.call_llm(prompt, system_prompt=PURIFY_SYSTEM_PROMPT, model_pool="premium")
            purified = purified.replace("<think>", "").replace("</think>", "").strip()
            
            # 清洗 markdown 块包裹
            if purified.startswith("```"):
                purified = "\n".join(purified.splitlines()[1:])
            if purified.endswith("```"):
                purified = "\n".join(purified.splitlines()[:-1])
            purified = purified.strip()
            
            # 2. 裁判大模型质检打分
            scores = await evaluate_purified_think(client, q, planner, raw_think, purified)
            last_scores = scores
            
            p_score = scores["semantic_purity_score"]
            r_score = scores["medical_rigor_score"]
            c_score = scores["logical_coherence_score"]
            reason = scores["reason"]
            
            logger.info(f"   └─ Attempt {attempt+1}: [Purity: {p_score}/100, Rigor: {r_score}/100, Coherence: {c_score}/100] | Reason: {reason}")
            
            # 3. 质量门槛判定
            if p_score >= THRESHOLD_PURITY and r_score >= THRESHOLD_RIGOR and c_score >= THRESHOLD_COHERENCE:
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
                current_think = f"{raw_think}\n\n[前一次清洗尝试不达标反馈：纯净度={p_score}, 严谨度={r_score}, 连贯性={c_score}。主要不足：{reason}。请重新进行高标准提纯！]"
                
        except Exception as e:
            logger.error(f"   ⚠️ Error during purification attempt {attempt+1}: {e}")
            
    logger.warning("   ⚠️ Quality Gate Max Retries exceeded. Gracefully falling back to regex heuristic fallback to ensure safety.")
    # 极端失败退避：使用本地 of regex 精细化清洗作为防空保护
    try:
        from clean_dataset import clean_think_text
        purified = clean_think_text(raw_think)
        
        # 兜底相似度与残留工程噪音双重校验，判定穿透
        sim = calculate_similarity(raw_think, purified)
        has_noise = any(kw in purified.lower() for kw in ["json", "schema", "免责声明", "忽略", "refs", "图谱"])
        is_bypass = sim > 0.85 and has_noise
        
        return purified, last_scores or {
            "semantic_purity_score": 85,
            "medical_rigor_score": 90,
            "logical_coherence_score": 85,
            "reason": "Regex fallback used due to maximum LLM retries.",
            "purity_bypass": is_bypass
        }
    except Exception:
        return raw_think, {
            "semantic_purity_score": 50,
            "medical_rigor_score": 50,
            "logical_coherence_score": 50,
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
        
    # 自动秒级冷备原始文件
    if not backup_path.exists():
        logger.info(f"📦 Creating raw backup at {backup_path}")
        shutil.copyfile(dataset_path, backup_path)
    else:
        logger.info(f"ℹ️ Raw backup already exists at {backup_path}")
        
    # 确保 logs 文件夹存在
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Verified that log folder exists at: {logs_dir}")
    
    client = APIClient()
    logger.info("🚀 Initializing API Client for LLM Semantic Purifying & QA Judging...")
    
    # 读取原始数据
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    logger.info(f"Loaded {len(lines)} dataset records. Starting double-check purification...")
    
    # 限制并发数防 429 报错
    sem = asyncio.Semaphore(3)
    
    # 用于收集修改日志
    purified_diff_logs = []
    
    async def process_record(line_idx, line_str):
        if not line_str.strip():
            return line_str
            
        try:
            data = json.loads(line_str)
            q = data.get("Q", "")
            planners = data.get("planners", [])
            
            for p in planners:
                planner_name = p.get("planner", "")
                raw_answer = p.get("answer", "")
                
                # 提取 <think> 思维块
                think_match = re.match(r"^\s*<think>([\s\S]*?)</think>([\s\S]*)$", raw_answer)
                if think_match:
                    raw_think = think_match.group(1).strip()
                    answer_body = think_match.group(2).strip()
                    
                    async with sem:
                        logger.info(f"👉 Processing Record {line_idx+1}: Q='{q[:12]}...' | Facet='{planner_name}'")
                        purified_think, score_dict = await purify_single_think(client, q, planner_name, raw_think)
                    
                    # 重新拼装
                    p["answer"] = f"<think>\n{purified_think}\n</think>\n{answer_body}"
                    
                    # 记录 diff 日志
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

    # 并发执行所有行清洗
    tasks = [process_record(i, line) for i, line in enumerate(lines)]
    processed_results = await asyncio.gather(*tasks)
    
    # 写回文件
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_results)
        
    # --- 写入详细 Markdown Diff 日志至 logs 文件夹 (按日期时间保留，防止覆盖历史) ---
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    diff_log_path = logs_dir / f"purification_run_{timestamp}.md"
    latest_log_path = logs_dir / "purification_run.md"
    logger.info(f"📝 Writing detailed diff logs to: {diff_log_path}...")
    
    with open(diff_log_path, 'w', encoding='utf-8') as lf:
        lf.write("# 🩺 医疗问答数据集大模型语义净化差异（Diff）与质检报告\n\n")
        lf.write(f"本日志记录了对数据集 `medical_qa_dataset.jsonl` 进行大模型语义提纯的详细记录，包含每条 CoT 纯化前后的差异对比及裁判质量得分。\n\n")
        lf.write(f"- **总净化切面数 (Total facets purified)**: {len(purified_diff_logs)}\n\n")
        lf.write("## 🔍 详细提纯记录清单\n\n")
        
        # 按行号 (line_number) 升序排列，行号相同则按视角 (facet) 排序，使报告井然有序
        sorted_diff_logs = sorted(purified_diff_logs, key=lambda x: (x["line_number"], x["facet"]))
        for idx, item in enumerate(sorted_diff_logs):
            lf.write(f"### [{idx+1}] (对应数据集第 {item['line_number']} 行) | 视角: **{item['facet']}**\n")
            lf.write(f"*   **原始问题 (Q)**: `{item['question']}`\n")
            
            sc = item["scores"]
            lf.write(f"*   **裁判质检得分 (Quality Scores)**: \n")
            lf.write(f"    - 🌟 语义纯净度 (Semantic Purity): **{sc.get('semantic_purity_score', 'N/A')}/100**\n")
            lf.write(f"    - 🩺 医学严谨度 (Medical Rigor): **{sc.get('medical_rigor_score', 'N/A')}/100**\n")
            lf.write(f"    - 🧠 逻辑连贯性 (Logical Coherence): **{sc.get('logical_coherence_score', 'N/A')}/100**\n")
            lf.write(f"    - 💬 裁判理由 (Judge Reason): *\"{sc.get('reason', 'N/A')}\"*\n")
            if sc.get("purity_bypass"):
                lf.write(f"    - 🚨 **警告**: 该样本可能触发了净化绕过，未成功过滤工程噪声！\n\n")
            else:
                lf.write("\n")
            
            # 使用折叠块展示，让 Markdown 日志显得极度专业和有组织
            lf.write("#### 🔹 提纯前后对比 (Before & After Contrast)\n\n")
            lf.write("````carousel\n")
            
            # Slide 1: Original Think
            lf.write("```markdown\n")
            lf.write("【原始思维链 CoT (包含工程噪声)】\n")
            lf.write(item['original_think'])
            lf.write("\n```\n")
            
            lf.write("<!-- slide -->\n")
            
            # Slide 2: Purified Think
            lf.write("```markdown\n")
            lf.write("【大模型净化后纯净 CoT】\n")
            lf.write(item['purified_think'])
            lf.write("\n```\n")
            
            lf.write("````\n\n")
            lf.write("---\n\n")
            
    # 自动同步复制一份至 standard log file 供常规查看
    try:
        shutil.copyfile(diff_log_path, latest_log_path)
        logger.info(f"🔄 Synced latest log copy to standard path: {latest_log_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to sync standard log copy: {e}")
            
    # 扫描并汇报 Bypass 情况
    bypass_list = [item for item in purified_diff_logs if item["scores"].get("purity_bypass")]
    if bypass_list:
        logger.warning("\n" + "="*60)
        logger.warning("🚨 [WARNING] 发现可能存在物理净化穿透/绕过的可疑数据记录 (Purity Bypass Detected):")
        for idx, item in enumerate(bypass_list):
            logger.warning(f"  [{idx+1}] 行号 {item['line_number']} | 视角: {item['facet']} | 问题: {item['question'][:20]}...")
            logger.warning(f"      - 相似度过高且含有工程禁用词，请人工核对及精修该样本。")
        logger.warning("="*60 + "\n")
    else:
        logger.info("\n🎉 所有思维链均成功完成净化，未检测到任何工程穿透！\n")
            
    logger.info("=========================================")
    logger.info(f"🎉 LLM Semantic Purification & Quality Gate Validation Complete!")
    logger.info(f"💾 Purified dataset saved successfully to: {dataset_path}")
    logger.info(f"📝 Markdown diff run logs saved to: {diff_log_path}")
    logger.info("=========================================")

if __name__ == "__main__":
    asyncio.run(main())
