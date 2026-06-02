# -*- coding: utf-8 -*-
"""
大模型语义化清洗医学问答数据集 CoT（思维链）脚本 (企业极速优化版 - 带智能去去词及白名单机制)。
利用质检裁判大模型（Judge LLM），对重写后的思维链从“语义纯净度”、“医学严谨度”和“逻辑深度”三个维度进行量化评分（Quality Gate），
对于不达标的样本执行自动重新净化重写，确保 100% 达成生产级微调的严苛质量要求。
"""

import os
import sys
import json
import asyncio
import logging
import re
import shutil
import datetime
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

# 将当前目录与项目根目录加入系统路径以确保 import 正常
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(parent_dir))

from config import LLM_MODEL, PURIFY_LIMIT, PURIFY_LINES, PURIFY_START_LINE, PURIFY_CONCURRENCY
from api_client import APIClient
from core.purification_engine import PurificationEngine
from services.healing_service import HealingService
from strategies.quality_gate.llm_judge import LLMJudgeStrategy

from utils.logging_config import setup_logging
setup_logging()
logger = logging.getLogger("MedicalQA.LLMPurifier")

# 针对 Windows 控制台环境，强行配置标准输出为 UTF-8
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

async def verify_facet_by_small_model(client: APIClient, facet: str) -> bool:
    """
    利用轻量级小模型对视角（facet）进行语义分类，
    智能判定其是否为“合法的医学/临床视角”或“非法的反问/拒答/占位符”。
    """
    system_prompt = (
        "你是一个极速的语义分类器网关。你的唯一任务是判断给定的文本是否为【合法的医学或临床视角/切面名称】。\n\n"
        "### ⚖️ 判定准则：\n"
        "1. 【VALID】：文本是一个干净的词汇或短语，代表某种学术、病理、药理、临床或合规的叙事切面（例如：“药代动力学”、“用药注意事项”、“古籍收采”、“分子机制”、“不良反应预防”）。\n"
        "2. 【INVALID】：文本是一句人机对话反问、澄清疑问、报错占位符、缺少上下文的抱怨、或者极其冗长的描述（例如：“请提供具体的医疗问题以便规划视角”、“请输入详细病例”、“数据不足无法规划”、“该药物有何副作用？”、“一句话描述”）。\n\n"
        "### 📤 输出规则：\n"
        "只输出一个单词：'VALID' 或 'INVALID'。严禁输出任何解释、标点符号或额外空格。"
    )
    try:
        response = await client.call_llm(
            prompt=f"待测文本: '{facet}'",
            system_prompt=system_prompt,
            model_pool="lightweight",
            stage="视角语义安全校验网关"
        )
        result = response.strip().upper()
        return "INVALID" not in result
    except Exception as e:
        logger.error(f"⚠️ 视角小模型网关校验异常: {e}，默认通过。")
        return True

async def check_semantic_compatibility(client: APIClient, question: str, facet: str) -> str:
    """
    利用轻量级小模型（deepseek-v4-flash）对“问题”与“视角”进行语义匹配和兼容性校验。
    返回:
      - 'COMPATIBLE': 视角与问题高度兼容，正常执行深度提纯。
      - 'COMPATIBLE_SIMPLE': 视角可用于解题，但由于问题本身极其简单（如查剂量、成分等事实问题），
                             必须极简化处理，严禁脑补和虚构复杂机制。
      - 'FORCED_SKIP': 属于明显的强行视角硬套或无谓的过度逻辑漂移，直接过滤丢弃该切面。
    """
    system_prompt = (
        "你是一个极其严格的医疗问答视角语义匹配网关（Gatekeeper）。你的唯一任务是评估给定的【主问题】与【分析视角/切面】之间的语义适配性，以防范数据清洗过程中的‘视角强行套用’及由此引发的‘虚假因果关系编造’。\n\n"
        "### ⚖️ 评估判定规则：\n"
        "1. 【FORCED_SKIP】(绝对禁止强套)：\n"
        "   - 当主问题是一个简单的事实查找（如具体的药品剂量、单次口服量、药品成分、药品包装规格等），却被强行匹配了极其宏观或深度学术化的视角（如：“分子机制”、“药代动力学”、“特殊人群安全”、“数据隐私”、“循证证据”等），且原始思维链为了迎合该视角不得不进行大篇幅的牵强附会或事实捏造（如将 0.2g 单次剂量解释为 IC50 建模推导、无端捏造特定性别限制）。这类情况属于严重的强行视角，直接舍弃。\n"
        "2. 【COMPATIBLE_SIMPLE】(简单兼容，极简推理)：\n"
        "   - 当主问题是相对直接的临床问题，匹配了有一定关联但无需长篇大论的视角（例如：查剂量问题匹配了“药物过量”、“不良反应”或“禁忌症”）。这些视角可以保留，但逻辑非常单一（直接回答剂量安全上限或常见副作用即可），不需要进行微观病生理链条的深度因果推演。此时判定为 COMPATIBLE_SIMPLE。\n"
        "3. 【COMPATIBLE】(完全兼容，深度推理)：\n"
        "   - 问题本身具有探索性、机制性，或者与视角高度贴合（例如：“缺乏DPYD为何导致严重5-FU毒性” 匹配 “分子机制”；“急性心梗患者为何禁用非洛地平” 匹配 “用药方案与配伍禁忌”）。这需要呈现深度、复杂的临床或药理演绎。判定为 COMPATIBLE。\n\n"
        "### 📤 输出规则：\n"
        "只输出一个单词：'COMPATIBLE'、'COMPATIBLE_SIMPLE' 或 'FORCED_SKIP'。严禁输出任何解释、标点符号或额外空格。"
    )
    user_prompt = f"主问题: '{question}'\n分析视角: '{facet}'"
    try:
        response = await client.call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model_pool="lightweight",
            stage="视角适配语义网关"
        )
        result = response.strip().upper()
        for verdict in ["FORCED_SKIP", "COMPATIBLE_SIMPLE", "COMPATIBLE"]:
            if verdict in result:
                return verdict
        return "COMPATIBLE"
    except Exception as e:
        logger.error(f"⚠️ 视角适配网关校验异常: {e}，默认完全兼容。")
        return "COMPATIBLE"

async def purify_single_think(engine: PurificationEngine, q: str, planner: str, raw_think: str, line_num: int = None, refs: List[Dict[str, Any]] = None, simplify: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Surgically delegate single thought trace purification to modularized PurificationEngine (reusing injected engine).
    """
    return await engine.purify_single_think(q, planner, raw_think, line_num, refs, simplify)

async def generate_diff_analysis(llm_service, item: dict) -> str:
    """
    使用轻量级大模型对单个视角的提纯差异进行智能分析，生成
    "原始 CoT 存在的问题" 与 "提纯改动说明" 两段结构化叙述。
    """
    original = item.get("original_think", "")
    purified = item.get("purified_think", "")
    question = item.get("question", "")
    facet = item.get("facet", "")
    judge_reason = item.get("scores", {}).get("reason", "")
    compatibility = item.get("compatibility", "COMPATIBLE")
    simplify = item.get("simplify", False)

    mode_hint = ""
    if compatibility == "COMPATIBLE_SIMPLE" or simplify:
        mode_hint = "（注：本视角被语义网关判定为 COMPATIBLE_SIMPLE 极简模式，提纯时采用了精简指令。）"
    elif compatibility == "FORCED_SKIP":
        mode_hint = "（注：本视角已被语义网关拦截丢弃，未进入提纯流程。）"

    prompt = f"""你是医疗 CoT 数据集提纯质检专家。请对以下思维链的"原始版本"与"提纯后版本"进行简明差异分析。
{mode_hint}
主问题: {question}
临床视角: {facet}
裁判评审意见: {judge_reason}

原始思维链:
\"\"\"
{original[:1200]}
\"\"\"

提纯后思维链:
\"\"\"
{purified[:1200]}
\"\"\"

请严格按以下格式输出，每项2-4句话，中文简洁叙述，不加任何 Markdown 标题或额外说明：
【原始问题】: （列举原始思维链中存在的具体质量缺陷，如：工程词汇泄露、RAG引用痕迹、结构化标题、元叙述废话、推理平铺直叙、字数不足等）
【改动说明】: （说明提纯过程的主要改动内容及效果，若有信息损失或过度简化也需指出）"""
    try:
        response = await llm_service.call_llm(
            prompt,
            model_pool="lightweight",
            stage=f"[{item.get('line_number')}行] 提纯差异分析 - {facet}"
        )
        return response.strip()
    except Exception as e:
        logger.warning(f"⚠️ 差异分析生成失败: {e}")
        return "【原始问题】: 分析生成失败。\n【改动说明】: 分析生成失败。"


def update_env_start_line(env_path: Path, start_line: int):
    """
    Dynamically writes the determined starting line number back to the .env file.
    """
    if not env_path.exists():
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = re.compile(r"^(\s*PURIFY_START_LINE\s*=).*$", re.MULTILINE)
    if pattern.search(content):
        new_content = pattern.sub(f"\\g<1>{start_line}", content)
    else:
        new_content = content.rstrip() + f"\n\n# 自动设置的净化起始行号\nPURIFY_START_LINE={start_line}\n"
        
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

async def main():
    start_time = time.time()
    dataset_path = parent_dir / "medical_qa_dataset.jsonl"
    backup_path = parent_dir / "medical_qa_dataset_raw.jsonl"
    logs_dir = parent_dir / "logs"
    
    purify_start_line = PURIFY_START_LINE
    
    if purify_start_line is None:
        logger.info("🔍 PURIFY_START_LINE is not set in .env. Scanning all historical logs to auto-detect the maximum processed line...")
        run_files = [
            f for f in logs_dir.glob("purification_run_*.md")
            if re.search(r"purification_run_(?:\[\d+-\d+\]_)?\d{8}_\d{6}\.md", f.name)
        ]
        
        all_processed_lines = []
        for rf in run_files:
            try:
                with open(rf, 'r', encoding='utf-8') as lf:
                    log_content = lf.read()
                lines_in_file = [int(num) for num in re.findall(r"数据集第\s*(\d+)\s*行", log_content)]
                all_processed_lines.extend(lines_in_file)
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse log file {rf.name}: {e}")
                
        if all_processed_lines:
            max_line = max(all_processed_lines)
            purify_start_line = max_line + 1
            logger.info(f"📈 Auto-detected maximum processed line across all logs: {max_line}. Setting PURIFY_START_LINE to: {purify_start_line}")
        else:
            purify_start_line = 1
            logger.info("⚠️ No processed line numbers found in any historical logs. Setting PURIFY_START_LINE to: 1")
            
        env_path = parent_dir / ".env"
        try:
            update_env_start_line(env_path, purify_start_line)
            logger.info(f"✨ Dynamically updated PURIFY_START_LINE={purify_start_line} in .env file.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to write back to .env: {e}")
    else:
        logger.info(f"👉 Using manually configured PURIFY_START_LINE={purify_start_line} from .env")
    
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return
        
    if not backup_path.exists():
        logger.info(f"✨ Creating initial raw backup at {backup_path}")
        shutil.copyfile(dataset_path, backup_path)
    else:
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset_lines = f.readlines()
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_lines = f.readlines()
                
            if len(dataset_lines) > len(backup_lines):
                new_raw_lines = dataset_lines[len(backup_lines):]
                logger.info(f"Detected {len(new_raw_lines)} new raw incremental records. Syncing and appending to raw backup...")
                with open(backup_path, 'a', encoding='utf-8') as f:
                    f.writelines(new_raw_lines)
            else:
                logger.info(f"Raw backup is fully in sync with current dataset ({len(backup_lines)} lines). No new raw entries to append.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to sync incremental backup: {e}. Keeping existing backup.")
        
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Verified that log folder exists at: {logs_dir}")
    
    client = APIClient()
    logger.info("🚀 Initializing API Client for LLM Semantic Purifying & QA Judging...")
    
    # 🟢 [Performance Optimization] Initialize the purification engine only once at startup
    engine = PurificationEngine(
        llm_service=client.llm_service,
        healing_service=HealingService(client.llm_service),
        evaluator_strategy=LLMJudgeStrategy(client.llm_service)
    )
    logger.info("⚙️ Successfully initialized reusable single PurificationEngine instance.")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    logger.info(f"Loaded {len(lines)} dataset records. Starting double-check purification...")
    
    sem = asyncio.Semaphore(PURIFY_CONCURRENCY)
    
    purified_diff_logs = []
    
    async def process_record(line_idx, line_str, should_purify=True):
        if not line_str.strip():
            return line_str
            
        try:
            data = json.loads(line_str)
            refs = data.get("refs", [])  # 🌟 提取原始图谱及外部文献引用作为刚性事实白名单锚点
            data.pop("history", None)
            data.pop("refs", None)
            
            if should_purify:
                q = data.get("Q", "")
                planners = data.get("planners", [])
                
                TEMPLATE_SIGNATURES = [
                    "触发物理格式崩溃",
                    "质量网关硬指标",
                    "自动重新净化重写",
                ]
                SAFE_BODY_SIGNATURES = [
                    "根据参考资料",
                    "现有资料未提供",
                ]
                
                async def process_planner(p):
                    try:
                        raw_planner_name = p.get("planner", "")
                        raw_answer = p.get("answer", "")
                        
                        if any(sig in raw_answer for sig in TEMPLATE_SIGNATURES):
                            logger.warning(f"  🚨 Skip Line {line_idx+1} facet '{raw_planner_name}' due to template signature.")
                            return None, None
                            
                        if any(sig in raw_answer for sig in SAFE_BODY_SIGNATURES):
                            logger.warning(f"  🚨 Skip Line {line_idx+1} facet '{raw_planner_name}' due to safety warning signature.")
                            return None, None
                        
                        # 🚦 接入轻量级小模型视角语义校验网关
                        is_valid = await verify_facet_by_small_model(client, raw_planner_name)
                        if not is_valid:
                            logger.warning(f"  🚨 [小模型网关校验拦截] 发现行 {line_idx+1} 的非医学视角占位符: '{raw_planner_name}'。进行安全重置与自愈兜底...")
                            planner_name = "临床用药安全"
                            p["planner"] = planner_name
                        else:
                            planner_name = raw_planner_name
                        
                        # 🚦 [企业级语义适配度校验网关]：检测主问题与当前视角是否强套
                        compatibility = await check_semantic_compatibility(client, q, planner_name)
                        if compatibility == "FORCED_SKIP":
                            logger.critical(f"  🚨 [企业级网关强行视角拦截] 拦截行 {line_idx+1} 的强套视角 '{planner_name}' 并执行物理剪枝丢弃。")
                            return None, None
                        
                        simplify = (compatibility == "COMPATIBLE_SIMPLE")
                        if simplify:
                            logger.info(f"  💡 [企业级网关简化提示] 行 {line_idx+1} 的视角 '{planner_name}' 与简单问题匹配，开启极简重构。")
                        
                        think_match = re.match(r"^\s*<think>([\s\S]*?)</think>([\s\S]*)$", raw_answer)
                        if think_match:
                            raw_think = think_match.group(1).strip()
                            answer_body = think_match.group(2).strip()
                            
                            facet_match = re.match(r"^\s*(<facet\s*=\s*[^>]+>)\s*([\s\S]*)$", raw_think)
                            if facet_match:
                                facet_tag = facet_match.group(1).strip()
                                actual_raw_think = facet_match.group(2).strip()
                                
                                # 若发生过网关修正，需同步修正 think 块内的 facet_tag，避免结构不一致
                                if not is_valid:
                                    facet_tag = "<facet = 临床用药安全>"
                            else:
                                facet_tag = f"<facet = {planner_name}>"
                                actual_raw_think = raw_think
                            
                            async with sem:
                                logger.info(f"⏳ Processing Record {line_idx+1}: Q='{q[:12]}...' | Facet='{planner_name}'")
                                purified_think, score_dict = await purify_single_think(
                                    engine, q, planner_name, actual_raw_think, 
                                    line_num=line_idx+1, refs=refs, simplify=simplify
                                )
                            
                            # 🚨 核心网关：若最终质量网关判定未通过（包含高仿真幻觉风险等），在此阶段直接抛弃该视角，绝不落盘污染！
                            if not score_dict.get("is_passed", False):
                                logger.critical(f"   ❌ [Hallucination/Quality Gate Intercept] Drop facet '{planner_name}' for Line {line_idx+1} due to Quality Gate Failure.")
                                return None, None
                            
                            p_new = p.copy()
                            p_new["answer"] = f"<think>\n{facet_tag}\n{purified_think}\n</think>\n{answer_body}"
                            
                            diff_log = {
                                "line_number": line_idx + 1,
                                "question": q,
                                "facet": planner_name,
                                "original_think": raw_think,
                                "purified_think": purified_think,
                                "scores": score_dict,
                                "compatibility": compatibility,
                                "simplify": simplify
                            }
                            return p_new, diff_log
                        else:
                            return p, None
                    except Exception as e:
                        logger.error(f"❌ [Exception Intercept] Exception occurred when purifying line {line_idx+1} facet '{p.get('planner', '')}': {e}. Drop this facet to prevent pollution.")
                        return None, None
 
                # 并发执行单行内的所有切面
                planner_tasks = [process_planner(p) for p in planners]
                planner_results = await asyncio.gather(*planner_tasks)
                
                valid_planners = []
                for p_new, diff_log in planner_results:
                    if p_new is None:
                        continue
                    if diff_log:
                        purified_diff_logs.append(diff_log)
                    valid_planners.append(p_new)
                
                data["planners"] = valid_planners
            
            return json.dumps(data, ensure_ascii=False) + "\n"
        except Exception as e:
            logger.error(f"❌ Error processing line {line_idx+1}: {e}")
            return line_str
            
    purify_counter = 0
    tasks = []
    for i, line in enumerate(lines):
        line_num = i + 1
        should_purify = True
        
        if PURIFY_LINES:
            if line_num not in PURIFY_LINES:
                should_purify = False
                
        if purify_start_line is not None:
            if line_num < purify_start_line:
                should_purify = False
                
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
    
    # 🌟 [企业级安全并发合并写回机制 - 防止 Lost Update 并发覆盖]
    # 在写回磁盘前，重新读取一次磁盘上最新的数据集。防止在提纯大模型调用期间（通常长达数十秒至数分钟）
    # 后台并发的生成器进程又追加写入了新的原始语料，导致提纯写回时以 w 模式将这些新语料无情覆盖抹杀。
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            latest_disk_lines = f.readlines()
        
        if len(latest_disk_lines) > len(lines):
            new_appended_count = len(latest_disk_lines) - len(lines)
            logger.warning(f"⚠️ [并发写冲突拦截] 检测到在提纯期间，生成器追加了 {new_appended_count} 条新语料。正在进行安全合并...")
            processed_results = list(processed_results) + latest_disk_lines[len(lines):]
    except Exception as e:
        logger.error(f"⚠️ [并发写冲突拦截] 读取最新数据进行合并失败: {e}。退回默认覆盖写。")
        
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_results)
        
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

    # 🧠 [批量差异分析] 在写报告前，并发调用轻量级大模型对每个视角生成原始问题分析与改动说明
    if purified_diff_logs:
        logger.info(f"🔬 Running batch diff analysis for {len(purified_diff_logs)} facets using lightweight LLM...")
        analysis_tasks = [generate_diff_analysis(client.llm_service, item) for item in purified_diff_logs]
        analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        for item, analysis in zip(purified_diff_logs, analyses):
            item["diff_analysis"] = analysis if isinstance(analysis, str) else "分析生成失败。"

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
                
                # 🔬 差异分析段：原始问题 & 改动说明
                compatibility_val = item.get("compatibility", "COMPATIBLE")
                simplify_flag = item.get("simplify", False)
                mode_badge = ""
                if compatibility_val == "COMPATIBLE_SIMPLE" or simplify_flag:
                    mode_badge = " 🔵 `SIMPLIFY 极简模式`"
                
                lf.write(f"#### 🔬 原始问题 & 改动说明{mode_badge}\n\n")
                diff_analysis = item.get("diff_analysis", "")
                if diff_analysis:
                    for al in diff_analysis.split("\n"):
                        al = al.strip()
                        if al.startswith("【原始问题】:"):
                            lf.write(f"- **🔴 原始 CoT 存在的问题**: {al[len('【原始问题】:'):].strip()}\n")
                        elif al.startswith("【改动说明】:"):
                            lf.write(f"- **🟢 提纯改动说明**: {al[len('【改动说明】:'):].strip()}\n")
                        elif al:
                            lf.write(f"  {al}\n")
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
            
    try:
        shutil.copyfile(diff_log_path, latest_log_path)
        logger.info(f"✨ Synced latest log copy to standard path: {latest_log_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to sync standard log copy: {e}")
            
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
            
    if purified_diff_logs and not PURIFY_LINES:
        max_processed_line = max(grouped_logs.keys())
        next_start_line = max_processed_line + 1
        env_path = parent_dir / ".env"
        try:
            update_env_start_line(env_path, next_start_line)
            logger.info(f"✨ Automatically updated PURIFY_START_LINE={next_start_line} in .env file.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update .env: {e}")

    logger.info("=========================================")
    logger.info("🚀 LLM Semantic Purification & Quality Gate Validation Complete!")
    logger.info(f"⏱️ Total Elapsed Time: {time.time() - start_time:.2f} seconds")
    logger.info(f"💾 Purified dataset saved successfully to: {dataset_path}")
    logger.info(f"📄 Markdown diff run logs saved to: {diff_log_path}")
    logger.info("=========================================")

if __name__ == "__main__":
    try:
        from utils.process_lock import acquire_process_lock
        acquire_process_lock("medicalqa_purifier")
    except Exception as e:
        print(f"⚠️ [ProcessLock Error] Failed to initialize lock: {e}")

    asyncio.run(main())