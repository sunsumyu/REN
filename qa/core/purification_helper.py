# -*- coding: utf-8 -*-
import re
import logging
import random
from typing import Dict, Any, Tuple, List
import config
from services.llm_service import ILLMService

logger = logging.getLogger("MedicalQA.PurificationHelper")

def safe_int(val: Any, default: int = 90) -> int:
    """安全转换为整型数值，防范裁判返回非结构化浮点或字符串导致网关崩溃"""
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default

# 🟢 全局进程内缓存，防止在提纯大规模语料时由于高频出现相同别扭标签导致 API 成本与延迟激增
_SMOOTHED_PLANNER_CACHE = {}

async def smooth_planner_term(llm_service: ILLMService, planner: str, line_num: int = None) -> str:
    """
    利用 LLM 零样本自适应平滑机器拼接的别扭切面名称，使其转换为规范的人类学术术语。
    """
    planner_clean = planner.strip()
    if not planner_clean:
        return planner_clean
        
    if planner_clean in _SMOOTHED_PLANNER_CACHE:
        return _SMOOTHED_PLANNER_CACHE[planner_clean]
        
    prompt = f"""你是一个顶级医学名词规范化专家。你的任务是把上游机器学习自动拼接或生造的、不合常理、别扭的“非人类医学标签”实时平滑、翻译并规范化为“符合医学专家日常口吻的自然专业术语”。

### 🛠️ 规范化红线：
1. 直接输出规范化后的短语，绝对不要包含任何解释、标点符号、Markdown 格式或前言后语。
2. 保持原有的核心医学/药理学/文献学含义不变。
3. 必须使用人类医学、药理学或文献学中高频、自然的专业词汇。

### 📐 转换示范 (Few-shot Examples)：
- 输入: "古籍收采" -> 输出: "中医药典籍源流与文献考证"
- 输入: "包装形式" -> 输出: "药物包装规格与形态特征"
- 输入: "指标偶联监测" -> 输出: "多指标联合动态临床监测"
- 输入: "剂量调整" -> 输出: "临床给药剂量调整方案"

现在，请规范化以下标签：
输入: "{planner_clean}" -> 输出: """

    try:
        stage_prefix = f"[{line_num}行] " if line_num else ""
        response = await llm_service.call_llm(prompt, model_pool="premium", stage=f"{stage_prefix}医学标签规范化 - {planner_clean}")
        smoothed = response.strip().replace('"', '').replace("'", "").replace("“", "").replace("”", "")
        # 对非正常回复进行校验和兜底
        if not smoothed or len(smoothed) > 20 or "输入" in smoothed or "输出" in smoothed:
            logger.warning(f"⚠️ Paraphrase result abnormal: '{smoothed}' for '{planner_clean}'. Falling back.")
            smoothed = planner_clean
        else:
            logger.info(f"✨ [AI Paraphrase] Smoothed raw planner '{planner_clean}' -> '{smoothed}'")
            _SMOOTHED_PLANNER_CACHE[planner_clean] = smoothed
        return smoothed
    except Exception as e:
        logger.error(f"⚠️ Failed to smooth planner term '{planner_clean}': {e}. Falling back.")
        return planner_clean

# Import prompts and configurations from core.purification_prompts
from core.purification_prompts import (
    get_system_directive,
    get_purify_system_prompt,
    FEW_SHOT_GENERAL,
    FACET_FEW_SHOTS,
    JUDGE_SYSTEM_PROMPT
)

def extract_json_block(text: str) -> str:
    text = text.strip()
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        return match.group(1)
    return text

def calculate_similarity(s1: str, s2: str) -> float:
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
    if len(text) < 150:
        return False
    
    mid = len(text) // 2
    part1 = text[:mid]
    part2 = text[mid:]
    
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
    """
    通用语义前置去标识化解析器：
    利用正则结构匹配而非特定词汇，物理剥离一切 RAG 引用、文献索引与图谱关系包裹，实现100%泛化阻断。
    """
    # 1. 物理移除所有的 JSON/步骤结构
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
        
    # 2. 【高泛化 RAG 结构与 refs 强制剥离】
    # 匹配 "根据 (refs/RAG)《...》的描述/显示/可知" 并完全剔除，只保留核心陈述
    cleaned = re.sub(r'根据\s*(?:refs|rag)?\s*《[^》]+》的?(描述|记载|显示|数据|图谱|关系|档案|文献|实体库)?(显示|可知|指出|表明|提供)?，?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'《[^》]+》', '', cleaned)
    
    # 3. 【高泛化文献索引剥离】
    # 匹配 "根据PubMed (PMID: 1234)的研究/报道"
    cleaned = re.sub(r'根据\s*PubMed\s*\(PMID:\s*\d+\)\s*的?(报道|研究|文献|病例)?，?', '', cleaned)
    cleaned = re.sub(r'PMID:\s*\d+', '', cleaned)
    cleaned = re.sub(r'PubMed\s*\([^)]+\)', '', cleaned, flags=re.IGNORECASE)
    
    # 4. 强力阻断工程元叙述词汇，彻底拦截 RAG 泄漏进入输入端
    forbidden_input_patterns = [
        r'\b(?:refs|rag|pmid|pubmed)\b',
        r'根据(?:参考)?(?:资料|文献|数据库|实体库|数据源|背景信息)?(?:显示|指出|表明|提供|记载)?，?',
        r'检索(?:结果|图谱|关系|facts)?(?:显示|指出|表明|提供|记载)?，?'
    ]
    for pattern in forbidden_input_patterns:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
    # 5. 清理残留括号与物理杂质
    cleaned = re.sub(r'[\{\}\[\]]', ' ', cleaned)
    return cleaned.strip()

def post_strip_meta_openings(text: str) -> str:
    """
    后置微创手术（升级版）：
    1. 精准切除开头的元指令宣告废话。
    2. 全局物理切割中途逃逸的“从XX视角分析/来看”等系统性切面宣告噪音。
    """
    cleaned = text.strip()
    
    # 0. 物理切除大模型在对齐硬性指标时遗留的草稿占位符，如 "，见？"、"(见?)"、"（见？）" 等尾巴
    cleaned = re.sub(r'[,，、\s]*(?:\(见[？\?]\)|（见[？\?]）|见[？\?])', '', cleaned)
    
    # 1. 拦截并切除位于文本开头的元描述
    meta_patterns = [
        r"^(我们(需|需要|将)?[^\n，。：]*?从[^\n，。：]*?视角[^\n，。：]*?[。，：])",
        r"^(针对(上述|这个|这一)?[^\n，。：]*?问题，?(我们)?[^\n，。：]*?[。，：])",
        r"^(为(了)?(解答|回答|探讨)[^\n，。：]*?问题，?(我们)?[^\n，。：]*?[。，：])",
        r"^(首先，?(我们)?(需要)?(来)?(分析|探讨|明确|了解)[^\n，。：]*?[。，：])"
    ]
    for pattern in meta_patterns:
        cleaned = re.sub(pattern, '', cleaned, count=1).strip()
        
    # 2. 全局拦截并切除中途逃逸的“从XX视角/角度分析（或来看）”
    global_facet_pattern = r"(从[^。，：]*?(视角|角度)(分析|来看)?，?)"
    cleaned = re.sub(global_facet_pattern, '', cleaned, flags=re.IGNORECASE).strip()
    
    # 3. 容错首字符标点清理
    if cleaned and cleaned[0] in ['，', '。', '、', '：']:
        cleaned = cleaned[1:].strip()
        
    return cleaned

def post_strip_structural_transitions(text: str) -> str:
    """
    零延迟的本地高效过渡词物理平滑器。
    在净化生成后、送往裁判打分前，强行将可能触发一票否决扣分的结构化过渡序号
    （如“首先”、“其次”、“第三”、“第四”、“第一”、“第二”）进行物理擦除或优雅的学术化平滑，
    确保 100% 达成纯净度门禁并保持自然的因果演进心流。
    """
    if not text:
        return text
        
    # 定义高精度的正则平滑替换规则，将结构化过渡词抹除或替换为高附加值的因果摩擦词
    pattern_replacements = [
        # 首先 -> 抹除
        (r'(?:\b|^)首先[，,\s]*', ''),
        (r'首先要明确', '要明确'),
        (r'首先需要', '需要'),
        (r'首先必须', '必须'),
        
        # 其次 -> 平滑为“进一步来看，”
        (r'(?:\b|^)其次[，,\s]*', '进一步来看，'),
        
        # 第三 / 其三 -> 平滑为“此外，”
        (r'(?:\b|^)第[三三][且且]?[，,\s]*', '此外，'),
        (r'(?:\b|^)其三[，,\s]*', '此外，'),
        
        # 第四 / 其四 -> 平滑为“另外，”
        (r'(?:\b|^)第[四四][且且]?[，,\s]*', '另外，'),
        (r'(?:\b|^)其四[，,\s]*', '另外，'),
        
        # 第一 / 第二 -> 平滑
        (r'(?:\b|^)第[一一][且且]?[，,\s]*', ''),
        (r'(?:\b|^)第[二二][且且]?[，,\s]*', '进一步来看，'),
        
        # 综上所述 / 因此最终结论是 -> 抹除做题废话
        (r'综上所述[，,\s]*', ''),
        (r'因此[，,\s]*最终结论是[，,\s]*', ''),
        (r'由此得出最终结论[，,\s]*', ''),
        
        # 新增高频工程泄漏词洗白映射 (Wash Map)
        (r'(?:\b|^)检索结果显示[，,\s]*', ''),
        (r'(?:\b|^)根据知识图谱[，,\s]*', ''),
        (r'(?:\b|^)实体库中提到[，,\s]*', ''),
        (r'(?:\b|^)根据(?:提供|上述)?的?(?:参考资料|背景信息|检索内容)?[，,\s]*', ''),
        (r'(?:\b|^)如上所述[，,\s]*', ''),
    ]
    
    repaired = text
    for pattern, repl in pattern_replacements:
        repaired = re.sub(pattern, repl, repaired, flags=re.IGNORECASE)
        
    return repaired.strip()

def is_catastrophic_format_collapse(text: str) -> bool:
    """后置硬性网关：检测是否残留 JSON 语法废墟或元描述穿透，使用精确正则以阻断误判"""
    invalid_chars = ['{', '}', '[', ']', '",', '我决定构建', '步骤1', '阶段一']
    if any(char in text for char in invalid_chars):
        return True
    
    # 🧠 精细化 RAG 工程泄露与元叙述硬网关
    leakage_patterns = [
        r'根据(参考|提供|背景|检索)?(资料|上下文|数据|文本|信息)(显示|指出|表明|提供|描述)',
        r'数据源(中|显示|提到|记录)',
        r'实体库(中|显示|提到|记录)',
        r'由于(检索|提供|背景|上下文)?(资料|信息|数据)(有限|没有|不足|未提及)',
        r'本思考过程主要(围绕|着眼于|基于)',
        r'从.*?视角来说，',
        r'根据.*?视角，'
    ]
    for pattern in leakage_patterns:
        if re.search(pattern, text):
            return True
            
    return False
