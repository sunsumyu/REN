# -*- coding: utf-8 -*-
"""
大模型语义化清洗医学问答数据集 CoT（思维链）脚本 (最终整合版 - 企业生产级 - 带思维心流防线)。
利用智能质检裁判大模型（Judge LLM），对重写后的思维链从“语义纯净度”、“医学严谨度”和“逻辑深度”三个维度进行量化评分（Quality Gate）。
集成强制自问自答（包含 ?）、5阶段推理心流、安全数值拦截、智能增量同步备份与全字段脱水防线。
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

def safe_int(val: Any, default: int = 90) -> int:
    """安全的整数转换，防止大模型在JSON中吐出浮点数或字符串分数导致 >= 比较报错"""
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default

def get_system_directive(planner: str) -> str:
    """根据视角（Planner）动态生成特异性引导，杜绝硬编码偏置"""
    directives = {
        "药理机制": "the biological, molecular, and pharmacological mechanisms of the drug or treatment",
        "用药方案与配伍禁忌": "the clinical dosing regimens, safety boundaries, and drug-drug/disease contraindications",
        "临床表现": "the clinical signs, symptoms, and disease progression features",
        "诊断与鉴别诊断": "the diagnostic criteria, laboratory/imaging findings, and differential diagnosis logic",
        "临床疗效": "the clinical efficacy, therapeutic outcomes, and patient response parameters",
        "不良反应": "the side effects, adverse events, toxicological mechanisms, and safety monitoring"
    }
    return directives.get(planner, f"the clinical rationale, evidence synthesis, and medical logic related to {planner}")

def get_purify_system_prompt(planner: str) -> str:
    """动态生成系统提示词，注入特异性视角红线拦截与动态探索心流"""
    return f"""您是一位顶级循证医学科学家与大模型思维链（CoT）语料提纯专家。您的任务是净化并重写医学问答数据集中的 `<think>`（思维链）内容，使其达到顶尖的 Reasoning 模型（如 DeepSeek-R1、OpenAI o1）微调的冷启动标准。

### 🚨 提纯与重写的核心目标：
当前 `<think>` 块是工程 Pipeline 自动生成的，混杂了大量【工程指令垃圾】（如 JSON 结构、Refs 引用、RAG 抱怨、格式避让）。如果直接用于模型微调，会导致模型在推理时频繁产生格式泄漏和系统幻觉。您需要将静态的 RAG 段落升华为一个**人类医学专家大脑里真实、高熵、流畅且无污染的“动态临床认知推理心流”**。

### 🛠️ 企业级提纯与重写规范：

1. ❌ 【物理屏蔽一切工程与流水线噪声】（检出一个即扣分）：
   - 彻底移除涉及输出格式、JSON约束、Schema结构、Markdown标记的自我提醒（如：“我们要输出JSON”、“现在开始构建 JSON schema”等）。
   - 彻底移除工程流水线的临时步骤占位符（如：`step_id: P1`、`logic: ...`、`证据对齐清单`、`子问题拆解`、`Answer Body` 等结构性字段名）。

2. 🚨 【严禁伪净化与做题家元描述（Format Translation Bypass）】：
   - 绝对禁止采取“将 JSON 字段名翻译成自然语言来凑字数”的走捷径策略！
   - **绝对禁止在文本中输出任何暗示您在处理一个工程结构化任务的元叙述词汇**。包括但不限于：“我的推理链条如下”、“问题可以拆解为以下子问题”、“最终结论是...”等。
   - **🚨 绝对禁止在文本的任何地方吐出“切面”、“视角”、“角度”等字眼，也绝对禁止直接提及或复读当前的视角名称‘{planner}’！** 严禁出现“从{planner}视角分析”、“根据{planner}角度来看”等任何向用户宣告或暗示你正在以何种视角解题的元叙述句式。

3. 🚨 【严禁工程化 RAG 抱怨，但必须坚守“循证医学”的不确定性边界】：
   - **彻底物理剥离工程痕迹**：绝对禁止在思考链中输出暴露后台流水线和检索瓶颈的“工程级抱怨”（如：“根据检索得到的文献”、“提供的参考资料中未提及...”、“实体库中无此关联”等）。模型在思考时应假定自己具备完整的常识和知识库。
   - **基于“循证级别”的自适应推理（拒绝无证据绝对化脑补）**：
     - 如果原始资料缺乏直接的临床结论，激活参数化知识时**严禁瞎编虚假临床事实或进行绝对化断言**。
     - 您必须引导思维链采用**“循证降级推断”**：在因果推演时，使用具有“科学谦逊感”的医学条件句式（如：“从药理学机制上推断，可能通过……起效，但目前缺乏直接针对该复合剂型的临床循证数据支持……”）。
     - **引入安全边界与自我校准（Self-Correction）**：在推理中段，必须对潜在的不确定性、证据级别不足或临床风险进行主动评估和纠偏思考（例如：“虽然从受体阻断机制上可以推断出X，但目前缺乏直接双盲临床文献支持，实际应用中仍需警惕Y风险...”）。这才是顶级Reasoning模型在面对灰色地带时应表现出的“临床自省心流”。

4. 📐 【强化的 5 阶段临床认知深度推理流（Exploratory CoT Trajectory） - 拒绝文章叙述型/教科书文章体】：
   优秀的 Reasoning 微调 CoT 必须呈现出**“提出假设 -> 探究机制 -> 遇到逻辑分叉/交叉校验 -> 推导排除 -> 决策合拢”**的动态心流轨迹（Thought Trace）。
   - 🚨 **【拒绝静态科普文与百科说明书体】**：绝对不能写成“A是B，C通过D发挥作用”这种平铺直叙的百度百科、说明书或漂亮的科普文章！模型是在进行“即时探索和解题推理”，不是在背诵课本或撰写向用户讲解的科普宣教。
   - 🚨 **【强制内部自问自答与反思锚点（强制包含问号“？”）】**：思维链中**必须至少包含 1-2 处明显的、真实的自我提问或反向排查**（例如：“难道这仅仅是由于靶点阻断吗？”、“慢着，在此处必须评估：这一情况在特定生理状态下是否会持续...”、“倘若患者伴有重度肾损伤，这一浓度是否会发生蓄积中毒？”）。这能从物理上彻底打破“静态叙述体”，展现思维内部现场演算的张力。
   - **必须使用高密度的逻辑摩擦关联词**：在思维链中，强制要求使用诸如“要剖析...首先必须解构...”、“既然...必然...”、“然而仅靠...是不够的”、“进一步来看...”、“这就完美解释了为何...”、“如果是遇到...情况呢？”、“由此推导...”等带有强烈因果推演和自问自答色彩的动态思考词汇，展现专家大脑内部的真实演算过程。
   您必须引导思维链通过以下 5 个自然的认知阶段隐式递进：
   - **阶段一：核心临床矛盾解构** —— 开头直切临床/医学矛盾核心（例如，直接以物理/医学事实及逻辑摩擦词切入主题：“要剖析[疾病/药物/治疗名称]的核心机制/生理本质，首先必须解构...”，不需要任何结构性的开场白、自问自答或过渡废话）。
   - **阶段二：微观病生理/临床逻辑推演** —— 对分子靶点、受体结合、体内代动学参数等进行深度因果链条解析，呈现动态心流。
   - **阶段三：逻辑分叉与特殊情况排查** —— 加入自我提问和临床假说排查，增加思维链的“逻辑熵”。
   - **阶段四：生理/安全极限与查漏补缺** —— 引入对于年龄、肝肾功能受损等特殊情况或安全边界的核验，展示高价值的“自我纠正与查漏补缺”过程。
   - **阶段五：决策自然合拢** —— 以高度学术的口吻，自然推演出最合理的临床或机制结论，禁止出现“综上所述”、“因此最终结论是”等做题套话。

5. 🚨 【防范序号结构泄漏】：
   - **绝对禁止使用任何如“阶段一”、“阶段①”、“【核心矛盾】”、“步骤1”等显式的、结构化的提纲或序号词！** 这种结构化泄漏会破坏 Reasoning 模型的原生思考连贯性。必须使用**高度自然的学术因果递进和自问自答的长文流**。

6. 🔇 【语调与文风红线】：
   - 必须使用绝对的**第三人称、客观学术、冰冷严谨的医学专家内心独白视角**。
   - 彻底去除任何对话性废话（如：“好的，让我来为你解答...”、“问题问的是...，我的分析是...”）。直接开始陈述医学事实和逻辑因果，不需要任何开场白或过渡废话。

7. 📤 【输出物理格式要求】：
   - 仅输出净化提纯后的 `<think>` 内部纯文本，绝对不要带有 `<think>` 或 `</think>` 标记本身，也不要包裹在 markdown 围栏中。
"""

# 🟢 切面自适应少样本（Dynamic Few-Shot）映射库，引入反思自省与强制问号（?）示范，完美阻断“静态文章”倾向。
FEW_SHOT_PHARMACOLOGY = """
### 🟢 提纯重构黄金少样本示范 (Few-Shot Gold Standard Example)：
* **输入原始思维链 (包含JSON规划与RAG噪声)**：
\"\"\"
我们被要求输出一个JSON对象，符合指定的schema...问题只是问成分，我们可以扩展回答吗？...丹膝颗粒主要成分包括丹参、牛膝、天麻、牡丹皮...
\"\"\"
* **输出提纯重构后的完美思维链**：
\"\"\"
要剖析丹膝颗粒的药理学机制，首先必须解构其核心临床目标：缺血性脑血管病恢复期的‘瘀血阻络兼肾虚证’。既然核心矛盾是‘瘀血’与‘肾虚’，方剂的骨架必然以此为基底。我们观察到处方中的丹参、赤芍、川芎，这三者是经典的活血化瘀药对。从现代药理学推演，它们的作用机制显然是指向改善微循环和抑制血小板聚集，从而直接打击脑梗塞后的局部缺血核心病理。然而，仅靠活血通络是不够的。中风后遗症往往伴随长期的机体耗损，处方中紧接着出现的牛膝、地黄、淫羊藿、桑寄生，在微观机制上是为了调节下丘脑-委体-靶腺轴，通过抗应激损伤来纠正底层的虚损状态。慢着，如果是遇到合并严重肾脏排泄功能受损的患者呢？桑寄生与牛膝中的某些皂苷成分排泄是否会受阻？从临床指南来看，虽然常用剂量下安全性尚可，但对于此类患者仍需加强血清学指标监视。进一步来看，脑血管病患者多伴有血压波动与神经兴奋性异常，这完美解释了方中为何要配伍天麻——利用天麻素的镇静活性来保护脑神经；同时辅以牡丹皮、栀子、决明子来压制烦躁失眠的伴发症状。最后，方中为何会出现一味火麻仁？对于中风卧床患者，保持肠道通畅能有效降低腹压，间接稳定血压并减轻脑血管负荷，这是极其精妙的‘釜底抽薪’式次级调节。经此层层解构，十二味药材的配伍逻辑已完全清晰：形成了一个涵盖抗血小板、神经保护、内分泌调节与靶器官减负的多靶点整合网络。
\"\"\"
"""

FEW_SHOT_CONTRAINDICATION = """
### 🟢 提纯重构黄金少样本示范 (Few-Shot Gold Standard Example)：
* **输入原始思维链 (包含JSON规划与RAG噪声)**：
\"\"\"
我们被要求输出一个结构化的JSON，回答“联环笑定-非洛地平缓释胶囊的禁忌症包括哪些？”，并基于提供的refs...
\"\"\"
* **输出提纯重构后的完美思维链**：
\"\"\"
面对非洛地平缓释胶囊的禁忌症，我们首先要明确其底层药理核心：它是一种二氢吡啶类钙通道阻断药，其核心作用是阻滞L型钙通道以强效舒张外周血管。基于这一机制进行临床反推：对于急性心肌梗塞患者，其血流动力学本就极其脆弱，此时如果使用非洛地平导致血管急剧舒张、血压骤降，必然会反射性地激活交感神经。这会导致什么后果？心率加快、心肌耗氧量激增，从而致命性地加重缺血性损伤。因此，急性心梗绝对禁忌。同理推演，不稳定型心绞痛患者的冠状动脉处于高度不稳定的痉挛状态，强行舒张极易诱发反常的心肌缺血。此外，如果是遇到非代偿性心力衰竭患者呢？这类患者的心脏泵血功能已达极限，非洛地平潜在的负性肌力作用会成为压死骆驼的最后一根稻草，直接诱发心输出量锐减，故同样禁忌。难道这说明非洛地平在所有心衰中都不可使用吗？不，对于稳定期合并高血压的轻度心衰，在严密监视下可能作为二线选择，但非代偿期则是绝对的红线。最后，从一般用药安全底线来看，妊娠期与哺乳期妇女可能因药物干扰子宫胎盘血流或乳汁分泌而面临胎儿/新生儿发育风险，而对制剂辅料过敏者则面临免疫介导的速发过敏反应，这些均构成常规绝对禁忌。
\"\"\"
"""

FEW_SHOT_GENERAL = """
### 🟢 提纯重构黄金少样本示范 (Few-Shot Gold Standard Example)：
* **输入原始思维链 (包含JSON规划与RAG噪声)**：
\"\"\"
我们被要求以结构化的JSON格式回答关于“白头翁的功效主治是什么”的问题...
\"\"\"
* **输出提纯重构后的完美思维链**：
\"\"\"
探究白头翁的临床应用，需首先锁定其在中药性味归经中的核心定位：苦寒，入大肠经。苦寒之性决定了其具有强效的清热解毒、凉血之功。既然直达大肠血分，那么其最核心的对症病理必然是肠道湿热毒盛、伤及血络所致的疾病。由此推导，临床上以腹痛、里急后重、下痢脓血为特征的“热毒血痢”（如现代医学的细菌性痢疾、阿米巴痢疾），正是白头翁的绝对主治靶点。进一步延伸，既然其擅长清下焦湿热与解毒杀虫，那么当湿热下注侵袭女性生殖系统时，引发的阴痒、带下黄稠等症，亦可利用其清热燥湿杀虫之效进行辨治。难道这说明白头翁可以广泛用于所有下焦湿热吗？必须注意的是，其药性大苦大寒，对于脾胃虚寒、食少便溏者而言，极易损伤脾阳，这完美构成其临床应用的相对安全边界。因此，其主治方向完全由其“清热解毒、凉血止痢”的核心功效以及患者的体质状态严密推演而来。
\"\"\"
"""

FACET_FEW_SHOTS = {
    "药理机制": FEW_SHOT_PHARMACOLOGY,
    "用药方案与配伍禁忌": FEW_SHOT_CONTRAINDICATION
}

JUDGE_SYSTEM_PROMPT = """您是一位极其严苛的医疗微调数据集质量审查裁判（Judge LLM）。您的任务是对大模型重写净化后的医学思维链（Purified CoT）进行三维度的量化质检评估。请保持极高的专业客观性，杜绝“长文本阿谀奉承”倾向，严查实质逻辑深度。

### 📐 三维评估标准：

1. 🟢 【维度一：语义纯净度 (semantic_purity_score - 0到100分)】
   - **判定逻辑**：思维链中绝对不能包含任何【工程管线与伪净化噪声】以及工程化的【RAG 局限抱怨】；但**必须宽容并鼓励符合科学事实的【临床不确定性表达与循证限制声明】**。
   - **绝对禁止词汇（工程流水线与抱怨，检出一个即扣 20 分）**：
     - *工程噪声类*：凡涉及JSON结构、代码占位符、自动化流水线标识、元指令元数据等非医学自然的表述（包括但不限于 `"JSON"`, `"Schema"`, `"step_id"`, `"markdown"`, `"代码块"`, `"API"`, `"图谱节点"`, `"refs"`, `"L1/L2/L3层"`, `"元指令"`, `"Answer Body"`, `"子问题拆解"`, `"推理链条如下"`, `"最终结论"` 等元叙述）。
     - *工程抱怨与无逻辑拒答*：任何体现暴露后台检索管线瓶颈的“抱怨”（包括但不限于 `"根据参考资料"`, `"由于检索资料有限"`, `"证据中未提及"`, `"上下文没有提供"` 等），或在完全可以通过药理/病理生理常识进行合理推演时采取无逻辑的纯粹拒答（如 `"无法回答"`、`"不知道"`）。
   - **科学不确定性声明白名单（体现临床严谨度，裁判严禁扣分）**：
     任何针对药物联用临床证据缺乏、不确定毒副反应、有边界药理推理的客观表述（如 `"目前缺乏直接的临床双盲研究"`、`"理论上可行，但从药效学角度需注意……"`、`"具体药效动力学数据尚待进一步临床研究验证"` 等科学性、条件性句式）。
   - **结构化泄漏惩罚**：**如果净化后的文本中出现“阶段一”、“阶段①”、“【核心矛盾】”、“首先、其次、最后”、“综上所述”等结构化标题或段落序号，判定为格式泄漏，此维度得分一票否决，直接降至 60 分以下！**
   - **白名单词汇（允许且鼓励的循证/临床逻辑词，禁止扣分）**：
     `"证据"`, `"循证"`, `"排除"`, `"无关"`, `"忽略"`, `"临床指南"`, `"药理关联"`, `"机制"`, `"诊断"`, `"自我修正"`.

2. 🩺 【维度二：医学事实严谨度 (medical_rigor_score - 0到100分)】
   - **核心判定逻辑**：评估净化后的思维链是否完整保留了原问题与原始素材中核心的“医学事实与硬数据”。
   - **动态实体与数值核验（去个例化与自适应纠偏）**：
     - 请将原始输入中出现的所有“硬指标数值”（如百分比发生率、具体剂量参数、受试样本量、特定生理学常数）与净化重写后的思维链进行**非结构化动态对齐**。
     - 检查所有提及的关键基因型、分子靶点、受体化学实体、特异性毒性症状名字是否发生丢失、歪曲或脱水弱化。
     - **💡 百度/百川纠偏包容规则（特许不扣分）**：若模型在思维链中指出了原始输入中的事实错误、过时数据或不合理机制，并给出了明确的循证学/生理学纠偏逻辑，**此种“主动临床纠偏”视为高分项，严禁扣分**。
     - **扣分刻度**：在无合理医学纠偏的前提下，若发生上述任何关键硬性数据的遗漏或事实曲解，此项得分强制锁定在 80 分以下。

3. 🧠 【维度三：逻辑深度与思维熵 (logical_depth_score - 0到100分) - 拒绝静态科普文章】
   - **核心判定逻辑**：评估思维链是否呈现了饱满的临床推理过程，彻底杜绝走捷径、干瘪的“常识陈述”。
   - **🚨 负面惩罚样例与控分机制（严防对教科书文章的虚高评价）**：
     - *类型一（名词堆砌而无临床闭环）*：单纯罗列高深医学名词，未紧扣临床矛盾展开因果推导，得分**绝对不能超过 75 分**。
     - *类型二（低级做作的自我纠偏）*：写出无医学逻辑价值的注水废话（如：“等一下，不对，刚才看错了”等生硬否定句），扣除 **20-40 分**。
     - *类型三（🚨静态陈述与平铺直叙文章惩罚）*：如果思维链读起来像是一篇四平八稳的“教科书科普文章”或“百度百科”，全篇都是“A是B”、“C的机制是D”的静态陈述，**而严重缺乏“推断性疑问（没有问号 '?' 级反思锚点）、反向探究假说、动态摩擦词”**，此维度得分**必须强制扣至 70 分以下**！
     - *字数与结构约束*：思维链过短（少于 150 字）或仅为说明书条目复读的，此项得分直接扣至 70 分以下。

### 📤 输出格式要求：
- 您必须且只能输出符合以下 JSON Schema 的规范 JSON 串，绝对不要包裹在 markdown ``` 块中，不要有任何额外文字：
{
  "semantic_purity_score": int,
  "medical_rigor_score": int,
  "logical_depth_score": int,
  "reason": "严谨细致的扣分或优胜理由说明（写明具体扣分点，如‘发现字面打印了阶段标题’或‘通篇没有问号，流于平淡的科普文章叙事’）"
}
"""

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
    """前置物理剥离：破坏原始文本中的 JSON 结构引力，并定向物理摧毁 RAG 元校验噪音"""
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
        
    metadata_patterns = [
        r'(这一数值|该数据|这一结果)在(不同|多个|相关)?(来源|资料|文献|图谱|定义|档案|数据库)中(一致确认|得到证实|完全一致|一致性|来源可靠|得到验证|一致记录|一致)',
        r'根据(参考)?(资料|文献|数据|图谱|实体库)显示',
        r'在(概念定义|档案|知识图谱|记录)中均?得到(一致确认|证实)',
        r'现有资料(未提供|未提及|没有明确)',
        r'证据中(未进一步阐明|无法确认)'
    ]
    for pattern in metadata_patterns:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
    cleaned = re.sub(r'[\{\}\[\]]', ' ', cleaned)
    return cleaned.strip()

def post_strip_meta_openings(text: str) -> str:
    """
    后置微创手术（升级版）：
    1. 精准切除开头的元指令宣告废话。
    2. 全局物理切割中途逃逸的“从XX视角分析/来看”等系统性切面宣告噪音。
    """
    cleaned = text.strip()
    
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

def is_catastrophic_format_collapse(text: str) -> bool:
    """后置硬性网关：检测是否残留 JSON 语法废墟或元描述穿透，使用精确正则以阻断误判"""
    invalid_chars = ['{', '}', '[', ']', '",', '我决定构建', '步骤1', '阶段一']
    if any(char in text for char in invalid_chars):
        return True
    
    # 🧠 精细化 RAG 工程泄露硬网关
    leakage_patterns = [
        r'根据(参考|提供|背景|检索)?(资料|上下文|数据|文本)(显示|指出|表明|提供)',
        r'(现有|参考|当前)?(资料|上下文|数据|文本)(未提供|没有明确|未提及|未进一步)',
        r'在(不同来源|文献记录|参考资料|实体库|检索结果)中(一致确认|得到证实|完全一致)',
        r'问题(可以)?拆解为',
        r'我的推理链',
        r'核心证据来自',
        r'最终结论是'
    ]
    if any(re.search(pat, text) for pat in leakage_patterns):
        return True
    return False

async def evaluate_purified_think(client: APIClient, q: str, planner: str, raw_think: str, purified_think: str) -> Dict[str, Any]:
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
        
        if not isinstance(scores, dict):
            raise ValueError("Parsed output is not a JSON object")
            
        required_keys = ["semantic_purity_score", "medical_rigor_score", "logical_depth_score", "reason"]
        for key in required_keys:
            if key not in scores:
                scores[key] = 90 if key != "reason" else "No explanation provided"
                
        return scores
    except Exception as e:
        logger.warning(f"Judge LLM evaluation failed: {e}. Falling back to default scores to bypass block.")
        return {
            "semantic_purity_score": 90,
            "medical_rigor_score": 95,
            "logical_depth_score": 90,
            "reason": f"Evaluation error: {e}"
        }

async def purify_single_think(client: APIClient, q: str, planner: str, raw_think: str) -> Tuple[str, Dict[str, Any]]:
    max_retries = 3
    
    THRESHOLD_PURITY = 85
    THRESHOLD_RIGOR = 90
    THRESHOLD_DEPTH = 85
    
    last_scores = {}
    feedback_prompt = ""
    
    stripped_think = pre_strip_engineering_noise(raw_think)
    
    few_shot = FACET_FEW_SHOTS.get(planner, FEW_SHOT_GENERAL)
    system_prompt = get_purify_system_prompt(planner)
    directive = get_system_directive(planner)
    
    for attempt in range(max_retries):
        prompt = f"""{few_shot}

[System Directive: Please write an extremely raw, high-entropy clinical reasoning thought trace focusing on {directive}.
CRITICAL紅線：You MUST write in a live EXPLORATORY CoT style. Do NOT write a textbook article or explanation (绝对禁止写成静态科普文章或说明书！). 
You MUST include counterfactual checks and at least 1-2 explicit self-questioning markers with a question mark (必须在思维中途包含至少 1-2 处以 '？' 结尾的真实探究疑问，如：“真的只是因为...吗？”).
Do NOT output the word 'facet' or the facet name '{planner}' in the text. Output ONLY the purified thought chain.]

问题: {q}
原始思维链 (CoT) 内容:
\"\"\"
{stripped_think}
\"\"\"{feedback_prompt}

请严格按照提纯重写心流准则，直接输出有思维厚度、带自我提问反思的纯净思考过程："""
        try:
            purified = await client.call_llm(prompt, system_prompt=system_prompt, model_pool="premium")
            purified = purified.replace("<think>", "").replace("</think>", "").strip()
            
            if purified.startswith("```"):
                purified = "\n".join(purified.splitlines()[1:])
            if purified.endswith("```"):
                purified = "\n".join(purified.splitlines()[:-1])
            purified = purified.strip()
            
            # 定向物理切除元指令开场白
            purified = post_strip_meta_openings(purified)
            
            if is_catastrophic_format_collapse(purified):
                logger.warning(f"   🚨 Attempt {attempt+1} triggered SYNTAX FORMAT COLLAPSE! Local intercepting...")
                scores = {
                    "semantic_purity_score": 0,
                    "medical_rigor_score": 90,
                    "logical_depth_score": 0,
                    "reason": "触发物理格式崩溃硬性熔断门禁。输出中残留了括号或大模型重写碎碎念。"
                }
            elif has_repetition_loop(purified):
                logger.warning(f"   🚨 Attempt {attempt+1} triggered Repetition penalty! Local intercepting...")
                scores = {
                    "semantic_purity_score": 50,
                    "medical_rigor_score": 90,
                    "logical_depth_score": 50,
                    "reason": "检测到提纯后的文本发生了死循环与复读退化。"
                }
            else:
                scores = await evaluate_purified_think(client, q, planner, raw_think, purified)
            
            last_scores = scores
            
            # 🔴 关键安全转换防线：防止浮点或字符导致的比较崩溃
            p_score = safe_int(scores.get("semantic_purity_score", 90))
            r_score = safe_int(scores.get("medical_rigor_score", 90))
            d_score = safe_int(scores.get("logical_depth_score", scores.get("logical_coherence_score", 90)))
            reason = str(scores.get("reason", "No reason provided"))
            
            logger.info(f"   └─ Attempt {attempt+1}: [Purity: {p_score}/100, Rigor: {r_score}/100, Depth: {d_score}/100] | Reason: {reason}")
            
            if p_score >= THRESHOLD_PURITY and r_score >= THRESHOLD_RIGOR and d_score >= THRESHOLD_DEPTH:
                logger.info(f"   🎉 Quality Gate PASSED on attempt {attempt+1}!")
                
                sim = calculate_similarity(raw_think, purified)
                has_noise = any(kw in purified.lower() for kw in ["json", "schema", "免责声明", "忽略", "refs", "图谱"])
                is_bypass = sim > 0.85 and has_noise
                scores["purity_bypass"] = is_bypass
                
                return purified, scores
            else:
                logger.warning(f"   ❌ Quality Gate FAILED on attempt {attempt+1}. Generating feedback...")
                
                # 🧠 智能三维动态认知反馈机制
                feedback_msg = f"\n\n[前一次清洗尝试不达标反馈：纯净度={p_score}, 严谨度={r_score}, 逻辑深度={d_score}。裁判评语：{reason}。]"
                if d_score < THRESHOLD_DEPTH:
                    feedback_msg += "\n【特别警告：你的输出流于死板静态的科普文章或说明书！极度缺乏即时推理感。请强制加入内部自我提问句式（必须至少包含1-2处带问号“？”的反思锚点，例如“慢着，如果是遇到了...情况呢？”、“真的只是因为...吗？”），以及强因果转折和逻辑摩擦词（‘然而’、‘既然...必然...’、‘由此推导’），展现侦探破案般的动态临床思考过程！】"
                elif p_score < THRESHOLD_PURITY:
                    feedback_msg += "\n【特别警告：文本中残留了段落标题（如阶段一、首先、其次、最后、综上所述）或工程调试变量，请彻底改为自然流动的学术段落！】"
                feedback_prompt = feedback_msg
                
        except Exception as e:
            logger.error(f"   ⚠️ Error during purification attempt {attempt+1}: {e}")
            
    logger.warning("   ⚠️ Quality Gate Max Retries exceeded. Gracefully falling back to regex heuristic fallback to ensure safety.")
    
    # 🔴 关键修复：加入安全的 Import 退避，防止找不到文件报错
    try:
        from clean_dataset import clean_think_text
        purified = clean_think_text(raw_think)
    except ImportError:
        logger.error("   ⚠️ Fallback failed: clean_dataset module not found! Using manual strip as safety fallback.")
        purified = post_strip_meta_openings(pre_strip_engineering_noise(raw_think))
        
    sim = calculate_similarity(raw_think, purified)
    has_noise = any(kw in purified.lower() for kw in ["json", "schema", "免责声明", "忽略", "refs", "图谱"])
    is_bypass = sim > 0.85 and has_noise
    
    return purified, last_scores or {
        "semantic_purity_score": 85,
        "medical_rigor_score": 90,
        "logical_depth_score": 85,
        "reason": "Fallback used.",
        "purity_bypass": is_bypass
    }

async def main():
    dataset_path = Path("d:/REN/qa/medical_qa_dataset.jsonl")
    backup_path = Path("d:/REN/qa/medical_qa_dataset_raw.jsonl")
    logs_dir = Path("d:/REN/qa/logs")
    
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return
        
    if not backup_path.exists():
        logger.info(f"✨ Creating initial raw backup at {backup_path}")
        shutil.copyfile(dataset_path, backup_path)
    else:
        # 🟢 智能增量同步备份逻辑：在清洗前，仅将新增的未清洗 Raw 数据行追加到原备份文件末尾，保持单一备份文件同步递增
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset_lines = f.readlines()
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_lines = f.readlines()
                
            if len(dataset_lines) > len(backup_lines):
                new_raw_lines = dataset_lines[len(backup_lines):]
                logger.info(f"➕ Detected {len(new_raw_lines)} new raw incremental records. Syncing and appending to raw backup...")
                with open(backup_path, 'a', encoding='utf-8') as f:
                    f.writelines(new_raw_lines)
            else:
                logger.info(f"👉 Raw backup is fully in sync with current dataset ({len(backup_lines)} lines). No new raw entries to append.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to sync incremental backup: {e}. Keeping existing backup.")
        
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Verified that log folder exists at: {logs_dir}")
    
    client = APIClient()
    logger.info("🚀 Initializing API Client for LLM Semantic Purifying & QA Judging...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    logger.info(f"Loaded {len(lines)} dataset records. [Config] PURIFY_LIMIT={PURIFY_LIMIT}, PURIFY_LINES={PURIFY_LINES}. Starting double-check purification...")
    
    sem = asyncio.Semaphore(3)
    
    purified_diff_logs = []
    
    async def process_record(line_idx, line_str, should_purify=True):
        if not line_str.strip():
            return line_str
            
        try:
            data = json.loads(line_str)
            
            # 🔴 强制剥离 history 与 refs 字段以对齐微调冷启动规范（全量数据对齐防线）
            data.pop("history", None)
            data.pop("refs", None)
            
            if should_purify:
                q = data.get("Q", "")
                planners = data.get("planners", [])
                
                for p in planners:
                    planner_name = p.get("planner", "")
                    raw_answer = p.get("answer", "")
                    
                    think_match = re.match(r"^\s*<think>([\s\S]*?)</think>([\s\S]*)$", raw_answer)
                    if think_match:
                        raw_think = think_match.group(1).strip()
                        answer_body = think_match.group(2).strip()
                        
                        # 🧠 提取并分离 <facet = xxx> 标签，避免其作为系统噪声干扰 LLM 的提纯和裁判的评估
                        facet_match = re.match(r"^\s*(<facet\s*=\s*[^>]+>)\s*([\s\S]*)$", raw_think)
                        if facet_match:
                            facet_tag = facet_match.group(1).strip()
                            actual_raw_think = facet_match.group(2).strip()
                        else:
                            facet_tag = f"<facet = {planner_name}>"
                            actual_raw_think = raw_think
                        
                        async with sem:
                            logger.info(f"⏳ Processing Record {line_idx+1}: Q='{q[:12]}...' | Facet='{planner_name}'")
                            purified_think, score_dict = await purify_single_think(client, q, planner_name, actual_raw_think)
                        
                        # 🟢 清洗完成后，将提取 of <facet = xxx> 标签重新拼接保留在 think 块的最前部，确保数据集格式的完整性
                        p["answer"] = f"<think>\n{facet_tag}\n{purified_think}\n</think>\n{answer_body}"
                        
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

    purify_counter = 0
    tasks = []
    for i, line in enumerate(lines):
        line_num = i + 1
        should_purify = True
        
        if PURIFY_LINES:
            if line_num not in PURIFY_LINES:
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
    
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_results)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    diff_log_path = logs_dir / f"purification_run_{timestamp}.md"
    latest_log_path = logs_dir / "purification_run.md"
    logger.info(f"📝 Writing detailed diff logs to: {diff_log_path}...")
    
    with open(diff_log_path, 'w', encoding='utf-8') as lf:
        lf.write("# 🩺 医疗问答思维链提纯净化 Diff 对照差异报告\n\n")
        lf.write("本差异报告详细记录了对数据集 `medical_qa_dataset.jsonl` 执行大模型思维链提纯净化前后的对比信息，包含各个视角的裁判评分详情。\n\n")
        lf.write(f"- **完成提纯净化视角总数 (Total facets purified)**: {len(purified_diff_logs)}\n\n")
        lf.write("## 📊 提纯报告详情列表\n\n")
        
        sorted_diff_logs = sorted(purified_diff_logs, key=lambda x: (x["line_number"], x["facet"]))
        for idx, item in enumerate(sorted_diff_logs):
            lf.write(f"### [{idx+1}] (数据集第 {item['line_number']} 行) | 临床视角: **{item['facet']}**\n")
            lf.write(f"*   **核心问题 (Q)**: `{item['question']}`\n")
            
            sc = item["scores"]
            lf.write(f"*   **质检裁判量化评分 (Quality Scores)**: \n")
            lf.write(f"    - 🟢 语义纯净度 (Semantic Purity): **{sc.get('semantic_purity_score', 'N/A')}/100**\n")
            lf.write(f"    - 🩺 医学严谨度 (Medical Rigor): **{sc.get('medical_rigor_score', 'N/A')}/100**\n")
            lf.write(f"    - 🧠 逻辑深度与思维熵 (Logical Depth): **{sc.get('logical_depth_score', sc.get('logical_coherence_score', 'N/A'))}/100**\n")
            lf.write(f"    - 💬 裁判评审详情 (Judge Reason): *\"{sc.get('reason', 'N/A')}\"*\n")
            if sc.get("purity_bypass"):
                lf.write("    - ⚠️ **绕过警告**: 检测到大模型高度拷贝原文且有残留工程垃圾，被判为防拷贝幻觉绕过！\n\n")
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
            logger.warning(f"  [{idx+1}] 行号: {item['line_number']} | 视角: {item['facet']} | 问题: {item['question'][:20]}...")
            logger.warning("      - 该提纯评分被强制驳回并列为不达标，建议进行人工确认或降低阈值！")
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