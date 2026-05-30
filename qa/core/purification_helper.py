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
    """动态生成系统提示词，注入特异性视角红线拦截"""
    return f"""您位顶级循证医学科学家与大模型思维链（CoT）语料提纯专家。您的任务是净化并重写医学问答数据集中的 `<think>`（思维链）内容，使其达到顶尖的 Reasoning 模型（如 DeepSeek-R1、OpenAI o1）微调的冷启动标准。

### 🚨 提纯与重写的核心目标：
当前 `<think>` 块是工程 Pipeline 自动生成的，混杂了大量【工程指令垃圾】（如 JSON 结构、Refs 引用、RAG 抱怨、格式避让）。如果直接用于模型微调，会导致模型在推理时频繁产生格式泄漏和系统幻觉。您需要将静态的 RAG 段落升华为一个**人类医学专家大脑里真实、高熵、流畅且无污染的“动态临床认知推理心流”**。

### 🚨 思维链概念准入白名单（Ontological Whitelist）：
在重构思维链时，你大脑的“认知世界”中只允许存在以下两类医学实体和逻辑，其他概念对你而言在物理上均不存在：
1. 【患者的生理、病理、临床状态与体征】（如：肾小球滤过率、急性心梗、高频放电、体温、疼痛）。
2. 【药物的分子、受体、药代/药效动力学行为与治疗方案】（如：稳定失活态、竞争性拮抗、半衰期、给药剂量、配伍禁忌）。

**【绝对禁忌红线】**：
绝对禁止在思维链的任何地方提及、暗示或讨论“信息获取途径”（如资料、文献、数据库、实体库、RAG、Refs、数据源、背景信息）、“数据一致性”（如一致确认、得到证实）或“数据结构处理”（如子问题、Answer Body、推理步骤）。任何此类概念一旦流出，均判定为严重泄露，该生成将被物理拦截。你应当假定你脑海里拥有最完备的医学专家常识，直接开始纯粹的临床与药理因果演绎。

### 🛠️ 企业级提纯与重写规范：

1. ❌ 【物理屏蔽一切工程与流水线噪声】（检出一个即扣分）：
   - 彻底移除涉及输出格式、JSON约束、Schema结构、Markdown标记的自我提醒（如：“我们要输出JSON”、“现在开始构建 JSON schema”等）。
   - 彻底移除工程流水线的临时步骤占位符（如：`step_id: P1`、`logic: ...`、`证据对齐清单`、`子问题拆解`、`Answer Body` 等结构性字段名）。

2. 🚨 【严禁伪净化与做题家元描述（Format Translation Bypass）】：
   - 绝对禁止采取“将 JSON 字段名翻译成自然语言来凑字数”的走捷径策略！
   - **绝对禁止在文本中输出任何暗示您在处理一个工程结构化任务的元叙述词汇**。包括但不限于：“我的推理链条如下”、“问题可以拆解为以下子问题”、“最终结论是...”等。
   - **🚨 绝对禁止在文本的任何地方吐出“切面”、“视角”、“角度”等字眼，也绝对禁止直接提及或复读当前的视角名称‘{planner}’！** 严禁出现“从{planner}视角分析”、“根据{planner}角度来看”等任何向用户宣告或暗示你正在以何种视角解题的元叙述句式。

3. 🚨 【严禁 RAG 边界抱怨与参数知识激活（Anti-RAG Complaining & Parametric Activation）】：
   - **绝对禁止在思考链中写出任何关于检索上下文边界局限性的“抱怨”与“免责”表述！**（如：“现有资料未提供该成分的药理作用”、“根据参考资料显示...”、“由于资料有限无法...”）。
   - **机制补偿重构**：即使原始检索资料极其有限，也必须假定您的脑中拥有最完备的医学专家常识，直接激活您的参数化临床知识，对药物各成分或疾病通路展开深度、科学、合理的微观机制推导与因果演绎，决不允许推卸推理责任。

4. 📐 【强化的 5 阶段临床认知深度推理流（Exploratory CoT Trajectory）】：
   优秀的 Reasoning 微调 CoT 必须呈现出“提出假设 -> 探究机制 -> 遇到逻辑分叉/交叉校验 -> 推导排除 -> 决策合拢”的动态心流轨迹（Thought Trace）。
   - **必须使用高密度的逻辑摩擦关联词**：在思维链中，强制要求使用诸如“要剖析...必须深层解构...”、“既然...必然...”、“然而仅靠...是不够的”、“进一步来看...”、“这就完美解释了为何...”、“如果是遇到...情况呢？”、“由此推导...”等带有强烈因果推演和自问自答色彩的动态思考词汇，展现专家大脑内部的真实演算过程。
   您必须引导思维链通过以下 5 个自然的认知阶段隐式递进：
   - **阶段一：核心临床矛盾解构** —— 开头直切临床/医学矛盾核心（例如，直接以物理/医学事实及逻辑摩擦词切入主题：“要剖析[疾病/药物/治疗名称]的核心机制/生理本质，必须深层解构...”，不需要任何结构性的开场白、自问自答或过渡废话）。
   - **阶段二：微观病生理/临床逻辑推演** —— 对分子靶点、受体结合、体内代动学参数或临床指南要点等进行深度因果链条解析，呈现动态心流。
   - **阶段三：逻辑分叉与特殊情况排查** —— 加入自我提问和临床假说排查，增加思维链的“逻辑熵”（例如：“慢着，在此处必须评估：这一情况在特定生理状态下是否会持续...”、“如果是遇到...情况呢？”）。
   - **阶段四：生理/安全极限与查漏补缺** —— 引入对于年龄、肝肾功能受损等特殊情况或安全边界的核验，展示高价值的“自我纠正与查漏补缺”过程。
   - **阶段五：决策自然合拢** —— 以高度学术的口吻，自然推演出最合理的临床或机制结论，禁止出现“综上所述”、“因此最终结论是”等做题套话。

5. 🚨 【防范序号与过渡结构泄漏】：
   - **绝对禁止使用任何如“阶段一”、“步骤1”等序号词，且绝对禁止使用如“首先”、“其次”、“此外”、“最后”、“综上所述”等顺序性或总结性过渡词！** 这些过渡词会暴露结构泄漏。必须使用高度自然的学术因果递进和自问自答。

6. 🔇 【语调与文风红线】：
   - 必须使用绝对的**第三人称、客观学术、冰冷严谨的医学专家视角**。
   - 彻底去除任何对话性废话（如：“好的，让我来为你解答...”、“问题问的是...，我的分析是...”）。直接开始陈述医学事实和逻辑因果，不需要任何开场白或过渡废话。

7. 📤 【输出物理格式要求】：
   - 仅输出净化提纯后的 `<think>` 内部纯文本，绝对不要带有 `<think>` 或 `</think>` 标记本身，也不要包裹在 markdown 围栏中。
"""

FEW_SHOT_PHARMACOLOGY = """### 🟢 提纯重构黄金少样本示范 (Few-Shot Gold Standard Example)：
* **输入原始思维链 (包含JSON规划与RAG噪声)**：
\"\"\"
...主要成分包括丹参、牛膝、天麻、牡丹皮...
\"\"\"
* **输出提纯重构后的完美思维链**：
\"\"\"
要剖析丹膝颗粒的药理学机制，必须深层解构其核心临床目标：缺血性脑血管病恢复期的‘瘀血阻络兼肾虚证’。既然核心矛盾是‘瘀血’与‘肾虚’，方剂的骨架必然以此为基底。我们观察到处方中的丹参、赤芍、川芎，这三者是经典的活血化瘀药对。从现代药理学推演，它们的作用机制显然是指向改善微循环和抑制血小板聚集，从而直接打击脑梗塞后的局部缺血核心病理。然而，仅靠活血通络是不够的。中风后遗症往往伴随长期的机体耗损，处方中紧接着出现的牛膝、地黄、淫羊藿、桑寄生，在微观机制上是为了调节下丘脑-垂体-靶腺轴，通过抗应激损伤来纠正底层的虚损状态。慢着，如果是遇到合并严重肾脏排泄功能受损的患者呢？桑寄生与牛膝中的某些皂苷成分排泄是否会受阻？从临床指南来看，虽然常用剂量下安全性尚可，但对于此类患者仍需加强血清学指标监视。进一步来看，脑血管病患者多伴有血压波动与神经兴奋性异常，这完美解释了方中为何要配伍天麻——利用天麻素的镇静活性来保护脑神经；同时辅以牡丹皮、栀子、决明子来压制烦躁失眠的伴发症状。而在方剂的边缘，为何会出现一味火麻仁？对于中风卧床患者，保持肠道通畅能有效降低腹压，间接稳定血压并减轻脑血管负荷，这是极其精妙的‘釜底抽薪’式次级调节。经此层层解构，十二味药材的配伍逻辑已完全清晰：形成了一个涵盖抗血小板、神经保护、内分泌调节与靶器官减负的多靶点整合网络。
\"\"\""""

FEW_SHOT_CONTRAINDICATION = """### 🟢 提纯重构黄金少样本示范 (Few-Shot Gold Standard Example)：
* **输入原始思维链 (包含JSON规划与RAG噪声)**：
\"\"\"
...非洛地平缓释胶囊的禁忌症包括哪些...
\"\"\"
* **输出提纯重构后的完美思维链**：
\"\"\"
面对非洛地平缓释胶囊的禁忌症，必须要明确其底层药理核心：它是一种二氢吡啶类钙通道阻断药，其核心作用是阻滞L型钙通道以强效舒张外周血管。基于这一机制进行临床反推：对于急性心肌梗塞患者，其血流动力学本就极其脆弱，此时如果使用非洛地平导致血管急剧舒张、血压骤降，必然会反射性地激活交感神经。这会导致什么后果？心率加快、心肌耗氧量激增，从而致命性地加重缺血性损伤。因此，急性心梗绝对禁忌。同理推演，不稳定型心绞痛患者的冠状动脉处于高度不稳定的痉挛状态，强行舒张极易诱发反常的心肌缺血。此外，如果是遇到非代偿性心力衰竭患者呢？这类患者的心脏泵血功能已达极限，非洛地平潜在的负性肌力作用会成为压死骆驼的最后一根稻草，直接诱发心输出量锐减，故同样禁忌。而在更宽泛的临床防御维度上，从一般用药安全底线来看，妊娠期与哺乳期妇女可能因药物干扰子宫胎盘血流或乳汁分泌而面临胎儿/新生儿发育风险，而对制剂辅料过敏者则面临免疫介导的速发过敏反应，这些均构成常规绝对禁忌。
\"\"\""""

FEW_SHOT_GENERAL = """### 🟢 提纯重构黄金少样本示范 (Few-Shot Gold Standard Example)：
* **输入原始思维链 (包含JSON规划与RAG噪声)**：
\"\"\"
...关于“白头翁的功效主治是什么”的问题...
\"\"\"
* **输出提纯重构后的完美思维链**：
\"\"\"
探究白头翁的临床应用，必须牢牢锁定其在中药性味归经中的核心定位：苦寒，入大肠经。苦寒之性决定了其具有强效的清热解毒、凉血之功。既然直达大肠血分，那么其最核心的对症病理必然是肠道湿热毒盛、伤及血络所致的疾病。由此推导，临床上以腹痛、里急后重、下痢脓血为特征的“热毒血痢”（如现代医学的细菌性痢疾、阿米巴痢疾），正是白头翁的绝对主治靶点。进一步延伸，既然其擅长清下焦湿热与解毒杀虫，那么当湿热下注侵袭女性生殖系统时，引发的阴痒、带下黄稠等症，亦可利用其清热燥湿杀虫之效进行辨治。因此，其主治方向完全由其“清热解毒、凉血止痢”的核心功效严密推演而来。
\"\"\""""

FACET_FEW_SHOTS = {
    "药理机制": FEW_SHOT_PHARMACOLOGY,
    "用药方案与配伍禁忌": FEW_SHOT_CONTRAINDICATION
}

JUDGE_SYSTEM_PROMPT = """您是一位极其严苛的医疗微调数据集质量审查裁判（Judge LLM）。您的任务是对大模型重写净化后的医学思维链（Purified CoT）进行三维度的量化质检评估。请保持极高的专业客观性，杜绝“长文本阿谀奉承”倾向，严查实质逻辑深度。

### 📐 三维评估标准：

1. 🟢 【维度一：语义纯净度 (semantic_purity_score - 0到100分)】
   - **核心判定逻辑（元叙述语义自检法）**：
     请利用你的高级语义理解力，深度审读净化后的思维链（Purified CoT），并一票否决任何包含【元叙述（Meta-narrative）】与【RAG工程泄露】的生成。
     - **元叙述的语义判定特征（一旦触碰，此维度得分直接降至 60 分以下并驳回重写）**：
       1. 文本是否在解释或讨论“本段文字自身的生成过程”？（如：“假如优先结合如此明确，为何证据源会出自一个名为‘5-羟色胺1A受体’的实体库？” —— 属于对外部数据源或自身生成逻辑的自我讨论，属于严重元叙述泄漏！）。
       2. 文本中是否提到了任何指代“外部参考数据、文献数据库、检索图谱、说明书记录、实体库、数据源”的词汇？（无论其名字是什么，只要在讨论或暗示“参考的信息来源”即属违规）。
       3. 文本中是否在向读者宣告自己正在进行格式拆解、步骤划分或切面解题？（绝对禁止出现“阶段一”、“步骤一”、“临床视角分析”等向外部表露结构的行为）。
   - **绝对禁止词汇（工程流水线与抱怨，检出一个即扣 20 分）**：
     - *工程噪声类*：凡涉及JSON结构、代码占位符、自动化流水线标识、元指令元数据等非医学自然的表述（包括但不限于 `"JSON"`, `"Schema"`, `"step_id"`, `"markdown"`, `"代码块"`, `"API"`, `"图谱节点"`, `"refs"`, `"L1/L2/L3层"`, `"元指令"`, `"Answer Body"`, `"子问题拆解"`, `"推理链条如下"`, `"最终结论"`, `"实体库"`, `"数据源"` 等元叙述）。
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
      - **💡【图数据库与 refs 刚性事实一致性核验】**：提纯后的思维链必须与原始素材中**来自于医学知识图谱数据库（Graph Database，即 refs 中标注为《实体库:xxx》或《图谱关系:xxx》的定义与关联）**进行硬性对齐校验。严禁篡改图数据库中明确定义的实体化学属性、靶点归宿、以及图谱中已确认的临床相互关系。任何与图谱数据库核心证据相冲突的编造（例如图谱表明两药有配伍禁忌而思维链内容谎称可以同用，或者图谱指出该药为噻唑并吡啶骨架而思维链篡改为噻吩并吡啶）一经判定，属于严重事实违规，医学严谨度得分一票否决直接扣至 50 分以下并驳回重写！
     - **🚨【绝对红线：严打学术伪造与高仿真幻觉】**：裁判大模型必须动用你知识库中最精密的生化与临床药理知识，严密审视提纯后的思维链是否存在“凭空捏造分子结构骨架（例如将噻唑并吡啶记错为噻吩并吡啶）、伪造 pKa 电离常数与电荷分布、虚构/篡改临床特异性逆转剂的实际结合效能（例如谎称 Andexanet 无法螯合艾多沙班）”等行为。一旦发现模型在“没有确切 refs 证据且缺乏医学公理支撑”的前提下，使用极为自信的伪学术行话进行事实虚构，判定为“高仿真幻觉违规”，此维度得分一票否决直接扣至 50 分以下并驳回重写！

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
