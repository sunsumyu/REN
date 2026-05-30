# -*- coding: utf-8 -*-
import re
from pathlib import Path

# Paths to the scripts
workspace = Path("d:/REN/qa")
purifier_path = workspace / "scripts/medicalqa_purifier.py"
dataset_path = workspace / "scripts/llm_purify_dataset.py"
dataset_opt_path = workspace / "scripts/llm_purify_dataset_opt.py"

NEW_ONTOLOGY = """

### 🚨 思维链概念准入白名单（Ontological Whitelist）：
在重构思维链时，你大脑的“认知世界”中只允许存在以下两类医学实体和逻辑，其他概念对你而言在物理上均不存在：
1. 【患者的生理、病理、临床状态与体征】（如：肾小球滤过率、急性心梗、高频放电、体温、疼痛）。
2. 【药物的分子、受体、药代/药效动力学行为与治疗方案】（如：稳定失活态、竞争性拮抗、半衰期、给药剂量、配伍禁忌）。

**【绝对禁忌红线】**：
绝对禁止在思维链的任何地方提及、暗示或讨论“信息获取途径”（如资料、文献、数据库、实体库、RAG、Refs、数据源、背景信息）、“数据一致性”（如一致确认、得到证实）或“数据结构处理”（如子问题、Answer Body、推理步骤）。任何此类概念一旦流出，均判定为严重泄露，该生成将被物理拦截。你应当假定你脑海里拥有最完备的医学专家常识，直接开始纯粹的临床与药理因果演绎。
"""

NEW_JUDGE = """JUDGE_SYSTEM_PROMPT = \"\"\"您是一位极其严苛的医疗微调数据集质量审查裁判（Judge LLM）。您的任务是对大模型重写净化后的医学思维链（Purified CoT）进行三维度的量化质检评估。请保持极高的专业客观性，杜绝“长文本阿谀奉承”倾向，严查实质逻辑深度。

### 📐 三维评估标准：

1. 🟢 【维度一：语义纯净度 (semantic_purity_score - 0到100分)】
   - **核心判定逻辑（元叙述语义自检法）**：
     请利用你的高级语义理解力，深度审读净化后的思维链（Purified CoT），并一票否决任何包含【元叙述（Meta-narrative）】与【RAG工程泄露】的生成。
     - **元叙述的语义判定特征（一旦触碰，此维度得分直接降至 60 分以下并驳回重写）**：
       1. 文本是否在解释或讨论“本段文字自身的生成过程”？（如：“假如此优先结合如此明确，为何证据源会出自一个名为‘5-羟色胺1A受体’的实体库？” —— 属于对外部数据源或自身生成逻辑的自我讨论，属于严重元叙述泄漏！）。
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
\"\"\""""

NEW_PRE_STRIP = """def pre_strip_engineering_noise(raw_text: str) -> str:
    \"\"\"
    通用语义前置去标识化解析器：
    利用正则结构匹配而非特定词汇，物理剥离一切 RAG 引用、文献索引与图谱关系包裹，实现100%泛化阻断。
    \"\"\"
    # 1. 物理移除所有的 JSON/步骤结构
    noise_patterns = [
        r'"sub_questions":\\s*\\[.*?\\]',
        r'"evidences":\\s*\\[',
        r'"reasoning_chains":\\s*\\[',
        r'\\{"step_id".*?"logic":\\s*',
        r'"location":\\s*".*?"',
        r'"source":\\s*".*?"'
    ]
    cleaned = raw_text
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.DOTALL)
        
    # 2. 【高泛化 RAG 结构剥离】
    # 匹配 "根据《...》的描述/显示/可知" 并完全剔除，只保留核心陈述
    cleaned = re.sub(r'根据《[^》]+》的?(描述|记载|显示|数据|图谱|关系|档案|文献|实体库)?(显示|可知|指出|表明|提供)?，?', '', cleaned)
    cleaned = re.sub(r'《[^》]+》', '', cleaned)
    
    # 3. 【高泛化文献索引剥离】
    # 匹配 "根据PubMed (PMID: 1234)的研究/报道"
    cleaned = re.sub(r'根据\\s*PubMed\\s*\\(PMID:\\s*\\d+\\)\\s*的?(报道|研究|文献|病例)?，?', '', cleaned)
    cleaned = re.sub(r'PMID:\\s*\\d+', '', cleaned)
    cleaned = re.sub(r'PubMed\\s*\\([^)]+\\)', '', cleaned, flags=re.IGNORECASE)
    
    # 4. 清理残留括号与物理杂质
    cleaned = re.sub(r'[\\{\\}\\[\\]]', ' ', cleaned)
    return cleaned.strip()"""

NEW_COLLAPSE = """def is_catastrophic_format_collapse(text: str) -> bool:
    \"\"\"后置硬性网关：检测是否残留 JSON 语法废墟或元描述穿透，使用精确正则以阻断误判\"\"\"
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
    return False"""

def update_script(path):
    print(f"Updating {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Replace target string for Ontological Whitelist
    target_sentence = "您需要将静态的 RAG 段落升华为一个**人类医学专家大脑里真实、高熵、流畅且无污染的“动态临床认知推理心流”**。"
    if target_sentence in content:
        if "思维链概念准入白名单（Ontological Whitelist）" not in content:
            # We want to insert the whitelist after target_sentence
            content = content.replace(target_sentence, target_sentence + NEW_ONTOLOGY)
            print("  Inserted Ontological Whitelist.")
        else:
            print("  Ontological Whitelist already exists.")
    else:
        # Fallback search if asterisks are slightly different or missing
        alt_sentence = "您需要将静态的 RAG 段落升华为一个人类医学专家大脑里真实、高熵、流畅且无污染的“动态临床认知推理心流”。"
        if alt_sentence in content:
            if "思维链概念准入白名单（Ontological Whitelist）" not in content:
                content = content.replace(alt_sentence, alt_sentence + NEW_ONTOLOGY)
                print("  Inserted Ontological Whitelist (alt match).")
            else:
                print("  Ontological Whitelist already exists.")
        else:
            # Try plain text search
            target_sentence_plain = "您需要将静态的 RAG 段落升华为一个人类医学专家大脑里真实、高熵、流畅且无污染的“动态临床认知推理心流”。"
            if target_sentence_plain in content:
                if "思维链概念准入白名单（Ontological Whitelist）" not in content:
                    content = content.replace(target_sentence_plain, target_sentence_plain + NEW_ONTOLOGY)
                    print("  Inserted Ontological Whitelist (plain match).")
                else:
                    print("  Ontological Whitelist already exists.")
            else:
                print("  Error: Target sentence not found for Ontological Whitelist!")

    # 2. Replace JUDGE_SYSTEM_PROMPT
    pattern = r'(JUDGE_SYSTEM_PROMPT\s*=\s*"""[\s\S]*?""")'
    match = re.search(pattern, content)
    if match:
        content = re.sub(pattern, NEW_JUDGE, content)
        print("  Replaced JUDGE_SYSTEM_PROMPT.")
    else:
        pattern2 = r'(JUDGE_SYSTEM_PROMPT\s*=\s*"""[\s\S]*?\}""")'
        if re.search(pattern2, content):
            content = re.sub(pattern2, NEW_JUDGE, content)
            print("  Replaced JUDGE_SYSTEM_PROMPT (ended with }).")
        else:
            print("  Error: JUDGE_SYSTEM_PROMPT block not found!")

    # 3. Replace pre_strip_engineering_noise function
    pre_strip_pattern = r'def pre_strip_engineering_noise\([\s\S]*?return cleaned\.strip\(\)'
    if re.search(pre_strip_pattern, content):
        content = re.sub(pre_strip_pattern, NEW_PRE_STRIP, content)
        print("  Replaced pre_strip_engineering_noise.")
    else:
        print("  Error: pre_strip_engineering_noise function not found!")

    # 4. Replace is_catastrophic_format_collapse function
    collapse_pattern = r'def is_catastrophic_format_collapse\([\s\S]*?return False'
    if re.search(collapse_pattern, content):
        content = re.sub(collapse_pattern, NEW_COLLAPSE, content)
        print("  Replaced is_catastrophic_format_collapse.")
    else:
        print("  Error: is_catastrophic_format_collapse function not found!")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Finished updating {path.name}")

if __name__ == "__main__":
    update_script(purifier_path)
    update_script(dataset_path)
    update_script(dataset_opt_path)
    print("All updates applied successfully!")
