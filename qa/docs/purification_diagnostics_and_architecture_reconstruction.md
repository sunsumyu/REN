# 🩺 医疗问答数据提纯质量诊断与企业级架构重构方案

在对数据集进行数据提纯净化过程中，系统经历了多起物理回滚（物理删除并隔离至 `purification_failures.jsonl`）以及部分成功行中残留 RAG/工程痕迹的现象。本标准技术规范文档旨在深度剖析这些问题的根本成因，对比业界主流大厂的落地实践，并设计出一套企业级的架构级与代码级重构方案。

---

## 一、 故障深度诊断与根本原因分析

结合日志审计与代码逻辑剖析，提纯流水线在运行过程中，故障与工程泄露主要归结为以下九大核心维度：

### 1. 宽窄域不匹配与代码传参缺陷导致思考链写宽 (Thinking-Answer Set Incongruence)
*   **故障现象**：第 195 行（罗浮山百草油主治）和第 231 行（索磷布韦维帕他韦片基因型）等数据触发质检红线，导致整行物理删除。
*   **成因分析**：
    *   **传参空转 Bug**：在 [purify_single_think](file:///d:/REN/qa/core/purification_engine.py#L67) 方法中，接口虽然接收了已缩窄的回答正文 `purified_answer: str` 作为输入，但**该参数并未在提示词模板或后续逻辑中被实际使用**。
    *   **盲盒式 CoT 重写**：由于 CoT 提纯模型在重写时完全看不到缩窄后的 Answer 边界，只能盲目遵循“深推理、高熵”提示指令，在重写 `<think>` 时依然对被 Answer 剔除的旁路病生理机制进行了大篇幅推演。
    *   **一致性红线否决**：裁判模型在打分时，受限于 **“思考链与最终回答一致性红线”**，检测到 CoT 中推导了 Answer 中被剥离的实体细节，判定为严重信息不对称，直接一票否决并回滚。

### 2. 裁判模型“盲评”导致事实对齐被误判为幻觉 (LLM Judge Lacks Grounding Facts)
*   **故障现象**：部分药物在提纯时表现出极佳的药理分析，但被裁判判定为“高仿真事实幻觉”，从而被强制打低分并回滚。
*   **成因分析**：
    *   **数据链路断裂**：在 [LLMJudgeStrategy.evaluate](file:///d:/REN/qa/strategies/quality_gate/llm_judge.py#L22) 中，裁判模型的 Prompt 仅接收了 `q`、`planner`、`raw_think`、`purified_think`，**完全没有传入原始医学参考依据（`refs`）**。
    *   **裁判认知受限**：这导致裁判模型将 `raw_think`（原版包含噪音且信息不全的草稿）默认为唯一的真理源。当提纯模型从真实的 `refs` 中提取出准确的配伍或临床事实写入 CoT 时，裁判 Judge 误信模型在“凭空捏造”，判定严谨度不及格，引发物理回滚。

### 3. RAG 检索抱怨穿透与边界暴露 (RAG Complaining & Network Grabbing Leakage)
*   **故障现象**：部分未收录或抓取失败的行在提纯时被回滚隔离。
*   **成因分析**：
    *   **原始数据污染**：原始 refs 包含了如 `【未收录或网络异常】: 当前未能在公开在线网络中抓取到...` 的检索失败日志。
    *   **抱怨穿透**：提纯模型在防幻觉退避时，在 CoT 或 Answer 中残留了如“由于公开数据库暂未收录”或“无法根据现有资料推断”等言论。这直接触发了格式崩溃熔断正则（`due to (资料/数据) (没有/有限/不足)`），导致纯净度不达标。

### 4. 硬性拦截字符设计过粗导致高风险“误杀” (Brittle Regex Overkill)
*   **故障现象**：格式崩溃硬性熔断门禁高频触发，导致评分被强制打零。
*   **成因分析**：
    *   **正则表达式脆弱**：在 [is_catastrophic_format_collapse](file:///d:/REN/qa/core/purification_helper.py#L239) 中，开发人员直接把方括号字符 `'['` 和 `']'` 列为了非法的 `invalid_chars`。
    *   **学术表达误杀**：医学领域正常出现的基因型标识 `[CYP2B6*6]`、标准文献索引 `[1]`、或者是英文说明书商品名 `[Epclusa]` 等内容均被强行断定为 JSON 语法废墟泄露，导致优质 CoT 被无情回滚并删除。

### 5. 药理问题偏离至药代动力学 (Irrelevant Pharmacokinetic Diversion)
*   **故障现象**：塞来昔布药理机制等逻辑深度得分过低导致回滚。
*   **成因分析**：
    *   **混入药代杂质**：塞来昔布发挥药理机制主要是 PD（选择性抑制 COX-2）。但模型在生成 CoT 时，由于原始 refs 提到了 CYP2C9 代谢酶，模型在思考链中进行了大篇幅的 PK（药物代谢暴露量）推演，被裁判判定为偏离主线并扣分。

### 6. 清洗映射表盲区与验证器漏检风险 (Wash Map Gaps & Validator Under-scanning)
*   **故障现象**：部分成功提纯的行仍残留了诸如“实体信息”、“概念定义中明确列出”、“信源”、“根据：基线值”等工程代偿词。
*   **成因分析**：
    *   **特征词表覆盖不足**：词表未将 `"实体信息"`、`"概念定义"`、`"信源"` 等中介代偿词纳入映射。
    *   **验证扫描盲区**：自动化验证程序 [verify_purification.py](file:///d:/REN/qa/scripts/verify_purification.py#L71) **仅扫描了 `think_content`，完全忽略了 `answer_body` 和 `summary`**。由于这些字段直接用于微调训练并直接展示给终端用户，这些地方残留工程词会导致训练出的模型在推理时严重泄漏工程特征。

### 7. RAG Source 直接注入生成 Prompt 导致噪声代偿
*   **故障现象**：模型输出中反复出现“根据信源显示”、“根据图谱数据库”等工程腔调。
*   **成因分析**：
    *   **源头注入污染**：在 [purification_engine.py](file:///d:/REN/qa/core/purification_engine.py#L109) 中，系统直接读取 raw refs 中的 `source` 并与 context 拼接为 `f"- [{src}] {clean_ctx}"` 喂给生成模型。如果 `src` 中含有 `refs:《实体库:xxx》`，生成模型为了对齐，便会在语言空间中模仿并代偿输出工程词汇。

### 8. 粗粒度证据路由导致非核心噪音引入
*   **成因分析**：
    *   [EvidenceScopeRouter](file:///d:/REN/qa/core/rag/evidence_scope_router.py#L101) 的证据分类较粗。在处理 `GENERAL_MEDICAL` 时，常把用药方法、肝肾功能、不良反应等次要证据全部塞入 Prompt 中，诱导模型在思考链中产生不必要的旁路分支。

### 9. 整行一票否决回滚机制导致高损耗与行号漂移
*   **成因分析**：
    *   在 [medicalqa_purifier.py](file:///d:/REN/qa/scripts/medicalqa_purifier.py#L805) 中，只要行内的**某一个 planner** 发生提纯失败或退出，整行数据就会被物理回滚删除。这不仅浪费了并发执行中已提纯成功的 planner 结果，也导致了数据集行号的不断漂移。

---

## 二、 业界大厂及前沿学术界技术方案对比

在处理合成 CoT 数据纯净度、逻辑一致性及评估门禁设计时，业界头部大厂（OpenAI、Anthropic、字节跳动、阿里巴巴等）沉淀了如下最佳实践：

### 1. 针对“思维-答案不一致（CoT-Answer Unfaithfulness）”
*   **大厂实践（OpenAI & Anthropic）**：
    *   **同步联合重写（Joint Rewriting）**：只用一个统一的 System Prompt 引导模型在单次生成中同时输出 CoT 和 Answer，在 Prompt 中强制约束“Answer 的每一步结论必须有 CoT 对应的显式引用”。
    *   **反向 CoT 合成（Reverse CoT Synthesis）**：如果 Answer 已经被专家或轻量大模型精简/裁切，应把此精简后的 Answer 作为强约束条件，反向输入给旗舰级模型，要求其**彻底重构一个与该 Answer 边界完全吻合的纯净 CoT**，实现“以终为始”的完美对齐。

### 2. 针对“裁判模型盲评（Referenceless Judging Bias）”
*   **行业标准（UCB / MT-Bench & Stanford AlpacaEval）**：
    *   在评测客观事实、知识提取、RAG 以及医疗/法律等严谨垂直领域时，**Reference-Guided Evaluation（循证参考引导评估）** 是目前工业界公认唯一可接受的方案。
    *   评测提示词必须包含“参考答案源”（Ground Truth / References），作为 Judge 模型的物理锚点，坚决避免 Judge 使用其自身的参数化知识盲目推断。这不仅保证了评分一致性，还消除了 verbosity bias（字数偏置）和对未知常识的物理误判。

### 3. 针对“工程残留与清洗”
*   **业界实践（Anthropic Constitutional AI / Llama Data Pipeline）**：
    *   **语法树解析器（AST Parser）**：使用抽象语法树（AST）或强类型的 JSON Schema 解析器代替简单的字符匹配和暴力正则。对于纯文本数据，仅当出现嵌套字典特征时才判定为格式泄露，不对方括号做硬性屏蔽。
    *   **Critique and Revision Loop（批判-修正闭环）**：利用旗舰级模型（如 Claude 3.5 Sonnet / GPT-4o）在数据落地前跑一层“自我纠错”（Self-Correction），智能识别并替换学术包装下的工程残留（如将“根据实体库记载”抹平为“文献报道”）。

---

## 三、 企业级重构与优化方案 (Code Blueprints)

为了在不直接修改源码逻辑的前提下（只分析、做方案，提供待实施的架构蓝图），设计以下可直接 drop-in 实施的企业级代码改动。

### 方案 1. 循证裁判评估重构 (Reference-Guided Judge with Clean Facts)
重构 [LLMJudgeStrategy.evaluate](file:///d:/REN/qa/strategies/quality_gate/llm_judge.py#L22)，将参考文献（`refs`）脱敏并解析为纯粹的**原子事实包（Cleaned Facts）**喂给 Judge 避免其盲评；坚决不喂原始 source 文本，防止裁判模型自身被工程标签二次污染。

#### 💡 代码层修改 blueprint：
```diff
# strategies/quality_gate/llm_judge.py

 class LLMJudgeStrategy(IEvaluationStrategy):
     def __init__(self, llm_service: ILLMService):
         self.llm_service = llm_service
 
-    async def evaluate(self, q: str, planner: str, raw_think: str, purified_think: str, line_num: int = None) -> Dict[str, Any]:
+    async def evaluate(self, q: str, planner: str, raw_think: str, purified_think: str, line_num: int = None, refs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
         """
         Evaluate the purified thought chain under strict three-dimensional quality gates.
         """
+        # 格式化原始数据源（剔除工程标签，保留纯净的循证事实供 Judge 核验）
+        cleaned_facts_text = ""
+        if refs:
+            cleaned_facts_text = "\n### 原始循证医学事实依据 (Ground Truth Cleaned Facts):\n"
+            for idx, r in enumerate(refs, start=1):
+                if isinstance(r, dict):
+                    ctx = r.get('context', 'N/A')
+                    # 清理可能携带的 RAG 抱怨和工程词头
+                    clean_ctx = ctx.replace("【互联网权威医疗站快讯】:", "").replace("【互联网权威医疗数据通报】:", "").strip()
+                    cleaned_facts_text += f"- fact_{idx:03d}: {clean_ctx}\n"
+
         prompt = f"""问题: {q}
 切面视角: {planner}
+
+{cleaned_facts_text}
+
 原始思维链 (包含噪声):
 \"\"\"
 {raw_think}
 \"\"\"
 
 净化重写后的思维链:
 \"\"\"
 {purified_think}
 \"\"\"
 
 请严格按照质检准则对净化后的思维链进行三维评分，并直接输出规范 of JSON 数据。"""
```

同时，在 [core/purification_engine.py](file:///d:/REN/qa/core/purification_engine.py#L186) 中，调用裁判时传入 `refs` 参数：
```diff
# core/purification_engine.py

-                    scores = await self.evaluator.evaluate(q, smoothed_planner, raw_think, purified, line_num=line_num)
+                    scores = await self.evaluator.evaluate(q, smoothed_planner, raw_think, purified, line_num=line_num, refs=active_refs)
```

---

### 方案 2. 结构感知型格式崩溃检测器 (AST/Structure-based Collapse Gate)
重构 [is_catastrophic_format_collapse](file:///d:/REN/qa/core/purification_helper.py#L239)，解除方括号 `[` 和 `]` 的直接查杀，改为利用 AST 结构进行稳健的语法树解析。

#### 💡 代码层修改 blueprint：
```python
# core/purification_helper.py

import ast

def is_catastrophic_format_collapse(text: str) -> bool:
    """
    企业级结构感知网关：
    1. 阻断硬性字符误杀，支持医学表达中的正常方括号和括号。
    2. 利用轻量语法树解析判定是否有真正的 JSON 废墟或元描述泄露。
    """
    if not text:
        return False
        
    # 检测是否包含 JSON 键值对废墟（如 `"sub_questions":` 或 `"step_id":`）
    json_ruin_patterns = [
        r'"sub_questions"\s*:',
        r'"evidences"\s*:',
        r'"reasoning_chains"\s*:',
        r'"step_id"\s*:',
        r'"logic"\s*:'
    ]
    if any(re.search(pattern, text) for pattern in json_ruin_patterns):
        return True

    # 仅当文本能被成功解析为字典或包含非法的未闭合 JSON 边界时才查杀
    if text.strip().startswith("{") and text.strip().endswith("}"):
        try:
            parsed = ast.literal_eval(text.strip())
            if isinstance(parsed, dict):
                return True # 说明提纯结果退化为纯数据结构，未重构为学术推理流
        except Exception:
            pass

    # 精细化 RAG 工程泄露与元叙述硬网关
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
```

---

### 方案 3. CoT 与 Answer 联合提纯与对齐重写 (Fixing empty parameter bug)
修复 [purify_single_think](file:///d:/REN/qa/core/purification_engine.py#L67) 中 `purified_answer` 空转的 Bug，将其作为强力事实边界显式灌入 CoT 生成模型的重写 Prompt 中，并与 System Prompt 进行融合。

#### 💡 代码层修改 blueprint (core/purification_engine.py)：
```diff
# core/purification_engine.py

         for attempt in range(max_retries):
             simplify_prompt_addition = ""
             if simplify:
                 simplify_prompt_addition = "\n【⚠️ 极简重构硬性要求】：该问题为简单事实查询，严禁脑补虚构复杂的分子机制...（略）"
             
+            # 显式拼接已缩窄的 Answer Body 限制生成边界，确保 Think 宽度绝不宽于 Answer
+            answer_boundary_prompt = ""
+            if purified_answer:
+                answer_boundary_prompt = f"""
+
+### 已提纯的回答正文 (Purified Answer Body Boundary):
+{purified_answer}
+【⚠️ 思考链事实边界硬对齐红线】：上文为该行提纯后的唯一最终回答正文。你的 CoT 思考流（Think）必须全程且仅围绕本正文中包含的事实展开。绝对禁止在思考链中讨论或推导演答正文未提及的任何旁路药物成分、次要机制、或临床研究！"""
+
             prompt = f"""{few_shot}
 
 ### 系统指令 (System Directive)：
-Please write an extremely raw, high-entropy clinical reasoning thought trace focusing on {directive}.（略）{anchors_prompt}
+Please write an extremely raw, high-entropy clinical reasoning thought trace focusing on {directive}.（略）{anchors_prompt}{answer_boundary_prompt}
 
 问题: {q}
```

---

### 方案 4. 去噪清洗中性转译 (Neutral Wash Map to Avoid Falsification)
修改 [services/healing_service.py](file:///d:/REN/qa/services/healing_service.py#L92) 中的映射表。**杜绝强行将“信源”伪造为“说明书”等越界行为**，改为使用中性学术词汇（如 `临床参考数据`、`科学文献记录`），并对多余冒号进行清洗。

#### 💡 代码层修改 blueprint：
```python
# services/healing_service.py

            # 升级为中性替换表，扫除代偿词的同时防止伪造事实源
            SEMANTIC_WASH_MAP = {
                "刚性事实锚点": "临床确证事实",
                "刚性事实": "文献记载事实",
                "刚性锚点": "临床证据",
                "图谱关系": "已知关联",
                "图谱显示": "文献记载",
                "根据参考资料显示": "文献数据显示",
                "根据参考资料": "文献记录",
                "根据背景信息": "循证文献报道",
                "根据背景数据": "临床数据表明",
                "根据检索": "科学文献记录",
                
                # 扫除盲区新增，采取中性学术化转译
                "实体库": "",
                "知识图谱": "",
                "数据源": "",
                "实体信息": "相关文献记录",
                "概念定义": "临床文献记载",
                "信源": "参考数据",
                "知识库": "循证科学文献",
                "查询确证记录": "文献研究报道",
                "现有的素材": "相关文献记载"
            }
            
            for eng_word, med_word in SEMANTIC_WASH_MAP.items():
                if eng_word in cleaned:
                    cleaned = cleaned.replace(eng_word, med_word)

            # 🛡️ 冒号自愈正则清洗：平滑形如 "根据:基线值" 的生硬工程符号
            cleaned = re.sub(r'根据\s*:\s*', '根据', cleaned)
            cleaned = re.sub(r'文献\s*:\s*', '文献', cleaned)
            cleaned = re.sub(r'参考数据\s*:\s*', '参考数据', cleaned)
            cleaned = re.sub(r'([，。：])\s*[:：]+', r'\1', cleaned)
```

---

### 方案 5. RAG Source 注入源头脱敏重构
重构 [purification_engine.py](file:///d:/REN/qa/core/purification_engine.py#L109) 中 anchors 的拼装逻辑，将 raw refs 中的工程源（如 `refs:《实体库:索磷布韦维帕他韦片》`）彻底剔除或转译为纯数字文献标签，阻断模型在源头接触工程噪音。

#### 💡 代码层修改 blueprint (core/purification_engine.py)：
```python
# core/purification_engine.py

        if active_refs:
            anchors = []
            for idx, r in enumerate(active_refs, start=1):
                if isinstance(r, dict):
                    ctx = r.get("context", "")
                    if ctx:
                        # 剥离多余的工程前缀，提取纯粹的事实陈述
                        clean_ctx = ctx.replace("【互联网权威医疗站快讯】:", "").replace("【互联网权威医疗数据通报】:", "").strip()
                        # 🛡️ 阻断原始 RAG source 标签暴露，Judge 和生成均以脱敏文献标签 [idx] 代表事实来源
                        anchors.append(f"- [文献_{idx:02d}] {clean_ctx}")
```

---

### 方案 6. 验证器多字段同步校验与禁词扩展
重构 [verify_purification.py](file:///d:/REN/qa/scripts/verify_purification.py#L70)，将禁词表匹配范围由单纯的 `think` 字段，同步扩展至 `answer_body` 和 `summary` 字段，以保微调（SFT）训练集输出无工程噪声污染。

#### 💡 代码层修改 blueprint (scripts/verify_purification.py)：
```python
# scripts/verify_purification.py

                        think_content = think_match.group(1)
                        answer_body = think_match.group(2)
                        
                        # 1. 扩展绝对禁止词词表
                        banned_keywords = [
                            "JSON", "Schema", "免责声明", "忽略", "refs", "根据参考文档", 
                            "图谱关系", "概念定义", "知识关联", "实体库", "知识图谱", 
                            "数据源", "实体信息", "网络异常", "抓取", "信源", "条目"
                        ]
                        
                        # 2. 🛡️ 多文本字段联动校验，覆盖 think、answer 以及整个输出
                        found_banned = []
                        for kw in banned_keywords:
                            # 同时核验 think_content 和 answer_body 甚至整个回答
                            if kw in think_content or kw in answer_body or kw in answer:
                                found_banned.append(kw)
                        
                        if found_banned:
                            noise_leaks += 1
                            print(f"[WARN] 行 {line_idx+1} [{planner_name}] 检出工程残留: {found_banned}")
```

---

### 方案 7. Planner 级局部失败隔离与事务逻辑重构
重构 [medicalqa_purifier.py](file:///d:/REN/qa/scripts/medicalqa_purifier.py#L805)，取消“一planner失败株连整行”的严苛回滚，实现 planner 级别的局部提纯持久化与隔离重试。

#### 💡 代码层修改 blueprint (scripts/medicalqa_purifier.py)：
```python
# scripts/medicalqa_purifier.py

            valid_planners = []
            local_diff_logs = []
            failed_planners = []
            pruned_count = 0
            planner_audit = []
            
            for p_new, diff_log, status, planner_event in planner_results:
                planner_audit.append(planner_event)
                if status == "failed":
                    failed_planners.append(planner_event.get("planner", "unknown"))
                if status == "pruned":
                    pruned_count += 1
                if p_new is not None:
                    valid_planners.append(p_new)
                if diff_log:
                    local_diff_logs.append(diff_log)

            # 🛡️ 局部持久化事务重构：只有当所有 planner 全盘失败，才执行行级别 rollback
            if len(valid_planners) == 0 and planners:
                logger.warning(f"↩️ Rollback Line {line_idx+1}: All planners failed in purification.")
                return None if PURIFY_DELETE_ON_FAIL else line_str
                
            # 部分 planner 失败：保留成功提纯的 planner，将失败的隔离在日志中供后续重试
            if failed_planners:
                logger.warning(f"⚠️ Partial Success Line {line_idx+1}: Planners {failed_planners} failed, but preserved {len(valid_planners)} successful planners.")
                record_audit_event({
                    "line_number": line_idx + 1,
                    "run_status": "partial_success",
                    "reason": f"planners {failed_planners} failed, but salvaged others",
                    "failed_planners": failed_planners,
                    **raw_mapping_meta
                })
```

---

### 方案 8. CPU 本地前置事实碰撞与幻觉查杀器 (Deterministic CPU-based Fact Pre-check)
在调用大模型裁判前，提取生成内容中的专有医药名词、推荐药物、数值剂量等核心指针，在 CPU 本地对 `refs` 事实库进行快速的布尔碰撞与轻量向量核验，前置拦截幻觉。

#### 💡 代码层修改 blueprint (core/purification_helper.py)：
```python
def check_factual_grounding(text: str, refs: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    CPU 级确定性前置事实校验：
    1. 提取生成内容中的药物实体及数值剂量。
    2. 硬比对 refs 事实包，若出现 refs 以外的幻觉药物实体，直接秒级拦截。
    """
    if not text or not refs:
        return True, ""
        
    # 提取 refs 中的所有纯文本事实
    raw_refs_pool = " ".join([r.get("context", "") for r in refs if isinstance(r, dict)]).lower()
    
    # 提取 text 中的潜在医学药物实体 (这里可以使用预设的药物词表或正则)
    # 示例：快速提取常见高频专有名词
    extracted_entities = re.findall(r'[a-zA-Z\u4e00-\u9fa5]+(?:片|注射液|胶囊|颗粒|软膏)', text)
    
    # 同义词对撞表
    SYNONYM_MAP = {
        "吉三代": "索磷布韦维帕他韦",
        "艾恒平": "布比卡因脂质体"
    }
    
    for entity in extracted_entities:
        entity_lower = entity.lower()
        # 转换为标准词或原词进行碰撞
        standard_entity = SYNONYM_MAP.get(entity_lower, entity_lower)
        if standard_entity not in raw_refs_pool and entity_lower not in raw_refs_pool:
            return False, f"检测到幻觉实体 '{entity}' 未在参考资料中出现，拒绝进入裁判打分。"
            
    return True, ""
```

---

### 方案 9. 带向诊断反馈的自愈重试机制 (Feedback-driven Self-Healing Retry Loop)
重构 `purify_single_think` 极其重试调用链，在 structured output 重试或 Judge 失败时，捕获异常与具体扣分维度原因，作为 Diagnostic Feedback 注入重试 Prompt，实现带反馈的有向纠偏。

#### 💡 代码层修改 blueprint (core/purification_engine.py)：
```python
# core/purification_engine.py

         # 局部带诊断反馈的重试机制
         feedback_log = ""
         for attempt in range(max_retries):
             feedback_prompt_addition = ""
             if feedback_log:
                 feedback_prompt_addition = f"\n\n【⚠️ 上一轮重写失败反馈】：\n{feedback_log}\n请针对性修正上述问题，避免再次犯错！"
                 
             prompt = f"""{few_shot}
             
             ### 系统指令：
             ... {feedback_prompt_addition}"""
             
             # 调用生成...
             # 如果裁判打分不合格：
             if not is_passed:
                 feedback_log = f"裁判打分未通过。原因: {judge_reason}. 扣分维度: {failed_dimensions}."
                 # 循环进入下一轮带 feedback 重试
```

---

### 方案 10. 智能复杂度路由与多级分流 (Smart Intent Routing)
重构 pipeline 任务规划器，识别问题意图与事实覆盖度，为简单 facts 题自动降级单切面并分流至 lightweight 模型，减少 premium API 的冗余消耗。

#### 💡 代码层修改 blueprint (core/pipeline_workflow.py)：
```python
# core/pipeline_workflow.py

    def _estimate_complexity(self, q: str, refs: List[Dict[str, Any]]) -> str:
        """
        根据 Query 及事实包估算问题复杂度：
        - 包含“趋势、机制、诊断、联合用药、相互作用”为 HIGH
        - 仅包含单一适应症、贮藏、包装等事实查询为 LOW
        """
        complexity_keywords = ["机制", "药理", "联合", "相互作用", "代谢", "通路", "不良反应"]
        if any(kw in q for kw in complexity_keywords):
            return "HIGH"
        if len(refs) <= 2:
            return "LOW"
        return "HIGH"

    async def route_task(self, q: str, refs: List[Dict[str, Any]]):
        complexity = self._estimate_complexity(q, refs)
        if complexity == "LOW":
            # 路由至简单模式：跳过 8-facet 并行规划，指派给轻量级模型快速生成单切面
            return await self.api_client.call_lightweight_model(q, refs)
        else:
            # 路由至复杂模式：进入多切面 premium 并发
            return await self.api_client.call_premium_model_pipeline(q, refs)
```

---

### 方案 11. 基于 FAISS 的认知经验与动作缓存 (Episodic Memory & Action Cache)
引入本地向量库，缓存高评分提纯动作。在新样本生成时进行语义相似度检索，召回做 Dynamic Few-Shot。

#### 💡 代码层修改 blueprint (services/healing_service.py)：
```python
# services/healing_service.py
import faiss
import numpy as np

class ActionCacheManager:
    def __init__(self):
        # 初始化 FAISS 本地向量库
        self.dimension = 768
        self.index = faiss.IndexFlatL2(self.dimension)
        self.cache_data = [] # 存放 {q: ..., purified_think: ..., score: ...}

    def add_to_memory(self, q: str, purified_think: str, score: float, vector: np.ndarray):
        """仅在提纯得分 > 9.0 时，才允许固化进长期认知经验库"""
        if score >= 9.0:
            self.index.add(np.array([vector]))
            self.cache_data.append({
                "q": q,
                "purified_think": purified_think,
                "score": score
            })

    def retrieve_few_shot(self, query_vector: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(np.array([query_vector]), top_k)
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.cache_data):
                results.append(self.cache_data[idx])
        return results
```

---

### 方案 12. 外部文献 API 双层限流与 429 自愈重试机制 (PubMed API Dual-Layer Throttling & 429 Self-Healing Retry)
针对 PubMed 等外部医学数据库的高频并发请求极易导致 HTTP 429 (Too Many Requests) 熔断的问题，我们引入双层控频防御与自动避让自愈算法，保护链路稳定性。

#### 💡 代码层修改 blueprint (retrieval/api_gateway.py)：
```python
# retrieval/api_gateway.py
import asyncio
import urllib.request
import urllib.parse
import urllib.error
import json
from typing import Dict, Any, List

class AsyncRateLimiter:
    def __init__(self, max_rate: float, time_period: float = 1.0):
        self.max_rate = max_rate
        self.time_period = time_period
        self.tokens = max_rate
        self.last_update = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            while True:
                now = asyncio.get_event_loop().time()
                elapsed = now - self.last_update
                # 动态补充令牌
                self.tokens = min(self.max_rate, self.tokens + elapsed * (self.max_rate / self.time_period))
                self.last_update = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                else:
                    wait_time = (1.0 - self.tokens) * (self.time_period / self.max_rate)
                    await asyncio.sleep(wait_time)

# 在 APIGatewayService 中调用：
# self.pubmed_api_key = os.getenv("PUBMED_API_KEY", "").strip()
# pubmed_rate_limit = float(os.getenv("PUBMED_RATE_LIMIT", "10"))
# self.rate_limiter = AsyncRateLimiter(max_rate=pubmed_rate_limit, time_period=1.0)

async def fetch_url_with_retry(base_url: str, params: Dict[str, Any], rate_limiter: AsyncRateLimiter) -> Dict[str, Any]:
    max_retries = 3
    backoff = 1.0
    loop = asyncio.get_event_loop()
    
    for attempt in range(max_retries):
        # 客户端令牌桶速率限流锁止 (硬限额，如 10 rps)
        await rate_limiter.acquire()
        try:
            def do_request():
                url = f"{base_url}?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=3.0) as r:
                    return json.loads(r.read().decode('utf-8'))
            return await loop.run_in_executor(None, do_request)
        except urllib.error.HTTPError as he:
            if he.code == 429:
                # 429 专属自愈等待：优先解析 Header 中的 Retry-After 避让时间，否则按指数退避递增
                retry_after = he.headers.get("Retry-After")
                wait_sec = float(retry_after) if (retry_after and retry_after.isdigit()) else backoff
                logger.warning(f"PubMed API HTTP 429 (Too Many Requests). Attempt {attempt+1}/{max_retries}. Waiting {wait_sec}s...")
                await asyncio.sleep(wait_sec)
                backoff *= 2.0
            else:
                raise he
    raise Exception("Failed to fetch from PubMed API due to persistent HTTP 429 rate limit errors.")
```

---

### 方案 13. 中文检索实体翻译与学术对齐 (Chinese Query Translation Alignment)
针对 PubMed 不支持中文导致检索落空，或者字符半部分泄漏导致匹配无关量子点/蜥蜴论文的问题，我们设计并实现双通道翻译对齐机制。

#### 💡 代码层修改 blueprint (retrieval/api_gateway.py)：
```python
# retrieval/api_gateway.py
import os
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("MedicalQA.APIGateway")

# 1. 通道一：静态中英医学词典对照表
STATIC_TRANSLATION_MAP = {
    "布洛芬": "Ibuprofen",
    "甲氨蝶呤": "Methotrexate",
    "西尼莫德": "Siponimod",
    "肾脏": "Kidney",
    "有机阴离子转运体（hoat家族）": "Organic Anion Transporters",
    "hoat家族": "OAT transporter",
    "hOAT家族": "OAT transporter",
    "螺旋藻胶囊": "Spirulina capsule",
}

async def translate_entity_to_english(entity_name: str, api_client: Any = None) -> str:
    """
    双通道中英医疗术语对齐：
    1. 优先检索本地静态高频词表
    2. 词表未命中时，调用轻量大模型翻译为标准英文医学术语（通用药名/疾病名/基因符号）
    """
    clean_name = entity_name.strip()
    if not clean_name:
        return ""
        
    # 通道一：本地对照映射
    if clean_name in STATIC_TRANSLATION_MAP:
        translated = STATIC_TRANSLATION_MAP[clean_name]
        logger.info(f"Translation HIT: Local Map matched '{clean_name}' -> '{translated}'")
        return translated
        
    # 通道二：LLM 医疗术语路由器对齐
    if api_client is None:
        # 降级：如果未传入 api_client，剔除所有中文仅保留英文字符作为安全检索词
        english_only = "".join(re.findall(r'[a-zA-Z0-9\s\-\*\/]+', clean_name)).strip()
        logger.warning(f"Translation Fallback: No LLM client. Strip Chinese from '{clean_name}' -> '{english_only}'")
        return english_only

    prompt = f"""你是一个专业的医学翻译助手。
请将中文医学实体翻译为 PubMed 文献数据库检索最常用、最标准的英文医学术语（如标准通用药物英文名、疾病 MeSH 词或基因/转运体缩写符号）。
【⚠️输出硬性要求】：你必须且只能输出翻译后的英文词，严禁包含任何中文、解释、拼音、括号、引号或标点符号。

待翻译实体：【{clean_name}】
翻译英文词："""

    try:
        # 使用 lightweight 级别大模型完成秒级学术对齐
        response = await api_client.call_llm(
            prompt,
            model_pool="lightweight",
            stage=f"PubMed检索实体学术对齐: {clean_name}"
        )
        translated = response.strip().strip('"').strip("'").strip()
        
        # 过滤可能残留的多余文本或解释
        translated = re.sub(r'[^a-zA-Z0-9\s\-\*\/]', '', translated).strip()
        
        logger.info(f"Translation LLM: Aligned '{clean_name}' -> '{translated}'")
        return translated if translated else clean_name
    except Exception as e:
        logger.error(f"Failed to translate entity '{clean_name}' via LLM: {e}")
        # 异常安全退避：提取英文字符
        return "".join(re.findall(r'[a-zA-Z0-9\s\-\*\/]+', clean_name)).strip()
```



