from jinja2 import Template

# 1. 问题创造者 (Question Creator) - 默认自引导模版
_BOOTSTRAP_QUESTION_CREATOR_TEMPLATE = """<role>
问题创造者（Question Creator）
</role>

<task>
你会收到一个上下文（context，可能由多段资料/片段组成）。你的任务是：严格基于这些上下文创造若干“可被上下文回答”的高质量问题。任何问题的答案都必须能在这些上下文中直接找到或通过合理归纳得到。每次生成三个左右的不同问题。
</task>

<rule>
- 只根据 context 造问题：不得依赖常识补全、不得引入 context 之外的新事实/新实体/新结论。
- 每个问题都必须可被 context 支撑：能在 context 中定位到明确依据（原句或可归纳的信息）。
- 问题必须独立表述：不要出现“根据以上/上述/这段话/文中”等指代性措辞。
- 不要生成需要外部资料才能回答的问题（如“最新进展”“现实世界数据”“超出材料的推测”）。
- 避免重复：同义改写视为重复；同一答案的不同问法只保留一个。
- 避免泄露过程：不要在问题中暗示“上下文提到/资料显示/检索到”等措辞。
- **多问题低交集**：不同问题尽量对应 context 的不同信息点/不同段落/不同概念；尽量避免共享同一核心答案或大量重叠依据。
- 若 context 信息不足以生成合格且低交集的问题集合，可以只生成一个问题，但是还是以数组的格式返回。
</rule>

<format>
- 只输出一个 JSON 数组（不要额外文字、不要解释、不要代码块标注）。
- 数组元素为字符串，每个字符串是一条问题。
- 示例：["问题1","问题2","问题3"]
</format>

<strategy>
- 先通读 context，将信息点按“主题簇”划分（定义/条件/流程/对比/机制/风险/限制/建议/例外/指标等）。
- 优先选择不同主题簇各出 1 题，确保问题之间的答案依据尽量不重叠。
- 优先覆盖高信息增益的问题：可验证、可定位、信息密度高、能区分概念边界。
- 自检：任意两道题若核心答案高度相同或主要依据重叠，则合并/替换其中一道，直到交集最小。
</strategy>

## 输入
context: [
{% for item in context_list %}
  { "context": "{{ item.context }}", "source": "{{ item.source }}" },
{% endfor %}
]

## 输出
- 只输出 JSON 数组
"""

# 2. 角度规划者 (Facet Planner) - 默认自引导模版
_BOOTSTRAP_FACET_PLANNER_TEMPLATE = """<role>
问题角度规划者（Facet Planner）
</role>

<task>
用户会给出一个 query。你需要为这个 query 规划“回答的侧重点（facet）”，用于分发给下游多个 agent 进行角色扮演式的完整回答。
</task>

<rule>
- 你只输出“角度（facet）”，不输出真实答案、不展开解释。
- facet 不是把问题拆成多个子问题，而是“同一个问题的不同叙事重心/回答框架”。
- 每一个 facet 都必须能引导出一篇**针对该 query 的完整回答**（而不是片面补充）。
- facet 之间要尽量互相区分，避免同义重复。
- facet 形式要尽量简短：**一个词或短词组**，不要一句话，更不要一段话。
- facet 数量由你根据 query 复杂度决定：可以 1 个，也可以多个。
</rule>

<format>
- 只输出一个 JSON 数组（不要额外文字、不要代码块标注、不要解释）。
- 数组元素为字符串，每个字符串就是一个 facet。
- 示例格式：[".....", ".....", "....."]
</format>

<strategy>
- 必须要能形成完整叙述的“回答框架”，例如：技术实现、应用落地、决策建议、对比评估、风险治理、方法论抽象、产品化视角、成本收益等。
- 如果 query 很简单：给 1-2 个 facet。
- 如果 query 很复杂：给 2-8 个 facet。
- 不要提出反问或澄清问题，直接给 facets。
</strategy>

## 输入
query: {{ query }}

## 输出
"""

# 3. 角度补充者 (Facet Expander) - 默认自引导模版
_BOOTSTRAP_FACET_EXPANDER_TEMPLATE = """<role>
Facet Expander
</role>

<task>
给定用户的 query 和上游已提供的 facets，你只负责“补充新增 facets”。从“回答框架/叙事重心”角度，找出上游未覆盖的视角，再补充 0~6 个。
</task>

<rule>
- 只返回新增 facet：输出中不得包含任何已给出的 facet（完全重复或近义都不允许）。
- 不得重排/复述/改写已有 facets：不要为了“更好看”而重新输出一遍。
- 如果没有可补充的，就返回 **空数组**：[]
</rule>

<format>
- facet 必须是一个词或短词组（越短越好）。
- 每个 facet 都要能引导出一篇完整回答，而不是零散细节。
- facets 之间尽量区分，避免同义重复。
- 只输出一个 JSON 数组（不要额外文字、不要解释、不要代码块标注）。
</format>

## 输入
- query: {{ query }}
- facets: {{ facets | tojson }}

## 输出
"""

# 4. 角度筛选者 (Facet Reducer) - 默认自引导模版
_BOOTSTRAP_FACET_REDUCER_TEMPLATE = """<role>
Facet Reducer
</role>

<task>
给定用户的 query 和一组 facets（数量通常超过 8 个），你需要从中筛选出最重要的 8 个 facets，用于下游 agent 分发。输出应覆盖该 query 的主要回答框架，尽量避免重复与边缘角度。
</task>

<rule>
- 你只输出“角度（facet）”，不输出真实答案、不展开解释。
- 只能从输入 facets 中挑选，不得新增、不得改写、不得合并生成新 facet。
- 必须严格输出 8 个 facet。
- facet 之间要尽量互相区分，避免同义重复；若有重复/近义项，只保留表达更通用、更高覆盖的那个。
</rule>

<format>
- 只输出一个 JSON 数组（不要额外文字、不要解释、不要代码块标注）。
- 数组元素为字符串，每个字符串就是一个 facet。
- 示例格式：[".....", ".....", ".....", ".....", ".....", ".....", ".....", "....."] (共8个元素)
</format>

<strategy>
- 优先保留覆盖面更广、能独立形成完整回答框架的 facets。
- 优先保留高价值维度。
- 如果出现多个候选 facet 指向同一叙事重心，只保留一个最代表性的。
</strategy>

## 输入
- query: {{ query }}
- facets: {{ facets | tojson }}

## 输出
"""

# 5. 按角度回答 (FacetGraph-QA Agent) - 默认自引导模版
_BOOTSTRAP_FACET_QA_TEMPLATE = """# FacetGraph-QA Agent（Synthesis / Evidence-First）

你是一个“Facet 驱动的知识图谱问答生成器（FacetGraph-QA Agent）”。你的任务是：在给定问题 **Q** 与**facet（回答出发点/侧重点）** 的同时，还会给你一组与问题高度相关的 **参考资料包（refs）**。

你需要生成一份**完整且高质量**的回答，并输出**可检查的显式证据链与推理过程**用于下游训练。
重要：无论在思考还是最终回答中，都不要显式描述“你去查了什么/检索了什么/工具返回了什么”。你应当像天然掌握这些知识一样陈述，但仍保留“出处式”的证据表达与可核验的定位信息。

---

## 1) 输入说明
给定三个核心输入：
- **Q**：用户问题（必须完整回答）
- **facet**：回答的“出发点与侧重点”（是完整解题视角下的组织与强调方式，而不是缩减问题范围）
- **refs**：与问题相关的参考资料包（造题时参考的资料，可能包含：文档片段、条款摘录、结构化记录、术语定义、表格、节点说明等；每条 refs 通常带有来源名与定位信息）

---

## 2) 知识使用原则（强制优先级）
你必须遵循以下优先级来构建答案与证据链：
1. **主要参考refs**：refs通常已覆盖问题的主干事实、核心定义与关键结论依据（但也有可能信息还不够充分）。
2. **GraphTool 用于补充**：当 refs 缺漏、存在歧义、需要更完整的边界条件/对比/风险说明时，用图谱补全。
3. **冲突处理**：若 refs 与图谱证据冲突，优先按“权威性/时效性/适用范围/来源稳定性”裁决；必要时给出并列解释与条件化结论。

---

## 3) 总目标（必须同时满足）
你必须同时满足以下目标：
1. **回答必须完整**：覆盖 Q 的关键维度，不允许因为 facet 而变成片面回答。
2. **facet 只是侧重点**：用 facet 来组织结构、强调重点与结论顺序，但不能遗漏 Q 的应答要点。
3. **证据链必须显式**：每个关键结论都要可追溯到明确证据（refs / 图谱记录 / 规范条款等），且能定位。
4. **推理必须可检查**：写清楚“哪些证据 → 哪些中间结论 → 最终结论”。

---

## 4) 工作方法（强制执行）
你必须按以下流程生成答案（输出必须符合格式）：
### Step A：问题拆解（面向验证）
- 将 Q 拆成若干可检查的子问题（定义、事实、条件、对比、风险、建议、边界条件等）
- 明确“必须回答点”和“可选加分点”
- 将 facet 映射为组织结构与强调重点（例如：风控、合规审计、临床安全、业务落地、成本收益等）

### Step B：证据对齐
- 先在 refs 中为每个子问题找到对应的依据：事实、条款、定义、记录、结论段落等
- 对每条依据提取“可引用定位信息”（来源名/章节/条款号/页码/段落号 等）
- 标注哪些结论是 refs 已充分支持的，哪些仍存在缺口

### Step C：补充完善
- 当出现以下情况时，补充证据（但输出中不描述获取过程）：
  - refs 覆盖不足：缺定义、缺条件、缺反例/边界、缺对比项、缺操作步骤、缺风险条款
  - refs 存在歧义：同名概念、口径不一致、适用范围不清
  - 需要更完整证据链：关键结论需要多跳关系或结构化记录支撑
  - 补充证据同样必须有定位信息（节点ID/关系路径ID/记录ID/来源文档定位等）

### Step D：一致性校验与冲突处理（必须写在推理中）
- 若证据之间存在冲突：指出冲突点，并说明采用哪条证据的原因（权威性/时效性/适用范围）
- 若证据不足：必须明确降级（保守结论 + 可执行的补充信息需求 + 不确定性边界）

### Step E：生成回答（证据驱动 + facet 强调）
- 关键结论必须由证据支撑
- 允许“合理推断”，但必须标注为推断，并写明推断前提来自哪些证据
- 输出需可读、结构化，便于下游 verifier 检查

---

## 5) 输出格式（严格遵循）
你必须输出两段内容：
### 第一段：<think><facet = {{ facet }}>(只放证据链与推理过程；同样禁止“获取过程痕迹”)</think>
在 <think><facet = {{ facet }}> 中写清楚：
- 问题拆解（S1/S2/...）
- 证据清单（逐条列出，包含定位信息）
- 证据到结论的推理链（逐步推导：E → C）
- 冲突识别 / 证据不足的降级策略（如有）
- 最终结论摘要（用于校验）
> 注意：<think> 内允许结构化标记（如 E1/E2...、结论C1/C2...），但绝对不要出现“我查阅/我检索/工具返回/调用接口”等过程性表达。

#### 证据引用格式（统一规范）
在 <think><facet = {{ facet }}> 内引用证据时，统一用下面格式：
- refs 证据：[证据R1：来源=refs:《文档名/记录名》，定位=具体位置，要点=...]

---

### 第二段：最终回答正文（严格“无工具痕迹”）
</think> 之后输出最终回答正文，要求：
- **严禁出现**以下表达或其同义句（无论中英文）：
  “查阅 / 检索 / 搜索 / 工具返回 / 调用接口 / 从图谱里查到 / 我查了一下 / 系统显示 / 根据检索结果 / retrieved / searched / looked up / tool says ...”
- 文风要像：**你天然掌握这些知识**，但仍然要写“出处式表达”，例如：
  ✅ “在《XX指南》第3章提到……”
  ✅ “某法规第X条明确规定……”
  ✅ “说明书【禁忌】部分写明……”
  ❌ “该记录（ID=...）指出……”
  ❌ “我去看了指南……”
  ❌ “我检索到一条记录……”
- 回答必须**完整覆盖 Q**，同时体现 facet 的组织方式与重点
- 尽量结构化输出（分点/分段/小标题），结论明确、建议可执行

---

## 6) 质量红线（任何一条触发都视为失败）
- ❌ 只围绕 facet 输出片面内容，遗漏 Q 的关键维度
- ❌ 关键结论没有证据支撑，或证据无法定位
- ❌ 证据与结论自相矛盾，不做解释
- ❌ 出现任何“检索/工具/系统返回”等过程痕迹词（包括在 <think> 内）
- ❌ 证据不足却强行给确定答案（必须降级）

---

## 输入数据
- Q: {{ query }}
- facet: {{ facet }}
- refs: {{ refs | tojson }}

## 输出要求
请直接生成包含 `<think><facet = {{ facet }}>...</think>` 推理部分以及后续正文回答的文本。
"""

# 6. 多角度回答去除冗余 (Facet Redundancy Detector) - 默认自引导模版
_BOOTSTRAP_FACET_REDUNDANCY_DETECTOR_TEMPLATE = """<role>
多角度回答冗余判别器（Facet Redundancy Detector）
</role>

<task>
你将收到：
- 一个问题 Q
- 多个“规划 + 对应回答”的组合（planners 数组）
这些回答声称来自不同角度。

你的任务是：
判断这些回答是否真的来自“不同回答框架”，还是只是“同一核心逻辑的不同表达方式”。

如果发现两个或多个回答本质相同（例如：
- 核心结论一致
- 论证路径相同
- 证据结构高度重叠
- 只是措辞变化或顺序不同
- 只是补充细节但逻辑骨架一致）
则认为它们是冗余表达。

你需要：
输出一个数组，数组中是“需要被排除的回答下标”。

注意：
- 下标从 0 开始
- 保留代表性最强、结构最完整的一份
- 删除其余冗余表达
- 只输出需要删除的下标数组
</task>

<model_constraint>
- 只输出 JSON 数组
- 只输出需要删除的下标数组
- 不要输出分析过程
- 如果没有需要删除的下标，则输出空数组
</model_constraint>

<input_format>
{
  "Q": "{{ query }}",
  "planners": [
    {% for p in planners %}
    {
      "planner": "{{ p.planner }}",
      "answer": "{{ p.answer }}"
    }{% if not loop.last %},{% endif %}
    {% endfor %}
  ]
}
</input_format>

<judging_strategy>
Step 1: 提取结构骨架
- 提取每个回答的核心结论
- 提取推理路径
- 提取证据类型
- 提取组织结构（是否风险框架 / 机制框架 / 合规框架 / 成本收益框架等）

Step 2: 判断是否为“真正不同角度”
真正不同角度应满足：
- 回答组织逻辑不同
- 强调重点不同
- 结论生成路径不同
- 关注维度不同

若满足以下任一情况，则视为冗余：
- 同一逻辑结构换说法
- 同一证据顺序重述
- 只是补充细节但框架一致
- 仅表达风格差异

Step 3: 去重原则
- 保留信息更完整的一份
- 保留证据更清晰的一份
- 保留逻辑更清楚的一份
- 删除其余重复表达
</judging_strategy>

<rule>
- 只输出 JSON 数组
- 不要解释
- 不要输出分析过程
- 不要输出保留项
- 不要输出文字说明
- 不要代码块
</rule>

<output_format>
示例：
[1,3]
表示删除下标 1 和 3 的回答
</output_format>
"""

# 7. 多答案综合总结器 (Multi-Answer Synthesis Agent) - 默认自引导模版
_BOOTSTRAP_MULTI_ANSWER_SYNTHESIS_TEMPLATE = """<role>
多答案综合总结器（Multi-Answer Synthesis Agent）
</role>

<task>
你将收到一个问题 Q，以及多个回答模型生成的完整回答（answers，为一个数组）。
你的任务是：
对这些回答进行综合分析与整合，生成一份结构清晰、信息完整、去重后的“高质量最终总结版本”。

你的输出必须：
- 基于 answers 的内容进行整合
- 不遗漏重要信息
- 不重复表达相同观点
- 若发现冲突，进行合理裁决或条件化整合
- 全部使用中文输出
</task>

<model_constraint>
- 请勿提及你调用的模型类型。
- 如果出现英文输出，必须强制要求改为中文输出
</model_constraint>

<input_format>
{
  "Q": "{{ query }}",
  "answers": {{ answers | tojson }}
}
</input_format>

<working_strategy>
Step 1：结构识别
- 提取每个回答的核心结论
- 提取支撑依据（法规条款/说明书段落/统计结果/实验数据等）
- 标记各回答的侧重点（风险、机制、流程、对比、建议等）

Step 2：信息融合
- 合并相同或高度重叠的信息（去重）
- 保留表达更完整、证据更充分的版本
- 将多个回答中的补充信息整合进统一结构

Step 3：冲突处理
- 若不同回答存在冲突：
  - 优先选择证据更充分/引用更明确的版本
  - 或给出条件化说明（在A条件下……；在B情况下……）
  - 若无法裁决，说明存在不同观点并标注适用条件

Step 4：结构重构
输出必须采用统一结构：
1. 结论概览（简洁明确）
2. 核心依据整合（引用式表达，指明出处章节/条目）
3. 完整展开说明（覆盖问题所有必答点）
4. 风险与边界条件
5. 实务/操作建议（如适用）
6. 若存在不确定性 → 说明证据边界

Step 5：质量自检
- 是否覆盖所有回答中的关键观点？
- 是否避免重复？
- 是否存在逻辑冲突？
- 是否保持证据表达？
- 是否完全中文？
</working_strategy>

<rule>
- 不要输出数组
- 不要引用“回答1/回答2”字样
- 不要提及“模型说”
- 不要暴露分析过程
- 不要输出思考链
- 不要出现英文
- 输出必须是一份自然流畅的最终总结版本
</rule>

<output_requirement>
输出：一份结构清晰、整合后的最终答案（非数组，直接输出排版好的正文即可）
</output_requirement>
"""

# 8. 多轮下一问题生成器 (Next Question Generator) - 默认自引导模版
_BOOTSTRAP_NEXT_QUESTION_TEMPLATE = """<role>
多轮对话下一问创造者 (Next Question Generator)
</role>

<task>
基于已有的医疗问答对话历史（history）以及最新一轮的回答总结（summary），从当前上下文中生成下一个逻辑连贯、深入的临床、法规、技术或应用层面的问题。
</task>

<rule>
- 必须基于当前 context 进行追问，不要超出 context 范围。
- 必须是一个逻辑连贯的高质量医疗或法规问题。
- 不能与历史问题同义或高度重叠。
- 必须表述独立完整，不得包含“那么上面提到的”、“针对你刚才说的”等依赖前文代词的词汇。
- 不要生成任何解释或额外文字，直接输出问题文本本身。
</rule>

## 上下文 context
context: [
{% for item in context_list %}
  { "context": "{{ item.context }}", "source": "{{ item.source }}" },
{% endfor %}
]

## 对话历史 history
{% for r in history %}
轮数 {{ loop.index }}:
问: {{ r.Q }}
答: {{ r.summary }}
{% endfor %}

## 本轮回答总结 summary
{{ summary }}

## 输出要求
直接输出下一个问题文本，不要有任何 JSON 格式包裹，不要有代码块。
"""

def render_prompt(template_str: str, **kwargs) -> str:
    """
    Renders a prompt template using Jinja2.
    """
    t = Template(template_str)
    return t.render(**kwargs)


# ==========================================
# 提示词版本管理器透明集成层
# ==========================================
from prompt_manager import PromptManager

_manager = PromptManager()

PROMPT_NAMES = [
    "QUESTION_CREATOR_TEMPLATE", 
    "FACET_PLANNER_TEMPLATE",
    "FACET_EXPANDER_TEMPLATE", 
    "FACET_REDUCER_TEMPLATE",
    "FACET_QA_TEMPLATE", 
    "FACET_REDUNDANCY_DETECTOR_TEMPLATE",
    "MULTI_ANSWER_SYNTHESIS_TEMPLATE", 
    "NEXT_QUESTION_TEMPLATE"
]

# 自动引导灌入默认版本
for name in PROMPT_NAMES:
    default_val = globals().get(f"_BOOTSTRAP_{name}")
    if default_val:
        _manager.register_prompt(name, default_val)

def __getattr__(name: str) -> str:
    """
    拦截对模板的变量读取，自动从数据库获取当前被激活的提示词。
    符合 Python 3.7+ 模块级动态属性规范。
    """
    if name in PROMPT_NAMES:
        return _manager.get_prompt(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def update_prompt(name: str, new_content: str, description: str = "") -> int:
    """
    保存并激活某个提示词的新版本，旧版本自动退役为历史记录。
    """
    if name not in PROMPT_NAMES:
        raise ValueError(f"Unknown prompt name: {name}. Must be one of {PROMPT_NAMES}")
    return _manager.save_new_version(name, new_content, description)

def rollback_prompt(name: str, version: int) -> bool:
    """
    将某个提示词回滚并重新激活历史指定版本。
    """
    if name not in PROMPT_NAMES:
        raise ValueError(f"Unknown prompt name: {name}")
    return _manager.rollback_to_version(name, version)

def list_prompt_versions(name: str) -> list:
    """
    查询某个提示词的完整历史版本清单。
    """
    if name not in PROMPT_NAMES:
        raise ValueError(f"Unknown prompt name: {name}")
    return _manager.list_versions(name)
