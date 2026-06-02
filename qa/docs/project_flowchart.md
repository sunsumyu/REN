# 🩺 医疗多视角思维链（CoT）数据合成与净化项目系统架构流程图

本模块文档对本项目（`Medical QA Facet Synthesis & CoT Purification`）的双阶段核心流转架构进行深度解构，并使用 **Mermaid** 绘制了精美的系统级数据流向与控制流图，存入 `docs` 目录作为项目的官方技术规格说明。

---

## 📊 1. 系统全局双轨运行图 (Global System Architecture)

本系统采用**双轨合流设计**：**第一轨（Generation Pipeline）**负责基于知识图谱与多视角 Agent 仿真合成多轮对话数据集；**第二轨（Purification Pipeline）**负责离线大模型思维链（CoT）语义提纯、学术对齐与 3D 质量门禁校验。

```mermaid
graph TD
    %% 阶段样式定义
    classDef genStage fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;
    classDef purStage fill:#eceff1,stroke:#37474f,stroke-width:2px,color:#263238;
    classDef storage fill:#fff8e1,stroke:#f57f17,stroke-width:2px,color:#e65100;
    classDef gate fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c;

    %% ------------------ 阶段一：数据集生成轨 ------------------
    subgraph Stage1["第一阶段：多视角多轮数据集仿真合成管线 (Generation)"]
        A["知识图谱与文献抽取 (qa_datasets.db)"] -->|Retrieval| B["初始问题生成 (Question Creator)"]
        B --> C["多切面视角规划 (Facet Planner)"]
        C --> D["视角合并、扩展与消冗 (Expander & Redundance Detector)"]
        D -->|并发分发| E["多切面 LLM Agent 仿真生成 (CoT + Response)"]
        E --> F["多视角最终答案综合总结 (Synthesis Agent)"]
        F --> G["多轮追问演进 (Next Question Generator)"]
    end
    class Stage1 genStage;

    %% ------------------ 存储落盘 ------------------
    G -->|追加写入| H[("原始数据集 (medical_qa_dataset.jsonl)")]
    class H storage;

    %% ------------------ 阶段二：思维链提纯轨 ------------------
    subgraph Stage2["第二阶段：思维链提纯、学术自愈与质量门禁拦截 (Purification)"]
        I["读取增量记录"] -->|动态行号| J["AI 视角安全校验网关 (Small Model Validator)"]
        J -->|重置占位符| K["前置物理去噪过滤 (pre_strip_engineering_noise)"]
        K --> L["循证医学刚性事实锚点注入 (refs fact-anchoring)"]
        L --> M["LLM 探索性 CoT 提纯重写 (PurificationEngine)"]
        M --> N["后置元描述与过渡词平滑 (post_strip_meta)"]
        N --> O["学术实体与病生理常识自愈 (HealingService)"]
        O --> P{"3-D 质量裁判门禁评分 (LLMJudgeStrategy)"}
        P -->|不达标 & < 3次重试| Q["反馈控制环路 (De-contaminated Feedback)"]
        Q -->|重写调整| M
        P -->|一票否决/超限| R["❌ 物理拦截并无情丢弃 (Drop Facet)"]
        P -->|✅ 达标通过| S["最终高熵纯净微调数据集 (medical_qa_dataset.jsonl)"]
    end
    class Stage2 purStage;
    class P gate;
    class R gate;
    class S storage;

    H -->|增量触发| I
```

---

## 🛠️ 2. 子系统流程详析 (Sub-System Deep Dive)

### 2.1 数据集仿真合成控制流 (Generation Pipeline Workflow)
第一阶段的控制流主要由 `pipeline_workflow.py` 驱动，实现了从非结构化文献图谱向结构化多视角 CoT 对话的演进：

```mermaid
sequenceDiagram
    autonumber
    participant App as main.py / run.bat
    participant WF as pipeline_workflow.py
    participant DB as Knowledge Database (SQLite)
    participant LLM as LLMService (premium/lightweight)
    participant File as medical_qa_dataset.jsonl

    App->>WF: 启动主生成循环
    WF->>DB: 根据种子实体提取文献 (context_list / refs)
    DB-->>WF: 返回图谱关系与说明书事实
    WF->>LLM: 初始问题生成 (Question Creator)
    LLM-->>WF: 返回核心问题 Q
    WF->>LLM: 规划回答切面 (Facet Planner)
    LLM-->>WF: 返回切面列表 (如: [药理机制, 用药禁忌])
    
    rect rgb(240, 248, 255)
        note right of WF: 多 Agent 并发执行
        loop 遍历切面 (Facets)
            WF->>LLM: 并发生成切面的 CoT 与回答
            LLM-->>WF: 返回结构化 choices (answer)
        end
    end

    WF->>LLM: 多视角综合总结 (Synthesis Agent)
    LLM-->>WF: 返回去重后的完美正文
    WF->>LLM: 多轮追问进化 (Next Question Generator)
    LLM-->>WF: 返回下一轮深入问题
    WF->>File: 剔除工程元数据后，追加追加写入磁盘
```

---

### 2.2 思维链语义提纯控制流 (CoT Purification Workflow)
第二阶段的控制流由 `medicalqa_purifier.py` 驱动，是保障生成数据可直接用于顶尖推理大模型（如 DeepSeek-R1、o1 等）微调的**工业级数据质检门禁**：

```mermaid
flowchart TD
    %% 样式
    style Start fill:#f9f9f9,stroke:#333,stroke-width:2px;
    style End fill:#f9f9f9,stroke:#333,stroke-width:2px;
    style Logic fill:#e1f5fe,stroke:#0288d1,stroke-width:1px;
    style Loop fill:#fffde7,stroke:#fbc02d,stroke-width:1px;

    Start([开始提纯一轮数据]) --> J1{{"AI 视角语义网关校验 (verify_facet)"}}
    
    J1 -- INVALID (占位符或报错) --> J2["重置为合法学术切面\n'临床用药安全'"]
    J1 -- VALID --> K["前置物理去噪过滤\n(擦除原始JSON、RAG字样)"]
    J2 --> K
    
    K --> L["事实白名单锚点注入\n(refs 刚性事实绑定)"]
    
    subgraph Loop ["反馈控制重试环 (最多3次)"]
        M["LLM 提纯重写\n(动态认知推理流重构)"] --> N["学术实体与病生理常识自愈\n(HealingService)"]
        N --> O["本地物理平滑\n(擦除'首先/其次/综上'等序号过渡词)"]
        O --> P["3-D 裁判打分\n(Purity/Rigor/Depth)"]
    end
    
    L --> M
    
    P --> Q{"是否达标且未发生\n结构/安全泄漏？"}
    Q -- YES (达标通过) --> S["✅ 高熵纯净思维链落盘写入"]
    Q -- NO & 重试次数 < 3 --> R["注入不达标报错反馈\n(Feedback Loop)"]
    R --> M
    
    Q -- NO & 超出最大重试 --> T{"判断是否触发\n防拷贝幻觉绕过？"}
    T -- YES (存在严重工程垃圾) --> U["❌ 物理强行抛弃该切面 (Drop Facet)"]
    T -- NO (无污染但质量一般) --> V["🛡️ 降级退回极简自愈兜底文本"]
    V --> S
    U --> End([结束当前行提纯])
    S --> End
```

---

## 💎 3. 核心质量门禁指标 (Quality Gate Metrics)

项目通过 `LLMJudgeStrategy` 对思维链重构实施三维质检标准：

| 评估维度 (Dimension) | 达标分数线 | 核心惩罚红线 (One-strike Penalty) | 优化处理机制 |
| :--- | :--- | :--- | :--- |
| **🟢 语义纯净度** | **85 / 100** | 包含 `JSON`、`Schema`、`refs` 等工程元数据；使用“首先、其次”等阶梯序号或出现“我将从以下切面回答”等元叙述。 | 一票否决降至 60 分以下，启动反馈控制环重写，若三次失败则**直接丢弃**。 |
| **🩺 医学严谨度** | **90 / 100** | 与知识图谱 `refs` 事实（受体、基因型、发生率）冲突；或无确切文献公理支撑下虚构分子骨架/特异性结合效能。 | 图谱强Fact一致性对齐，触发伪学术幻觉一票否决扣至 50 分以下。 |
| **🧠 逻辑深度** | **85 / 100** | 通篇为平淡无奇的说明书说明文，无动态摩擦词（“既然...必然...”），无探究性疑问反思（缺乏“?”疑问锚点）。 | 强制字数限制（需 > 150字），推导极简化直接惩罚扣分，通过重写激活参数化临床知识。 |

---

## 🎯 4. 数据提纯的学术理念

项目的提纯净化拒绝简单的“文本剔除”，而是为了生成最符合人类顶尖医生大脑认知的思维链。
在最终的 `medical_qa_dataset.jsonl` 中，净化后的思维链展现了经典的**五阶段动态临床推理心流**：

$$\text{核心矛盾解构} \xrightarrow{\text{因果推演}} \text{微观病生理逻辑} \xrightarrow{\text{假说排查}} \text{分叉与特殊情况} \xrightarrow{\text{核准校对}} \text{生理/安全极限} \xrightarrow{\text{自然合拢}} \text{决策得出}$$

这套高度规范的流程图已与目前最新的代码库保持 100% 同步，并作为指导本项目未来工程化、工业级数据微调准备的重要技术规范。
