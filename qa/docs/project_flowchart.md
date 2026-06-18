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
        I["读取增量记录"] -->|输入| FPB["事实抽取与分类 (FactPack Builder)"]
        FPB -->|生成 CleanedFact | ESR["精细化证据分级路由 (Evidence Scope Router)"]
        ESR -->|CORE / BOUNDARY / BLOCKED 分流| AR["回答正文重写与事实校验 (Answer Body Rewriter)"]
        AR -->|生成 narrowed_answer 边界| CP["探索性 CoT 提纯重写 (CoT Purifier)"]
        CP -->|输出 think, answer, summary| VAL["本地多层防御校验 (Validators)"]
        subgraph VAL ["本地多层防御校验 (Validators)"]
            SG["结构感知网关 (Structure Gate)"]
            LS["全字段泄漏扫描 (Leakage Scanner)"]
            SW["语义自愈与冒号平滑 (Semantic Wash)"]
        end
        VAL -->|清洗与格式核验通过| JDG{"循证参考引导裁判 (Reference-Guided Judge)"}
        JDG -->|不达标 & 重试 < 3| FB["反馈控制环路 (Feedback Loop)"]
        FB -->|重写指导| CP
        JDG -->|最终评分与裁决| TM["局部失败隔离与事务管理 (Transaction Manager)"]
        TM -->|全部失败 / 严重泄露| R["❌ 物理拦截并无情丢弃 (Rollback / Quarantine)"]
        TM -->|全切面成功 / 部分切面成功| S["最终高熵纯净微调数据集 (medical_qa_dataset.jsonl)"]
    end
    class Stage2 purStage;
    class JDG gate;
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

    Start([开始提纯当前行数据]) --> F1["抽取干净事实与脱敏引用 (FactPack Builder)"]
    F1 --> F2["划分证据意图与封锁路由 (Evidence Scope Router)"]
    F2 --> F3["回答正文窄域重写与事实校验 (Answer Body Rewriter)"]
    
    subgraph Loop ["CoT 提纯重试环 (最多 3 次)"]
        M["CoT 提纯重写 (CoT Purifier)"] --> SG["结构感知网关校验 (Structure Gate)"]
        SG -- YES (检测到JSON泄露) --> R["反馈重试"]
        SG -- NO (通过) --> LS["全字段泄漏扫描 (Leakage Scanner)"]
        LS -- YES (高置信禁词) --> R
        LS -- NO (通过) --> SW["中性语义清洗与自愈 (Semantic Wash)"]
        SW --> JDG["循证裁判打分 (Reference-Guided Judge)"]
    end
    
    F3 -->|传入 narrowed_answer 边界| M
    
    JDG --> Q{"三维指标是否达标？<br/>(Purity/Rigor/Depth)"}
    Q -- YES (达标通过) --> TM["局部失败隔离与保存 (Transaction Manager)"]
    Q -- NO & 重试次数 < 3 --> R
    R -->|注入报错反馈| M
    
    Q -- NO & 超出最大重试 --> TM
    
    TM --> TM_Check{"是否所有 Planner 全盘失败<br/>或核心切面彻底损坏？"}
    TM_Check -- YES (严重损坏) --> Rollback["↩️ 整行回滚隔离 (Rollback / Quarantine)"]
    TM_Check -- NO (局部或全部成功) --> Salvage["✅ 部分/全部成功切面持久化 (partial_success)"]
    
    Salvage --> End([结束当前行提纯])
    Rollback --> End
```

---

## 💎 3. 核心质量门禁指标 (Quality Gate Metrics)

项目通过 `LLMJudgeStrategy` 对思维链重构实施三维质检标准：

| 评估维度 (Dimension) | 达标分数线 | 核心惩罚红线 (One-strike Penalty) | 优化处理机制 |
| :--- | :--- | :--- | :--- |
| **🟢 语义纯净度** | **85 / 100** | think、answer_body、summary中包含 `JSON`、`Schema`、`refs` 等工程元数据或检索抱怨；出现“切面/视角”元叙述。 | 命中高置信禁词一票否决，启动反馈控制环重写，若三次失败则根据 Transaction 机制予以隔离或丢弃。 |
| **🩺 医学严谨度** | **90 / 100** | 生成的 CoT 推导中包含无法被 FactPack 事实库支持的医学因果断言，或篡改批准文号与药理限制。 | 裁判引入脱敏后事实包，将新增医学事实与 facts 进行比对，若未通过 Fact Entailment Check 判定为幻觉。 |
| **🧠 逻辑深度** | **85 / 100** | 简单适应症/剂量等事实题进行复杂的微观通路脑补；或者机制题缺乏推理心流与自我质疑疑问锚点（“?”）。 | 低证据题强制 simplify 缩窄深度；高证据题依据 planner 动态控制推理硬门槛与字数限制。 |

---

## 🎯 4. 数据提纯的学术理念

项目的提纯净化拒绝简单的“文本剔除”，而是为了生成最符合人类顶尖医生大脑认知的思维链。
在最终的 `medical_qa_dataset.jsonl` 中，净化后的思维链展现了经典的**五阶段动态临床推理心流**：

$$\text{核心矛盾解构} \xrightarrow{\text{因果推演}} \text{微观病生理逻辑} \xrightarrow{\text{假说排查}} \text{分叉与特殊情况} \xrightarrow{\text{核准校对}} \text{生理/安全极限} \xrightarrow{\text{自然合拢}} \text{决策得出}$$

这套高度规范的流程图已与目前最新的代码库与重构蓝图保持 100% 同步，并作为指导本项目未来工程化、工业级数据微调准备的重要技术规范。
