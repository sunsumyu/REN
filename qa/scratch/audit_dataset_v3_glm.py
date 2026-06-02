# -*- coding: utf-8 -*-
"""
医疗问答思维链提纯净化自动化审计脚本 (基于 GLM-5.1 与 便宜模型 混合架构)
"""

import os
import sys
import json
import re
import argparse
import asyncio
import httpx
import time
from pathlib import Path

# 动态载入项目根目录以读取配置文件
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import config

# 配置模型路由
AUDIT_MODEL = "Pro/zai-org/GLM-5.1"
REPORT_MODEL = "Pro/zai-org/GLM-4.7"

# 审计结果输出路径
DEFAULT_DATASET_PATH = r"d:\REN\qa\medical_qa_dataset.jsonl"
DEFAULT_MD_REPORT_DIR = r"d:\REN\qa\scratch"

# 并发限制信号量
SEMAPHORE = asyncio.Semaphore(5)

def get_auth_headers():
    headers = {
        "Content-Type": "application/json"
    }
    api_key = config.LLM_API_KEY.strip() if config.LLM_API_KEY else ""
    if not api_key:
        print("❌ 错误: 未在 .env 中检测到有效的 LLM_API_KEY！")
        sys.exit(1)
        
    if api_key.startswith("Bearer "):
        headers["Authorization"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def extract_json_object(text: str) -> dict:
    """从大模型输出的文本中稳健提取并解析 JSON 对象"""
    text = text.strip()
    # 移除 markdown 代码块包裹
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    
    # 截取首个大括号和最后一个大括号之间的内容
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace+1]
        
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失败: {e}. 原始文本: {text[:200]}...")
        return {
            "is_forced_facet": False,
            "forced_facet_reason": f"大模型输出未通过JSON解析: {str(e)}",
            "cot_quality_score": 0,
            "cot_quality_issues": ["JSON解析失败"],
            "hallucinations_detected": False,
            "hallucinations_reason": "",
            "verdict": "DISCARD",
            "recommended_edit": ""
        }

async def call_llm(client: httpx.AsyncClient, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """向大模型发送对话请求并带重试逻辑"""
    headers = get_auth_headers()
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with SEMAPHORE:
                response = await client.post(config.LLM_API_URL, headers=headers, json=data, timeout=90.0)
                if response.status_code == 200:
                    res_json = response.json()
                    if "choices" in res_json and len(res_json["choices"]) > 0:
                        return res_json["choices"][0]["message"]["content"]
                    elif "error" in res_json:
                        print(f"⚠️ API 业务报错: {res_json['error'].get('message')}")
                    else:
                        print(f"⚠️ 未知 API 响应格式: {res_json}")
                else:
                    print(f"⚠️ HTTP 报错 {response.status_code} on attempt {attempt+1}: {response.text}")
        except Exception as e:
            print(f"⚠️ 请求异常 on attempt {attempt+1}: {e}")
            
        if attempt < max_retries - 1:
            await asyncio.sleep(2.0 * (attempt + 1))
            
    raise Exception(f"大模型请求失败，重试 {max_retries} 次均无法恢复。")

async def audit_single_facet(client: httpx.AsyncClient, line_num: int, question: str, planner: dict) -> dict:
    """审计单条问答对下的单个视角"""
    facet = planner.get("planner", "")
    ans = planner.get("answer", "")
    
    # 提取 <think> 模块和回答正文
    think_match = re.search(r"<think>([\s\S]*?)</think>", ans)
    think_content = think_match.group(1).strip() if think_match else ""
    response_content = re.sub(r"<think>[\s\S]*?</think>", "", ans).strip()
    
    # 如果完全没有思维链，直接视为缺陷
    if not think_content:
        return {
            "line": line_num,
            "Q": question,
            "facet": facet,
            "is_forced_facet": False,
            "forced_facet_reason": "原始数据缺少 <think> 思维链模块",
            "cot_quality_score": 0,
            "cot_quality_issues": ["缺失思维链"],
            "hallucinations_detected": False,
            "hallucinations_reason": "",
            "verdict": "DISCARD",
            "recommended_edit": ""
        }
        
    system_prompt = """你是一位极其严谨的医疗问答数据集质检专家和审计法官。你的任务是评估大模型生成的医疗思维链（CoT）和最终回答的质量。
请根据提供的主问题 Q、当前分析视角（Facet）以及提取的思维链和回答内容，进行客观、深度的学术和逻辑判定。

你需要特别关注以下三大维度：
1. **硬套视角（Forced Facet）**：
   - 判定主问题与分析视角之间是否存在强行联系、牵强附会的“逻辑漂移”。
   - 例如：主问题是简单的“某种药物的成分/用法是什么”，却被强行套入“患者数据隐私”、“药物动力学”、“分子机制”等高阶学术视角，导致模型在思维链中花费大量篇幅自我解释“因为视角是数据隐私，我不得不将该问题与隐私联系起来...”。这种属于严重的“硬套视角”。
2. **思维链质量（CoT Quality）**：
   - 是否包含大模型开发阶段的工程提示词或指令痕迹泄露（如提及“JSON”、“Schema”、“refs”、“参考依据项”、“步骤P1/P2/P3”等）。
   - 思维链是否过于单薄简短（如少于150字，缺少推导过程）。
   - 是否存在大量低质量的自我纠偏、废话或者单纯的名词堆砌。
3. **幻觉与假信息（Hallucinations & False Info）**：
   - 是否存在由于RAG检索异常退避而产生的模板式幻觉（如将非药物实体如“密封”、“Rx”套入“药物【XX】目前在临床上用于辅助治疗...”的描述中）。
   - 是否存在编造统计数值、参考文献或临床病理机制等恶性事实捏造。

你必须且只能输出一个合法的 JSON 对象，不要包含 markdown 围栏（如 ```json）或任何多余文字，结构如下：
{
  "is_forced_facet": true/false,
  "forced_facet_reason": "如果是硬套视角，请详细阐述其逻辑漂移的违和点；如果不是，请填空字符串",
  "cot_quality_score": 0到100之间的整数评分,
  "cot_quality_issues": ["工程词汇泄露（如refs）", "逻辑单薄", "废话堆砌", "无明显问题"],
  "hallucinations_detected": true/false,
  "hallucinations_reason": "如果发现幻觉或假信息，请详细阐述捏造或不合常理的事实细节；如果不是，请填空字符串",
  "verdict": "KEEP" | "EDIT" | "DISCARD",
  "recommended_edit": "如果 verdict 是 EDIT，请提供一个去除了工程提示词干扰、修正了轻微事实错误、且逻辑自然流畅的纯净思维链；否则请填空字符串"
}
"""
    user_prompt = f"""主问题 Q: "{question}"
当前视角 Facet: "{facet}"
提取的思维链 (CoT):
{think_content}

提取的回答内容 (Response):
{response_content}

请开始质检与审计，并输出 JSON 格式的结果。"""

    # 移除 try-except 以便异常能直接冒泡传递，触发整批任务熔断，防止记录不完整/损坏的脏数据
    raw_output = await call_llm(client, AUDIT_MODEL, system_prompt, user_prompt, temperature=0.1)
    audit_res = extract_json_object(raw_output)
    
    # 补全基础元数据
    audit_res["line"] = line_num
    audit_res["Q"] = question
    audit_res["facet"] = facet
    audit_res["original_think"] = think_content
    
    # 控制台打印即时质检状态
    print(f"✅ 行号 {line_num} | 视角 [{facet}] - 质检完成. 裁决: {audit_res['verdict']} (评分: {audit_res['cot_quality_score']})")
    return audit_res

async def generate_markdown_report(client: httpx.AsyncClient, start_line: int, end_line: int, audit_results: list) -> str:
    """使用便宜模型 (GLM-4.7) 撰写精美详尽的 Markdown 审计报告"""
    system_prompt = """你是一个高级医疗数据分析师和质量控制专家。
你需要根据输入的一批医疗问答数据审计结果（JSON 格式），撰写一份美观、专业且易于阅读的 Markdown 格式的**医疗问答思维链提纯与清洗审计报告**。

报告结构必须包括：
1. **📊 审计概览**：总计清洗条数、通过率（KEEP）、需修改比例（EDIT）、强行硬套/严重幻觉需废弃比例（DISCARD）的统计分析。
2. **⚠️ 典型问题汇总**：分门别类总结这一批数据中发现的主要问题（如：哪些视角被强套、哪些地方存在模板幻觉、哪些工程词汇泄露最频繁）。
3. **🔍 逐条审计诊断明细**：
   - 给出每一条 QA 数据的具体行号、主问题、视角。
   - 详述判定结果：是否硬套（及原因）、思维链质量（及评分与问题列表）、是否检测到幻觉（及细节）、最终裁决（KEEP/EDIT/DISCARD）。
   - 如果裁决是 EDIT，列出推荐的修改方案。

请以极具医学专业度和美感的形式呈现，使用丰富的 GitHub Alert 块（如 > [!NOTE], > [!WARNING], > [!IMPORTANT]）来高亮关键诊断。不要输出任何多余的解释文字，直接输出 Markdown 文本即可。"""

    user_prompt = f"""审计数据集行数区间: 第 {start_line} 行 至 第 {end_line} 行。
所有审计详细结果 (JSON):
{json.dumps(audit_results, ensure_ascii=False, indent=2)}

请生成最终的 Markdown 审计报告。"""

    print(f"\n✍️ 正在使用便宜模型 ({REPORT_MODEL}) 整合并撰写 Markdown 审计报告...")
    try:
        md_report = await call_llm(client, REPORT_MODEL, system_prompt, user_prompt, temperature=0.3)
        return md_report
    except Exception as e:
        print(f"❌ 撰写 Markdown 报告失败: {e}. 将生成基础文本报告。")
        # 兜底生成一个极简报告
        lines = []
        lines.append(f"# 医疗问答思维链净化审计报告 (第 {start_line} - {end_line} 行 - 简易版)\n")
        lines.append(f"- 审计时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- 总条数: {len(audit_results)}\n")
        for item in audit_results:
            lines.append(f"### 行 {item['line']} | 问题: {item['Q']} | 视角: {item['facet']}")
            lines.append(f"- **裁决**: {item['verdict']} (分数: {item['cot_quality_score']})")
            if item['is_forced_facet']:
                lines.append(f"- **硬套原因**: {item['forced_facet_reason']}")
            if item['hallucinations_detected']:
                lines.append(f"- **幻觉问题**: {item['hallucinations_reason']}")
            lines.append("")
        return "\n".join(lines)

async def main():
    parser = argparse.ArgumentParser(description="分批医疗QA数据集大模型质量审计工具")
    parser.add_argument("--start", type=int, default=101, help="数据集的起始读取行号 (1-based index)")
    parser.add_argument("--limit", type=int, default=10, help="单次执行审计的总数据条数")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="QA数据集文件路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_MD_REPORT_DIR, help="审计报告输出目录")
    
    args = parser.parse_args()
    
    start_line = args.start
    limit = args.limit
    dataset_path = args.dataset
    output_dir = args.output_dir
    
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 未能找到数据集文件 {dataset_path}")
        return
        
    print(f"🚀 开始审计数据集 (自第 {start_line} 行起, 限制读取 {limit} 条)")
    print(f"🤖 核心评估模型: {AUDIT_MODEL} | 报告整合模型: {REPORT_MODEL}")
    
    records = []
    
    # 流式读取特定范围的行数
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_num = idx + 1
            if line_num < start_line:
                continue
            if len(records) >= limit:
                break
            if line.strip():
                try:
                    data = json.loads(line)
                    data["line_num"] = line_num
                    records.append(data)
                except Exception as e:
                    print(f"⚠️ 解析数据集第 {line_num} 行 JSON 失败: {e}")
                    
    if not records:
        print("⚠️ 未读取到符合范围的数据记录。请检查起始行号是否正确。")
        return
        
    actual_end_line = records[-1]["line_num"]
    print(f"📖 成功读取了 {len(records)} 条记录 (覆盖数据集第 {start_line} 至 {actual_end_line} 行)\n")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # 并发执行审计任务
        tasks = []
        for r in records:
            line_num = r["line_num"]
            question = r.get("Q", "")
            planners = r.get("planners", [])
            for p in planners:
                tasks.append(audit_single_facet(client, line_num, question, p))
                
        if not tasks:
            print("⚠️ 未在该范围内找到需要质检的视角(planners)。")
            return
            
        print(f"⏳ 正在并发审计 {len(tasks)} 个视角，请稍候...")
        start_time = time.time()
        try:
            audit_results = await asyncio.gather(*tasks)
        except Exception as e:
            print(f"\n❌ 严重错误: 核心审计任务因大模型接口或网络异常中断（例如504网关超时/网络重试耗尽）！")
            print(f"  - 异常详情: {e}")
            print(f"  - 提示: 已安全丢弃并放弃本次清洗任务，未写入任何不完整的 JSON 或 Markdown 报告，以防脏数据污染。")
            sys.exit(1)
            
        elapsed = time.time() - start_time
        print(f"\n📊 审计执行完毕! 耗时: {elapsed:.2f} 秒，平均每个视角耗时 {elapsed/len(tasks):.2f} 秒。")
        
        # 保存结构化 JSON
        json_report_path = os.path.join(output_dir, f"audit_report_lines_{start_line}_{actual_end_line}.json")
        with open(json_report_path, "w", encoding="utf-8") as j_f:
            json.dump(audit_results, j_f, ensure_ascii=False, indent=2)
        print(f"💾 结构化 JSON 详细记录已保存至: {json_report_path}")
        
        # 便宜模型生成 Markdown 审计报告
        md_report_content = await generate_markdown_report(client, start_line, actual_end_line, audit_results)
        
        md_report_path = os.path.join(output_dir, f"audit_report_lines_{start_line}_{actual_end_line}.md")
        with open(md_report_path, "w", encoding="utf-8") as m_f:
            m_f.write(md_report_content)
            
        print(f"🎉 审计完成！Markdown 格式报告已成功写入: {md_report_path}")
        print("=============================================================")
        
        # 控制台简易输出结果总结
        total = len(audit_results)
        keeps = sum(1 for x in audit_results if x["verdict"] == "KEEP")
        edits = sum(1 for x in audit_results if x["verdict"] == "EDIT")
        discards = sum(1 for x in audit_results if x["verdict"] == "DISCARD")
        print(f"📊 【本次执行数据清洗汇总 (第 {start_line} - {actual_end_line} 行)】")
        print(f"  - 视角总数 (Total Facets): {total}")
        print(f"  - 直接保留 (KEEP): {keeps} ({keeps/total*100:.1f}%) 🟢")
        print(f"  - 建议修改 (EDIT): {edits} ({edits/total*100:.1f}%) 🟡")
        print(f"  - 强烈废弃 (DISCARD): {discards} ({discards/total*100:.1f}%) 🔴")

if __name__ == "__main__":
    asyncio.run(main())
