# -*- coding: utf-8 -*-
"""
医疗问答思维链提纯净化自动化审计脚本。

Hard safety rules:
- The current dataset and raw backup are read-only inputs.
- All refs used for auditing must come from a verified current<->raw row mapping.
- If a current row cannot be mapped to raw safely, its refs are not used and the run aborts by default.
"""

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx


current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import config


try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


AUDIT_MODEL = config.AUDIT_MODEL
REPORT_MODEL = config.REPORT_MODEL

DEFAULT_DATASET_PATH = parent_dir / "medical_qa_dataset.jsonl"
DEFAULT_RAW_DATASET_PATH = parent_dir / "medical_qa_dataset_raw.jsonl"
DEFAULT_REPORT_DIR = parent_dir / "scratch"

SEMAPHORE = asyncio.Semaphore(int(os.getenv("AUDIT_CONCURRENCY", "5")))
MAX_REF_ITEMS = int(os.getenv("AUDIT_MAX_REF_ITEMS", "16"))
MAX_REF_CONTEXT_CHARS = int(os.getenv("AUDIT_MAX_REF_CONTEXT_CHARS", "800"))


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def short_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def assert_readable_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def assert_safe_output_path(output_path: Path, protected_paths: set[Path]) -> None:
    resolved = output_path.resolve()
    if resolved in protected_paths:
        raise RuntimeError(f"Refusing to write audit output to protected read-only input path: {resolved}")
    for protected in protected_paths:
        if resolved == protected:
            raise RuntimeError(f"Refusing to write audit output to protected input path: {resolved}")


def safe_write_json(path: Path, payload: Any, protected_paths: set[Path]) -> None:
    assert_safe_output_path(path, protected_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_write_text(path: Path, text: str, protected_paths: set[Path]) -> None:
    assert_safe_output_path(path, protected_paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_jsonl_readonly(path: Path, label: str) -> List[Dict[str, Any] | None]:
    """
    Read-only loader. This function must never open with a write/append/update mode.
    """
    assert_readable_file(path, label)
    rows: List[Dict[str, Any] | None] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                rows.append(None)
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"⚠️ {label} 第 {line_num} 行 JSON 解析失败: {e}")
                rows.append(None)
    return rows


def get_auth_headers() -> Dict[str, str]:
    api_key = config.LLM_API_KEY.strip() if config.LLM_API_KEY else ""
    if not api_key:
        raise RuntimeError("未在 .env 中检测到有效的 LLM_API_KEY")
    return {
        "Content-Type": "application/json",
        "Authorization": api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}",
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("parsed JSON is not an object")
        return data
    except Exception as e:
        return {
            "audit_status": "PARSE_ERROR",
            "error_message": f"大模型输出未通过 JSON 解析: {e}",
            "raw_model_output": text[:4000],
            "is_forced_facet": False,
            "forced_facet_reason": "",
            "cot_quality_score": 0,
            "cot_quality_issues": ["JSON解析失败"],
            "hallucinations_detected": False,
            "hallucinations_reason": "",
            "verdict": "REVIEW",
            "recommended_action": "MANUAL_REVIEW",
            "recommended_edit": "",
        }


def normalize_audit_result(result: Dict[str, Any]) -> Dict[str, Any]:
    result.setdefault("audit_status", "OK")
    result.setdefault("error_message", "")
    result.setdefault("is_forced_facet", False)
    result.setdefault("forced_facet_reason", "")
    result.setdefault("cot_quality_score", 0)
    result.setdefault("cot_quality_issues", [])
    result.setdefault("hallucinations_detected", False)
    result.setdefault("hallucinations_reason", "")
    result.setdefault("verdict", "REVIEW")
    result.setdefault("recommended_edit", "")

    verdict = str(result.get("verdict", "REVIEW")).upper()
    if verdict not in {"KEEP", "EDIT", "DISCARD", "REVIEW"}:
        verdict = "REVIEW"
    result["verdict"] = verdict

    if "recommended_action" not in result:
        if result["audit_status"] != "OK":
            action = "MANUAL_REVIEW"
        elif verdict == "KEEP":
            action = "KEEP"
        elif verdict == "EDIT":
            action = "REWRITE"
        elif verdict == "DISCARD":
            action = "PRUNE" if result.get("is_forced_facet") else "MANUAL_REVIEW"
        else:
            action = "MANUAL_REVIEW"
        result["recommended_action"] = action
    return result


async def call_llm(client: httpx.AsyncClient, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Tuple[str, int]:
    headers = get_auth_headers()
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    max_retries = int(os.getenv("AUDIT_MAX_RETRIES", "3"))
    for attempt in range(1, max_retries + 1):
        try:
            async with SEMAPHORE:
                response = await client.post(config.LLM_API_URL, headers=headers, json=data, timeout=90.0)
            if response.status_code == 200:
                res_json = response.json()
                if "choices" in res_json and res_json["choices"]:
                    return res_json["choices"][0]["message"]["content"], attempt
                if "error" in res_json:
                    print(f"⚠️ API 业务报错: {res_json['error'].get('message')}")
                else:
                    print(f"⚠️ 未知 API 响应格式: {res_json}")
            else:
                print(f"⚠️ HTTP 报错 {response.status_code} on attempt {attempt}: {response.text[:500]}")
        except Exception as e:
            print(f"⚠️ 请求异常 on attempt {attempt}: {e}")

        if attempt < max_retries:
            await asyncio.sleep(2.0 * attempt)

    raise RuntimeError(f"大模型请求失败，重试 {max_retries} 次均无法恢复。")


def format_refs_for_prompt(refs: List[Dict[str, Any]]) -> str:
    if not refs:
        return "未提供 refs。若无法基于事实锚点判断，请明确标记为 MANUAL_REVIEW，不要臆断。"

    chunks = []
    for idx, ref in enumerate(refs[:MAX_REF_ITEMS], start=1):
        source = str(ref.get("source", "")).strip()
        context = str(ref.get("context", "")).strip()
        if len(context) > MAX_REF_CONTEXT_CHARS:
            context = context[:MAX_REF_CONTEXT_CHARS] + "..."
        chunks.append(f"[{idx}] source: {source}\ncontext: {context}")
    if len(refs) > MAX_REF_ITEMS:
        chunks.append(f"... 其余 {len(refs) - MAX_REF_ITEMS} 条 refs 已截断。")
    return "\n\n".join(chunks)


def build_raw_mapping(records: List[Dict[str, Any]], raw_rows: List[Dict[str, Any] | None]) -> Tuple[Dict[int, Dict[str, Any]], List[str], List[str]]:
    raw_by_q: Dict[str, List[int]] = {}
    for idx, raw in enumerate(raw_rows, start=1):
        if not raw:
            continue
        key = normalize_key(raw.get("Q", ""))
        if key:
            raw_by_q.setdefault(key, []).append(idx)

    mapping: Dict[int, Dict[str, Any]] = {}
    warnings: List[str] = []
    errors: List[str] = []

    for current in records:
        line_num = current["line_num"]
        q = current.get("Q", "")
        key = normalize_key(q)
        same_line_raw = raw_rows[line_num - 1] if 0 <= line_num - 1 < len(raw_rows) else None

        if same_line_raw and normalize_key(same_line_raw.get("Q", "")) == key:
            mapping[line_num] = {
                "raw_line": line_num,
                "raw_record": same_line_raw,
                "mapping_status": "SAME_LINE",
                "mapping_warning": "",
            }
            continue

        candidates = raw_by_q.get(key, [])
        if len(candidates) == 1:
            raw_line = candidates[0]
            warning = (
                f"🚨 RAW/CURRENT 行号不一致: current line {line_num} maps to raw line {raw_line} by exact Q. "
                "将使用映射后的 raw refs，禁止按同号行取 refs。"
            )
            warnings.append(warning)
            mapping[line_num] = {
                "raw_line": raw_line,
                "raw_record": raw_rows[raw_line - 1],
                "mapping_status": "MAPPED_BY_Q",
                "mapping_warning": warning,
            }
        elif len(candidates) > 1:
            error = (
                f"🚨 RAW/CURRENT 映射歧义: current line {line_num} 的 Q 在 raw 中匹配多行 {candidates}. "
                "无法安全引用 refs。"
            )
            errors.append(error)
            mapping[line_num] = {
                "raw_line": None,
                "raw_record": None,
                "mapping_status": "AMBIGUOUS_Q",
                "mapping_warning": error,
            }
        else:
            error = (
                f"🚨 RAW/CURRENT 映射失败: current line {line_num} 的 Q 在 raw 中找不到。"
                "无法安全引用 refs。"
            )
            errors.append(error)
            mapping[line_num] = {
                "raw_line": None,
                "raw_record": None,
                "mapping_status": "NO_RAW_MATCH",
                "mapping_warning": error,
            }

    return mapping, warnings, errors


async def audit_single_facet(
    client: httpx.AsyncClient,
    run_id: str,
    line_num: int,
    question: str,
    planner_index: int,
    planner: Dict[str, Any],
    raw_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    started = time.time()
    facet = planner.get("planner", "")
    ans = planner.get("answer", "")
    think_match = re.search(r"<think>([\s\S]*?)</think>", ans)
    think_content = think_match.group(1).strip() if think_match else ""
    response_content = re.sub(r"<think>[\s\S]*?</think>", "", ans).strip()

    raw_record = raw_mapping.get("raw_record")
    refs = raw_record.get("refs", []) if raw_record else []
    raw_summary = raw_record.get("summary", "") if raw_record else ""
    raw_line = raw_mapping.get("raw_line")
    mapping_status = raw_mapping.get("mapping_status", "UNKNOWN")
    mapping_warning = raw_mapping.get("mapping_warning", "")

    base_meta = {
        "run_id": run_id,
        "audit_time": dt.datetime.now().isoformat(timespec="seconds"),
        "model_profile": config.MODEL_PROFILE,
        "audit_model": AUDIT_MODEL,
        "line": line_num,
        "raw_line": raw_line,
        "raw_mapping_status": mapping_status,
        "raw_mapping_warning": mapping_warning,
        "Q": question,
        "planner_index": planner_index,
        "facet": facet,
        "input_hash": short_sha256(json.dumps(planner, ensure_ascii=False, sort_keys=True)),
        "raw_refs_count": len(refs),
        "raw_has_history": bool(raw_record and "history" in raw_record),
        "original_think": think_content,
        "answer_body": response_content,
        "raw_summary": raw_summary,
    }

    if mapping_status in {"AMBIGUOUS_Q", "NO_RAW_MATCH", "UNKNOWN"}:
        return {
            **base_meta,
            "audit_status": "RAW_MAPPING_ERROR",
            "error_message": mapping_warning,
            "is_forced_facet": False,
            "forced_facet_reason": "",
            "cot_quality_score": 0,
            "cot_quality_issues": ["raw映射失败，禁止引用refs"],
            "hallucinations_detected": False,
            "hallucinations_reason": "",
            "verdict": "REVIEW",
            "recommended_action": "RESTORE_OR_MAP_RAW",
            "recommended_edit": "",
            "latency_seconds": round(time.time() - started, 2),
            "attempts": 0,
        }

    if not think_content:
        return {
            **base_meta,
            "audit_status": "MISSING_THINK",
            "error_message": "原始数据缺少 <think> 思维链模块",
            "is_forced_facet": False,
            "forced_facet_reason": "",
            "cot_quality_score": 0,
            "cot_quality_issues": ["缺失思维链"],
            "hallucinations_detected": False,
            "hallucinations_reason": "",
            "verdict": "REVIEW",
            "recommended_action": "RESTORE_RAW_OR_REGENERATE",
            "recommended_edit": "",
            "latency_seconds": round(time.time() - started, 2),
            "attempts": 0,
        }

    system_prompt = """你是一位极其严谨的医疗问答数据集质检专家和审计法官。你的任务是评估大模型生成的医疗思维链（CoT）和最终回答的质量。
请根据提供的主问题 Q、当前分析视角（Facet）、原始 raw refs 事实锚点、提取的思维链和回答内容，进行客观、深度的学术和逻辑判定。

你需要特别关注以下维度：
1. 硬套视角（Forced Facet）：主问题与分析视角之间是否存在强行联系、牵强附会的逻辑漂移。
2. 思维链质量（CoT Quality）：是否包含 JSON、Schema、refs、实体库、知识图谱、参考依据项、步骤P1/P2/P3 等工程痕迹；是否单薄；是否废话堆砌。
3. 幻觉与假信息（Hallucinations & False Info）：必须优先对照 raw refs 事实锚点。若无法从 raw refs 或医学公理支持，请标记风险，禁止臆断。
4. 处理建议（recommended_action）：KEEP、REWRITE、PRUNE、RESTORE_RAW_OR_REGENERATE、MANUAL_REVIEW 中选择一个。

你必须且只能输出一个合法 JSON 对象，不要包含 markdown 围栏或任何多余文字，结构如下：
{
  "is_forced_facet": true/false,
  "forced_facet_reason": "如果是硬套视角，请说明违和点；如果不是，填空字符串",
  "cot_quality_score": 0到100之间的整数评分,
  "cot_quality_issues": ["工程词汇泄露（如refs）", "逻辑单薄", "废话堆砌", "无明显问题"],
  "hallucinations_detected": true/false,
  "hallucinations_reason": "如果发现幻觉或假信息，请说明；如果不是，填空字符串",
  "verdict": "KEEP" | "EDIT" | "DISCARD" | "REVIEW",
  "recommended_action": "KEEP" | "REWRITE" | "PRUNE" | "RESTORE_RAW_OR_REGENERATE" | "MANUAL_REVIEW",
  "recommended_edit": "如果 recommended_action 是 REWRITE，请提供纯净思维链；否则填空字符串"
}
"""

    user_prompt = f"""主问题 Q: "{question}"
当前视角 Facet: "{facet}"
当前数据行: {line_num}
映射 raw 行: {raw_line}
raw 映射状态: {mapping_status}

raw refs 事实锚点:
{format_refs_for_prompt(refs)}

提取的思维链 (CoT):
{think_content}

提取的回答内容 (Response):
{response_content}

请开始质检与审计，并输出 JSON 格式的结果。"""

    try:
        raw_output, attempts = await call_llm(client, AUDIT_MODEL, system_prompt, user_prompt, temperature=0.1)
        audit_res = normalize_audit_result(extract_json_object(raw_output))
        audit_res.setdefault("raw_model_output", raw_output[:4000])
        audit_res["attempts"] = attempts
    except Exception as e:
        audit_res = normalize_audit_result({
            "audit_status": "MODEL_ERROR",
            "error_message": str(e),
            "is_forced_facet": False,
            "forced_facet_reason": "",
            "cot_quality_score": 0,
            "cot_quality_issues": ["接口报错"],
            "hallucinations_detected": False,
            "hallucinations_reason": "",
            "verdict": "REVIEW",
            "recommended_action": "MANUAL_REVIEW",
            "recommended_edit": "",
            "attempts": int(os.getenv("AUDIT_MAX_RETRIES", "3")),
        })

    audit_res.update(base_meta)
    audit_res["latency_seconds"] = round(time.time() - started, 2)
    print(
        f"✅ 行号 {line_num} | raw {raw_line} | 视角[{planner_index}:{facet}] "
        f"- 状态: {audit_res['audit_status']} 裁决: {audit_res['verdict']} "
        f"动作: {audit_res['recommended_action']} 分数: {audit_res['cot_quality_score']}"
    )
    return audit_res


async def generate_markdown_report(client: httpx.AsyncClient, start_line: int, end_line: int, audit_results: List[Dict[str, Any]]) -> str:
    system_prompt = """你是一个高级医疗数据分析师和质量控制专家。
你需要根据输入的一批医疗问答数据审计结果（JSON），撰写一份专业、可追踪的 Markdown 审计报告。

报告必须包含：
1. 审计概览：总视角数、audit_status 分布、verdict 分布、recommended_action 分布。
2. raw/current 映射安全性：列出所有 MAPPED_BY_Q、AMBIGUOUS_Q、NO_RAW_MATCH。
3. 典型问题汇总：强套视角、工程词泄露、事实幻觉、接口/解析错误。
4. 逐条明细：行号、raw行号、planner_index、facet、状态、裁决、建议动作、关键原因。

直接输出 Markdown，不要输出额外解释。"""
    user_prompt = f"""审计数据集行数区间: 第 {start_line} 行 至 第 {end_line} 行。
所有审计详细结果:
{json.dumps(audit_results, ensure_ascii=False, indent=2)}

请生成最终 Markdown 审计报告。"""

    print(f"\n✍️ 正在使用报告模型 ({REPORT_MODEL}) 整合 Markdown 审计报告...")
    try:
        md_report, _ = await call_llm(client, REPORT_MODEL, system_prompt, user_prompt, temperature=0.3)
        return md_report
    except Exception as e:
        print(f"❌ 撰写 Markdown 报告失败: {e}. 将生成基础文本报告。")
        lines = [f"# 医疗问答思维链净化审计报告 (第 {start_line} - {end_line} 行 - 简易版)\n"]
        lines.append(f"- 审计时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- 总视角数: {len(audit_results)}\n")
        for item in audit_results:
            lines.append(f"### 行 {item['line']} | raw {item.get('raw_line')} | 视角 {item.get('planner_index')}: {item['facet']}")
            lines.append(f"- 状态: {item['audit_status']}")
            lines.append(f"- 裁决: {item['verdict']}")
            lines.append(f"- 建议动作: {item['recommended_action']}")
            if item.get("raw_mapping_warning"):
                lines.append(f"- raw映射警告: {item['raw_mapping_warning']}")
            if item.get("error_message"):
                lines.append(f"- 错误: {item['error_message']}")
            lines.append("")
        return "\n".join(lines)


def summarize_counts(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        val = str(item.get(key, "UNKNOWN"))
        counts[val] = counts.get(val, 0) + 1
    return counts


async def main() -> None:
    parser = argparse.ArgumentParser(description="分批医疗QA数据集大模型质量审计工具")
    parser.add_argument("--start", type=int, default=101, help="当前数据集起始行号 (1-based)")
    parser.add_argument("--limit", type=int, default=10, help="读取当前数据集记录条数，不是 facet 数")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="当前待审计数据集路径，只读")
    parser.add_argument("--raw_dataset", type=str, default=str(DEFAULT_RAW_DATASET_PATH), help="raw备份数据集路径，只读")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_REPORT_DIR), help="审计报告输出目录")
    parser.add_argument("--allow_unmapped_refs", action="store_true", help="允许 raw 映射失败的条目继续审计，但不会注入 refs")
    args = parser.parse_args()

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = resolve_path(args.dataset)
    raw_dataset_path = resolve_path(args.raw_dataset)
    output_dir = resolve_path(args.output_dir)
    protected_paths = {dataset_path, raw_dataset_path}

    print(f"🚀 开始审计数据集 (自第 {args.start} 行起, 限制读取 {args.limit} 条)")
    print(f"🤖 核心评估模型: {AUDIT_MODEL} | 报告整合模型: {REPORT_MODEL}")
    print(f"🔒 当前数据集只读输入: {dataset_path}")
    print(f"🔒 raw 备份只读输入: {raw_dataset_path}")

    current_rows = load_jsonl_readonly(dataset_path, "current dataset")
    raw_rows = load_jsonl_readonly(raw_dataset_path, "raw backup")
    print("🔒 已完成只读加载：脚本不会写入 current dataset 或 raw backup。")

    records = []
    for line_num in range(args.start, min(len(current_rows), args.start + args.limit - 1) + 1):
        row = current_rows[line_num - 1]
        if row is None:
            continue
        row = dict(row)
        row["line_num"] = line_num
        records.append(row)

    if not records:
        print("⚠️ 未读取到符合范围的数据记录。")
        return

    actual_end_line = records[-1]["line_num"]
    raw_mapping, mapping_warnings, mapping_errors = build_raw_mapping(records, raw_rows)
    if mapping_warnings:
        print("\n🚨 RAW/CURRENT 行映射警示:")
        for warning in mapping_warnings:
            print(f"  - {warning}")
    if mapping_errors:
        print("\n🚨 RAW/CURRENT 行映射错误，禁止错误引用 refs:")
        for error in mapping_errors:
            print(f"  - {error}")
        if not args.allow_unmapped_refs:
            print("❌ 已中止审计。若要无 refs 继续审计，请显式加 --allow_unmapped_refs。")
            sys.exit(2)

    print(f"📖 成功读取 {len(records)} 条记录 (覆盖当前数据集第 {args.start} 至 {actual_end_line} 行)\n")

    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = []
        for record in records:
            line_num = record["line_num"]
            question = record.get("Q", "")
            for planner_index, planner in enumerate(record.get("planners", [])):
                tasks.append(audit_single_facet(client, run_id, line_num, question, planner_index, planner, raw_mapping[line_num]))

        if not tasks:
            print("⚠️ 未在该范围内找到需要质检的视角(planners)。")
            return

        print(f"⏳ 正在并发审计 {len(tasks)} 个视角，请稍候...")
        start_time = time.time()
        audit_results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        print(f"\n📊 审计执行完毕! 耗时: {elapsed:.2f} 秒，平均每个视角耗时 {elapsed/len(tasks):.2f} 秒。")

        json_report_path = output_dir / f"audit_report_lines_{args.start}_{actual_end_line}_{run_id}.json"
        latest_json_report_path = output_dir / f"audit_report_lines_{args.start}_{actual_end_line}.json"
        safe_write_json(json_report_path, audit_results, protected_paths)
        safe_write_json(latest_json_report_path, audit_results, protected_paths)
        print(f"💾 结构化 JSON 详细记录已保存至: {json_report_path}")

        md_report_content = await generate_markdown_report(client, args.start, actual_end_line, audit_results)
        md_report_path = output_dir / f"audit_report_lines_{args.start}_{actual_end_line}_{run_id}.md"
        latest_md_report_path = output_dir / f"audit_report_lines_{args.start}_{actual_end_line}.md"
        safe_write_text(md_report_path, md_report_content, protected_paths)
        safe_write_text(latest_md_report_path, md_report_content, protected_paths)
        print(f"🎉 Markdown 报告已写入: {md_report_path}")

        print("=============================================================")
        print(f"📊 【本次审计汇总 (当前第 {args.start} - {actual_end_line} 行)】")
        print(f"  - 视角总数: {len(audit_results)}")
        print(f"  - audit_status: {summarize_counts(audit_results, 'audit_status')}")
        print(f"  - verdict: {summarize_counts(audit_results, 'verdict')}")
        print(f"  - recommended_action: {summarize_counts(audit_results, 'recommended_action')}")
        print(f"  - raw_mapping_status: {summarize_counts(audit_results, 'raw_mapping_status')}")


if __name__ == "__main__":
    asyncio.run(main())
