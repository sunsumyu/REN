import json
import os
import re
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "medical_qa_dataset.jsonl"
RAW_DATASET_PATH = PROJECT_DIR / "medical_qa_dataset_raw.jsonl"
MAX_LINES = 165

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_jsonl(path):
    if not path.exists():
        print(f"Dataset not found at {path}")
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if line_num > MAX_LINES:
                break
            if not line.strip():
                rows.append(None)
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"Error parsing {path.name} line {line_num}: {e}")
                rows.append(None)
    return rows


def count_think(planners):
    return sum(
        1
        for p in planners or []
        if re.match(r"^\s*<think>([\s\S]*?)</think>", p.get("answer", ""))
    )


def detect_think_noise(planners):
    noisy_planners = []
    for p_idx, p in enumerate(planners or []):
        ans = p.get("answer", "")
        think_match = re.match(r"^\s*<think>([\s\S]*?)</think>", ans)
        if not think_match:
            continue

        think_content = think_match.group(1)
        traces = []
        if "被要求" in think_content:
            traces.append("被要求")
        if "JSON" in think_content or "Schema" in think_content or "schema" in think_content:
            traces.append("JSON/Schema")
        if "refs" in think_content or "参考依据" in think_content:
            traces.append("refs")
        if re.search(r"\b[Pp]\d+\b", think_content):
            traces.append("P1/P2/P3步骤")
        if "由于" in think_content and "要求输出" in think_content:
            traces.append("格式自我讨论")
        if traces:
            noisy_planners.append((p_idx, p.get("planner", "")[:30], traces))
    return noisy_planners


def detect_mock_refs(refs):
    mock_hallucinations = []
    for ref_idx, ref in enumerate(refs or []):
        source = ref.get("source", "")
        context = ref.get("context", "")

        if "用于辅助治疗" in context:
            drug_match = re.search(r"药物【(.*?)】目前主要在临床上用于辅助治疗", context)
            if drug_match:
                ent_name = drug_match.group(1)
                if ent_name in ["密封", "Rx", "UGT2B7", "G6PD缺乏患者", "右旋糖酐 40", "尿酸转化为尿囊素", "高尿酸血症", "药物过量"]:
                    mock_hallucinations.append((ref_idx, ent_name))

        if "在线公开检索系统抓取异常" in source or "未收录或网络异常" in context:
            mock_hallucinations.append((ref_idx, "网络异常兜底Mock"))
    return mock_hallucinations


def audit_dataset():
    current_rows = load_jsonl(DATASET_PATH)
    raw_rows = load_jsonl(RAW_DATASET_PATH)
    if not current_rows:
        return

    print(f"Auditing first {MAX_LINES} lines of {DATASET_PATH}")
    if raw_rows:
        print(f"Comparing against raw backup: {RAW_DATASET_PATH}\n")
    else:
        print("Raw backup unavailable; running current-file checks only.\n")

    structural_issues = []
    planner_name_issues = []
    mock_ref_issues = []
    think_noise_issues = []

    for idx, current in enumerate(current_rows):
        line_num = idx + 1
        if current is None:
            continue

        raw = raw_rows[idx] if idx < len(raw_rows) else None
        q = current.get("Q", raw.get("Q", "") if raw else "")
        planners = current.get("planners", [])
        refs = current.get("refs", [])

        raw_planners = raw.get("planners", []) if raw else []
        raw_refs = raw.get("refs", []) if raw else []
        raw_has_history = raw is not None and "history" in raw
        current_has_refs = "refs" in current
        current_has_history = "history" in current
        current_think = count_think(planners)
        raw_think = count_think(raw_planners)

        row_structural = []
        if raw and raw_refs and not current_has_refs and current_think == 0 and raw_think > 0:
            row_structural.append("raw had refs/think but current has no refs and no think")
        if raw and raw_refs and not current_has_refs and raw_think > current_think:
            row_structural.append(f"think count decreased {raw_think}->{current_think} while refs missing")
        if raw_has_history and not current_has_history and raw_think > current_think:
            row_structural.append("history missing on a row whose think/planner count decreased")
        if len(raw_planners) > 0 and len(planners) == 0:
            row_structural.append(f"planners physically deleted {len(raw_planners)}->0")
        elif len(raw_planners) > len(planners):
            row_structural.append(f"planner count decreased {len(raw_planners)}->{len(planners)}")
        if current_has_refs and current_think == 0:
            row_structural.append("refs/history present but no think remains to purify")

        if row_structural:
            structural_issues.append((line_num, q[:60], row_structural))

        corrupted_planners = []
        for p_idx, p in enumerate(planners):
            planner_name = p.get("planner", "")
            if "rigorous data" in planner_name or "Schema" in planner_name:
                corrupted_planners.append((p_idx, planner_name[:80]))
        if corrupted_planners:
            planner_name_issues.append((line_num, q[:60], corrupted_planners))

        mock_refs = detect_mock_refs(refs)
        if mock_refs:
            mock_ref_issues.append((line_num, q[:60], mock_refs))

        noisy = detect_think_noise(planners)
        if noisy:
            think_noise_issues.append((line_num, q[:60], noisy))

    print("=== Category 1: Structural Damage / Non-idempotent Purify Damage ===")
    for line_num, q, issues in structural_issues:
        print(f"Line {line_num}: Q='{q}' - {issues}")
    print(f"Subtotal: {len(structural_issues)} lines\n")

    print("=== Category 2: Planner Name Corruption ===")
    for line_num, q, issues in planner_name_issues:
        print(f"Line {line_num}: Q='{q}' - {issues}")
    print(f"Subtotal: {len(planner_name_issues)} lines\n")

    print("=== Category 3: Mock Hallucinations in Refs ===")
    for line_num, q, issues in mock_ref_issues:
        print(f"Line {line_num}: Q='{q}' - {issues}")
    print(f"Subtotal: {len(mock_ref_issues)} lines\n")

    print("=== Category 4: CoT Engineering Noise ===")
    for line_num, q, issues in think_noise_issues[:20]:
        print(f"Line {line_num}: Q='{q}' - {issues}")
    print(f"Subtotal: {len(think_noise_issues)} lines (showing first 20)\n")


if __name__ == "__main__":
    audit_dataset()
