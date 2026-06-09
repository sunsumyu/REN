# -*- coding: utf-8 -*-
import re
from typing import Any, Dict, List, Optional, Tuple

FORBIDDEN_EXPANSION_RULES = [
    {
        "id": "pharmacokinetics",
        "label": "药代细节",
        "source_patterns": [r"\bCYP\b", r"AUC", r"半衰期", r"首过", r"药代", r"清除率", r"代谢酶"],
        "detect_patterns": [r"\bCYP(?:3A4|2D6|2C9|2C19)?\b", r"AUC", r"半衰期", r"首过效应", r"清除率", r"代谢酶", r"药代动力学"],
        "message": "不得引入 CYP/AUC/半衰期/首过效应等药代细节。",
    },
    {
        "id": "alternative_drug",
        "label": "替代药物",
        "source_patterns": [r"替代", r"可换用", r"换用", r"其他药", r"替换", r"备选", r"可选"],
        "detect_patterns": [r"奥洛他定", r"酮替芬", r"西替利嗪", r"氯雷他定", r"孟鲁司特", r"替代药物", r"替代用药", r"改用", r"换用"],
        "message": "不得引入 refs 未提及的替代药物名称或换药方案。",
    },
    {
        "id": "clinical_study",
        "label": "临床研究证据",
        "source_patterns": [r"临床研究", r"临床试验", r"PMID", r"随机", r"双盲", r"队列", r"研究显示"],
        "detect_patterns": [r"临床研究", r"临床试验", r"PMID", r"随机", r"双盲", r"队列研究", r"研究显示"],
        "message": "不得声称 refs 未提供的临床研究、试验设计或文献证据。",
    },
]

NEGATION_PATTERNS = [
    r"未提供",
    r"未提及",
    r"无证据",
    r"没有证据",
    r"证据不足",
    r"不能判断",
    r"无法判断",
    r"不得",
    r"严禁",
    r"不应",
    r"不可",
    r"不能据此",
    r"缺乏",
]


def _normalize_text(value: Any) -> str:
    return str(value or "").replace("\r", "").strip()


def _ref_key(ref: Dict[str, Any]) -> Tuple[str, str]:
    return (_normalize_text(ref.get("source")), _normalize_text(ref.get("context")))


def _preview(text: str, limit: int = 420) -> str:
    text = _normalize_text(text)
    return text[:limit] + ("..." if len(text) > limit else "")


def _scope_lookup(routed_refs: Optional[Dict[str, List[Dict[str, Any]]]]) -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}
    if not routed_refs:
        return lookup
    for scope in ("CORE", "BOUNDARY", "BLOCKED", "UNUSED"):
        for ref in routed_refs.get(scope, []) or []:
            if isinstance(ref, dict):
                lookup[_ref_key(ref)] = scope.lower()
    return lookup


def _has_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _forbidden_expansions(all_text: str) -> List[str]:
    forbidden = [rule["message"] for rule in _forbidden_families(all_text)]
    if not _has_any(all_text, [r"\d", r"一日", r"每日", r"每次", r"mg", r"ml", r"%"]):
        forbidden.append("不得编造具体剂量、比例、发生率、AUC 增幅等数值。")
    return forbidden


def _forbidden_families(all_text: str) -> List[Dict[str, Any]]:
    families = []
    for rule in FORBIDDEN_EXPANSION_RULES:
        if not _has_any(all_text, rule["source_patterns"]):
            families.append({
                "id": rule["id"],
                "label": rule["label"],
                "detect_patterns": rule["detect_patterns"],
                "message": rule["message"],
            })
    return families


def _status_from_counts(core_count: int, boundary_count: int, active_count: int) -> str:
    if active_count <= 0:
        return "insufficient"
    if core_count <= 0 and boundary_count < 2:
        return "insufficient"
    if core_count <= 0:
        return "partial"
    if active_count < 2:
        return "partial"
    return "sufficient"


def build_evidence_contract(
    query: str,
    refs: List[Dict[str, Any]],
    routed_refs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Build a deterministic evidence contract for QA generation and audit.
    It records what facts are allowed and which common hallucination channels are blocked.
    """
    scope_by_key = _scope_lookup(routed_refs)
    facts: List[Dict[str, Any]] = []
    seen = set()
    core_count = 0
    boundary_count = 0

    for ref in refs or []:
        if not isinstance(ref, dict):
            continue
        key = _ref_key(ref)
        if key in seen or not key[1]:
            continue
        seen.add(key)

        scope = scope_by_key.get(key, "boundary" if not routed_refs else "unused")
        if scope == "blocked" or scope == "unused":
            continue
        if scope == "core":
            core_count += 1
        elif scope == "boundary":
            boundary_count += 1

        facts.append({
            "fact_id": f"F{len(facts) + 1:03d}",
            "support_level": scope,
            "source": key[0],
            "context_preview": _preview(key[1]),
            "metadata": ref.get("metadata", {}),
        })

    all_text = "\n".join(
        f"{_normalize_text(ref.get('source'))}\n{_normalize_text(ref.get('context'))}"
        for ref in refs or []
        if isinstance(ref, dict)
    )
    active_count = core_count + boundary_count

    return {
        "query": _normalize_text(query),
        "evidence_status": _status_from_counts(core_count, boundary_count, active_count),
        "allowed_fact_count": len(facts),
        "core_fact_count": core_count,
        "boundary_fact_count": boundary_count,
        "forbidden_expansions": _forbidden_expansions(all_text),
        "forbidden_term_families": _forbidden_families(all_text),
        "facts": facts,
    }


def render_evidence_contract_prompt(contract: Optional[Dict[str, Any]], fact_limit: int = 14) -> str:
    if not contract:
        return ""

    facts = contract.get("facts", []) or []
    fact_lines = []
    for fact in facts[:fact_limit]:
        fact_lines.append(
            f"- [{fact.get('fact_id')}] [{fact.get('support_level')}] "
            f"{fact.get('source')}: {fact.get('context_preview')}"
        )
    if len(facts) > fact_limit:
        fact_lines.append(f"- ...其余 {len(facts) - fact_limit} 条允许事实仅可按原文范围使用。")

    forbidden = contract.get("forbidden_expansions", []) or []
    forbidden_lines = [f"- {item}" for item in forbidden] or ["- 不得引入允许事实清单之外的新医学事实。"]

    return (
        "\n\n【证据契约硬约束】\n"
        f"- 证据状态: {contract.get('evidence_status', 'unknown')}\n"
        f"- 允许事实数: {contract.get('allowed_fact_count', 0)}\n"
        "- 只能使用下列允许事实进行推理；若当前 facet 所需信息不在允许事实中，必须说明证据不足，严禁用常识补齐具体药物事实。\n"
        "【允许事实清单】\n"
        f"{chr(10).join(fact_lines) if fact_lines else '- 无可用允许事实。'}\n"
        "【禁止外推清单】\n"
        f"{chr(10).join(forbidden_lines)}\n"
        "【输出约束】\n"
        "- evidences 字段必须引用允许事实清单中的事实，不得伪造来源、章节、研究或说明书内容。\n"
        "- reasoning_chains 与 answer_body 中的每个关键医学判断都必须能回扣到允许事实；证据不足时输出边界判断，不要扩写。\n"
    )


def _is_negated_context(text: str, start: int, end: int, window: int = 28) -> bool:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    return _has_any(snippet, NEGATION_PATTERNS)


def detect_forbidden_expansion(text: str, contract: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not text or not contract:
        return []

    violations = []
    for family in contract.get("forbidden_term_families", []) or []:
        hits = []
        for pattern in family.get("detect_patterns", []) or []:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                if not _is_negated_context(text, match.start(), match.end()):
                    hits.append(match.group(0))
        if hits:
            violations.append({
                "family": family.get("id"),
                "label": family.get("label"),
                "hits": sorted(set(hits)),
                "message": family.get("message"),
            })
    return violations
