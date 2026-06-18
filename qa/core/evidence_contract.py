# -*- coding: utf-8 -*-
import re
from typing import Any, Dict, List, Optional, Tuple

# 定义禁止扩展的规则集合，用于检测生成内容中是否包含未授权的药代动力学细节、替代药物建议或临床研究证据
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

# 定义否定语境的正则模式列表，用于判断检测到的敏感词是否处于否定语境中（如“未提供”、“无证据”等）
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
    """
    标准化输入值，将其转换为字符串并去除回车符和首尾空白。

    Args:
        value: 任意类型的输入值。

    Returns:
        处理后的标准化字符串。
    """
    return str(value or "").replace("\r", "").strip()


def _ref_key(ref: Dict[str, Any]) -> Tuple[str, str]:
    """
    从参考文献字典中提取来源和上下文，生成唯一的元组键值。

    Args:
        ref: 包含参考文献信息的字典。

    Returns:
        由标准化后的 source 和 context组成的元组。
    """
    return (_normalize_text(ref.get("source")), _normalize_text(ref.get("context")))


def _preview(text: str, limit: int = 420) -> str:
    """
    生成文本的预览片段，如果超过限制长度则截断并添加省略号。

    Args:
        text: 原始文本字符串。
        limit: 预览文本的最大长度，默认为 420。

    Returns:
        截断后的预览字符串。
    """
    text = _normalize_text(text)
    return text[:limit] + ("..." if len(text) > limit else "")


def _scope_lookup(routed_refs: Optional[Dict[str, List[Dict[str, Any]]]]) -> Dict[Tuple[str, str], str]:
    """
    根据路由后的参考文献构建查找表，映射每个参考文献的唯一键值到其作用域状态。

    Args:
        routed_refs: 可选的路由参考文献字典，键为作用域类别（如 CORE, BOUNDARY 等），值为参考文献列表。

    Returns:
        映射参考文献键值到小写作用域字符串的字典。
    """
    lookup: Dict[Tuple[str, str], str] = {}
    if not routed_refs:
        return lookup
    for scope in ("CORE", "BOUNDARY", "BLOCKED", "UNUSED"):
        for ref in routed_refs.get(scope, []) or []:
            if isinstance(ref, dict):
                lookup[_ref_key(ref)] = scope.lower()
    return lookup


def _has_any(text: str, patterns: List[str]) -> bool:
    """
    检查文本中是否包含给定正则模式列表中的任意一个匹配项。

    Args:
        text: 待检查的文本字符串。
        patterns: 正则表达式模式列表。

    Returns:
        如果找到任意匹配项则返回 True，否则返回 False。
    """
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _forbidden_expansions(all_text: str) -> List[str]:
    """
    根据所有参考文献文本生成禁止扩展的消息列表，包括基于规则的检测和数值编造的限制。

    Args:
        all_text: 所有参考文献的来源和上下文拼接而成的字符串。

    Returns:
        禁止扩展的消息字符串列表。
    """
    forbidden = [rule["message"] for rule in _forbidden_families(all_text)]
    if not _has_any(all_text, [r"\d", r"一日", r"每日", r"每次", r"mg", r"ml", r"%"]):
        forbidden.append("不得编造具体剂量、比例、发生率、AUC 增幅等数值。")
    return forbidden


def _forbidden_families(all_text: str) -> List[Dict[str, Any]]:
    """
    识别所有参考文献文本中触发的禁止扩展规则家族。
    仅当源模式未在文本中出现时，才将该规则家族标记为需要检测的目标（逻辑上似乎是反向筛选或特定业务逻辑，此处保留原逻辑）。

    Args:
        all_text: 所有参考文献的来源和上下文拼接而成的字符串。

    Returns:
        触发的禁止规则家族列表，包含 id, label, detect_patterns 和 message。
    """
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
    """
    根据核心事实、边界事实和活跃事实的数量确定证据状态。

    Args:
        core_count: 核心事实的数量。
        boundary_count: 边界事实的数量。
        active_count: 活跃事实的总数（核心 + 边界）。

    Returns:
        证据状态字符串："insufficient"（不足）, "partial"（部分）, 或 "sufficient"（充足）。
    """
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
    构建用于 QA 生成和审计的确定性证据契约。
    该契约记录了允许使用的事实以及被阻止的常见幻觉渠道。

    Args:
        query: 需要标准化并包含在契约中的输入查询字符串。
        refs: 包含来源信息、上下文和元数据的参考字典列表。
        routed_refs: 可选的字典，将参考键映射到其路由状态/作用域。
                     如果提供，它决定参考是 'core'（核心）、'boundary'（边界）、'blocked'（阻塞）还是 'unused'（未使用）。

    Returns:
        代表证据契约的字典，包含：
        - query: 标准化的输入查询。
        - evidence_status: 根据核心和边界事实数量得出的状态字符串。
        - allowed_fact_count: 契约中包含的事实总数。
        - core_fact_count: 分类为 'core' 的事实数量。
        - boundary_fact_count: 分类为 'boundary' 的事实数量。
        - forbidden_expansions: 从所有参考中derived的禁止文本扩展列表。
        - forbidden_term_families: 从所有参考中derived的禁止术语家族列表。
        - facts: 处理后的事实字典列表，包含 ID、支持级别、来源、预览和元数据。
    """
    # 确定每个参考文献的作用域（核心/边界/阻塞/未使用）
    scope_by_key = _scope_lookup(routed_refs)
    facts: List[Dict[str, Any]] = []
    seen = set()
    core_count = 0
    boundary_count = 0

    # 遍历参考文献以过滤、去重和分类事实
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

    # 聚合所有参考文献的来源和上下文文本，用于禁止模式分析
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
    """
    将证据契约渲染为提示词字符串，用于指导模型生成回答。
    包含证据状态、允许事实清单、禁止外推清单以及输出约束。

    Args:
        contract: 由 build_evidence_contract 生成的证据契约字典。
        fact_limit: 允许事实清单中显示的最大事实数量，默认为 14。

    Returns:
        格式化后的提示词字符串。如果契约为空，返回空字符串。
    """
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
    """
    检查文本中指定匹配位置附近的上下文是否包含否定模式。

    Args:
        text: 完整文本字符串。
        start: 匹配项的起始索引。
        end: 匹配项的结束索引。
        window: 向左右扩展的上下文窗口大小，默认为 28。

    Returns:
        如果上下文中存在否定模式则返回 True，否则返回 False。
    """
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    return _has_any(snippet, NEGATION_PATTERNS)


def detect_forbidden_expansion(text: str, contract: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    检测给定文本中是否存在违反证据契约的禁止扩展内容。

    Args:
        text: 待检测的文本字符串。
        contract: 证据契约字典，包含禁止术语家族信息。

    Returns:
        违规项列表，每个违规项包含家族 ID、标签、命中词列表和消息。
    """
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