import sys
import re

def main():
    with open('scripts/medicalqa_purifier.py', 'r', encoding='utf-8') as f:
        content = f.read()

    class_def = """
class PurifierTaskRunner:
    def __init__(self, client, engine, sem, raw_rows, raw_q_index, force_repurify, allow_prune, delete_on_fail):
        self.client = client
        self.engine = engine
        self.sem = sem
        self.raw_rows = raw_rows
        self.raw_q_index = raw_q_index
        self.force_repurify = force_repurify
        self.allow_prune = allow_prune
        self.delete_on_fail = delete_on_fail
        self.audit_events = []
        self.purified_diff_logs = []

    def record_audit_event(self, event: dict) -> None:
        import datetime
        event.setdefault("run_status", "unknown")
        event.setdefault("time", datetime.datetime.now().isoformat(timespec="seconds"))
        self.audit_events.append(event)

    async def _process_single_planner(self, p, q, refs, line_idx, planners_count, raw_mapping_meta):
        import copy, re
        from config import PURIFY_DELETE_ON_FAIL
        original_planner = copy.deepcopy(p)
        try:
            raw_planner_name = p.get("planner", "")
            raw_answer = p.get("answer", "")
                
            TEMPLATE_SIGNATURES = [
                "触发物理格式崩溃",
                "质量网关硬指标",
                "自动重新净化重写",
            ]
            SAFE_BODY_SIGNATURES = [
                "根据参考资料",
                "现有资料未提供",
            ]
                
            if any(sig in raw_answer for sig in TEMPLATE_SIGNATURES):
                logger.warning(f"  🚨 Rollback Line {line_idx+1} facet '{raw_planner_name}' due to template signature.")
                return original_planner, None, "failed", {"planner": raw_planner_name, "status": "failed", "reason": "template signature detected"}
                    
            if any(sig in raw_answer for sig in SAFE_BODY_SIGNATURES):
                logger.warning(f"  🚨 Rollback Line {line_idx+1} facet '{raw_planner_name}' due to safety warning signature.")
                return original_planner, None, "failed", {"planner": raw_planner_name, "status": "failed", "reason": "safety warning signature detected"}
                
            is_valid = await verify_facet_by_small_model(self.client, raw_planner_name)
            if not is_valid:
                logger.warning(f"  🚨 [小模型网关校验拦截] 发现行 {line_idx+1} 的非医学视角占位符: '{raw_planner_name}'。仅在成功清洗后写入安全视角名。")
                planner_name = infer_safe_repair_facet(q, raw_planner_name) or "临床用药安全"
            else:
                planner_name = raw_planner_name
                
            compatibility = await check_semantic_compatibility(self.client, q, planner_name)
            if compatibility == "FORCED_SKIP":
                should_downgrade, repaired_facet, repair_reason = should_downgrade_forced_skip(q, planner_name, planners_count)
                if should_downgrade:
                    old_planner_name = planner_name
                    planner_name = repaired_facet or planner_name
                    compatibility = "COMPATIBLE_SIMPLE"
                    is_valid = True
                    logger.warning(
                        f"  🟡 [FORCED_SKIP降级修复] 行 {line_idx+1} 的视角 '{old_planner_name}' "
                        f"改为 '{planner_name}' 并进入极简清洗；原因: {repair_reason}。"
                    )
                else:
                    if self.allow_prune:
                        logger.critical(f"  ✂️ [企业级网关强行视角剪枝] 行 {line_idx+1} 的强套视角 '{planner_name}' 被显式剪枝。")
                        return None, None, "pruned", {"planner": planner_name, "status": "pruned", "reason": "semantic compatibility forced skip"}
                    logger.critical(f"  🚨 [企业级网关强行视角拦截] 行 {line_idx+1} 的强套视角 '{planner_name}' 未通过；保留原 planner 并回滚整行。")
                    return original_planner, None, "failed", {"planner": planner_name, "status": "failed", "reason": "semantic compatibility forced skip without prune permission"}
                
            simplify = (compatibility == "COMPATIBLE_SIMPLE")
            if simplify:
                logger.info(f"  💡 [企业级网关简化提示] 行 {line_idx+1} 的视角 '{planner_name}' 与简单问题匹配，开启极简重构。")
                
            think_match = re.search(r"<think>([\s\S]*?)</think>([\s\S]*)$", raw_answer, re.IGNORECASE)
            if think_match:
                raw_think = think_match.group(1).strip()
                answer_body = think_match.group(2).strip()
                    
                purified_answer_body = await rewrite_answer_body(
                    self.client.llm_service,
                    q,
                    answer_body,
                    refs
                )
                original_answer_body = answer_body
                answer_body_after_leakage_scrub = scrub_engineering_leakage(purified_answer_body)
                purified_answer_body = scrub_unsupported_official_identifiers(answer_body_after_leakage_scrub, refs)

                facet_match = re.search(r"(<facet\s*=\s*[^>]+>)\s*([\s\S]*)$", raw_think)
                if facet_match:
                    facet_tag = facet_match.group(1).strip()
                    actual_raw_think = facet_match.group(2).strip()
                    if planner_name != raw_planner_name:
                        facet_tag = f"<facet = {planner_name}>"
                else:
                    facet_tag = f"<facet = {planner_name}>"
                    actual_raw_think = raw_think
                    
                async with self.sem:
                    logger.info(f"⏳ Processing Record {line_idx+1}: Q='{q[:12]}...' | Facet='{planner_name}'")
                    purified_think, score_dict = await purify_single_think(
                        self.engine, q, planner_name, actual_raw_think, purified_answer_body,
                        line_num=line_idx+1, refs=refs, simplify=simplify
                    )
                    
                if not score_dict.get("is_passed", False):
                    logger.critical(f"   ❌ [Hallucination/Quality Gate Intercept] Keep original facet '{planner_name}' for Line {line_idx+1} due to Quality Gate Failure.")
                    return original_planner, None, "failed", {
                        "planner": planner_name,
                        "status": "failed",
                        "reason": "quality gate failure",
                        "scores": score_dict,
                    }
                    
                p_new = copy.deepcopy(p)
                p_new["planner"] = planner_name
                purified_full_answer = f"<think>\\n{facet_tag}\\n{purified_think}\\n</think>\\n{purified_answer_body}"
                p_new["answer"] = purified_full_answer

                operations = [
                    "rewrite_think_with_fact_anchored_purification_engine",
                    f"semantic_compatibility={compatibility}",
                ]
                if planner_name != raw_planner_name:
                    operations.append(f"repair_planner_name:{raw_planner_name}->{planner_name}")
                if simplify:
                    operations.append("apply_simplified_reasoning_mode")
                if purified_answer_body != original_answer_body:
                    operations.append("rewrite_and_narrow_answer_body")
                if answer_body_after_leakage_scrub != purified_answer_body:
                    operations.append("scrub_engineering_leakage_from_answer_body")
                if purified_answer_body != answer_body_after_leakage_scrub:
                    operations.append("scrub_unsupported_official_identifiers_from_answer_body")
                    
                diff_log = {
                    "line_number": line_idx + 1,
                    "question": q,
                    "facet": planner_name,
                    "original_planner": raw_planner_name,
                    "final_planner": planner_name,
                    "original_think": raw_think,
                    "purified_think": purified_think,
                    "original_answer_body": original_answer_body,
                    "purified_answer_body": answer_body,
                    "purified_full_answer": purified_full_answer,
                    "operations": operations,
                    "scores": score_dict,
                    "compatibility": compatibility,
                    "simplify": simplify
                }
                return p_new, diff_log, "success", {"planner": planner_name, "status": "success", "reason": "purified", "compatibility": compatibility}
            else:
                return original_planner, None, "unchanged", {"planner": raw_planner_name, "status": "unchanged", "reason": "no think block"}
        except Exception as e:
            logger.error(f"❌ [Exception Intercept] Exception occurred when purifying line {line_idx+1} facet '{p.get('planner', '')}': {e}. Keep original planner and rollback this line.")
            return original_planner, None, "failed", {"planner": p.get("planner", ""), "status": "failed", "reason": f"exception: {e}"}

    async def process_record(self, line_idx, line_str, should_purify=True, skip_reason: str = ""):
        import json, copy, asyncio
        if not line_str.strip():
            return line_str
            
        try:
            original_data = json.loads(line_str)
            if not should_purify:
                if skip_reason:
                    self.record_audit_event({
                        "line_number": line_idx + 1,
                        "run_status": "skipped",
                        "reason": skip_reason,
                        "question": original_data.get("Q", ""),
                    })
                return line_str

            if not self.force_repurify and "refs" not in original_data and "history" not in original_data:
                logger.info(f"⏭️ Skip Line {line_idx+1}: no refs/history found; treated as already purified.")
                self.record_audit_event({
                    "line_number": line_idx + 1,
                    "run_status": "skipped_already_purified",
                    "reason": "no refs/history found",
                    "question": original_data.get("Q", ""),
                })
                return line_str

            data = copy.deepcopy(original_data)
            refs = data.get("refs", [])
            raw_mapping_meta = {
                "raw_line": None,
                "raw_mapping_status": "NOT_USED",
                "raw_mapping_warning": "",
            }

            if self.force_repurify:
                raw_record, raw_mapping_meta = resolve_raw_record_for_current(
                    original_data,
                    line_idx + 1,
                    self.raw_rows,
                    self.raw_q_index,
                )
                if raw_mapping_meta.get("raw_mapping_warning"):
                    logger.warning(f"🚨 {raw_mapping_meta['raw_mapping_warning']}")
                if raw_record is None:
                    reason = (
                        "raw mapping failed during force re-purify; refusing to purify without safe raw refs/history "
                        f"({raw_mapping_meta.get('raw_mapping_status')})"
                    )
                    logger.critical(f"🚨 Rollback Line {line_idx+1}: {reason}")
                    self.record_audit_event({
                        "line_number": line_idx + 1,
                        "run_status": "error",
                        "reason": reason,
                        "question": original_data.get("Q", ""),
                        **raw_mapping_meta,
                    })
                    return None if self.delete_on_fail else line_str

                if not refs and raw_record.get("refs"):
                    refs = raw_record.get("refs", [])
                    data["refs"] = refs
                if "history" not in data and "history" in raw_record:
                    data["history"] = copy.deepcopy(raw_record.get("history", []))
            
            q = data.get("Q", "")
            planners = data.get("planners", [])
                
            planner_tasks = [self._process_single_planner(p, q, refs, line_idx, len(planners), raw_mapping_meta) for p in planners]
            planner_results = await asyncio.gather(*planner_tasks)
                
            valid_planners = []
            local_diff_logs = []
            has_failed = False
            pruned_count = 0
            planner_audit = []
            for p_new, diff_log, status, planner_event in planner_results:
                planner_audit.append(planner_event)
                if status == "failed":
                    has_failed = True
                if status == "pruned":
                    pruned_count += 1
                if p_new is not None:
                    valid_planners.append(p_new)
                if diff_log:
                    local_diff_logs.append(diff_log)

            if has_failed or (len(valid_planners) + pruned_count) != len(planners) or (planners and not valid_planners):
                logger.warning(f"↩️ Rollback Line {line_idx+1}: planner purification failed or planner count changed ({len(planners)} -> {len(valid_planners)}).")
                self.record_audit_event({
                    "line_number": line_idx + 1,
                    "run_status": "rollback",
                    "reason": "planner purification failed or planner count changed",
                    "question": q,
                    **raw_mapping_meta,
                    "planner_count_before": len(planners),
                    "planner_count_after": len(valid_planners),
                    "pruned_count": pruned_count,
                    "planners": planner_audit,
                })
                return None if self.delete_on_fail else line_str

            if not local_diff_logs and pruned_count == 0:
                self.record_audit_event({
                    "line_number": line_idx + 1,
                    "run_status": "unchanged",
                    "reason": "no local diff logs and no pruned planners",
                    "question": q,
                    **raw_mapping_meta,
                    "planner_count_before": len(planners),
                    "planner_count_after": len(valid_planners),
                    "planners": planner_audit,
                })
                return line_str

            original_summary = data.get("summary", "")
            try:
                regenerated_summary = await regenerate_summary_after_purification(
                    self.client.llm_service,
                    q,
                    valid_planners,
                    refs,
                    line_idx + 1,
                )
            except Exception as e:
                logger.critical(f"↩️ Rollback Line {line_idx+1}: summary regeneration failed after planner purification: {e}")
                self.record_audit_event({
                    "line_number": line_idx + 1,
                    "run_status": "rollback",
                    "reason": f"summary regeneration failed: {e}",
                    "question": q,
                    **raw_mapping_meta,
                    "planner_count_before": len(planners),
                    "planner_count_after": len(valid_planners),
                    "pruned_count": pruned_count,
                    "planners": planner_audit,
                })
                return None if self.delete_on_fail else line_str

            for item in local_diff_logs:
                item["original_summary"] = original_summary
                item["regenerated_summary"] = regenerated_summary
                item.setdefault("operations", []).append("regenerate_summary_after_planner_purification")

            data["planners"] = valid_planners
            data["summary"] = regenerated_summary
            data.pop("history", None)
            data.pop("refs", None)
            self.purified_diff_logs.extend(local_diff_logs)
            self.record_audit_event({
                "line_number": line_idx + 1,
                "run_status": "success",
                "reason": "purified",
                "question": q,
                **raw_mapping_meta,
                "planner_count_before": len(planners),
                "planner_count_after": len(valid_planners),
                "pruned_count": pruned_count,
                "purified_facets": len(local_diff_logs),
                "summary_regenerated": True,
                "planners": planner_audit,
            })
            return json.dumps(data, ensure_ascii=False) + "\\n"
        except Exception as e:
            logger.error(f"❌ Error processing line {line_idx+1}: {e}")
            self.record_audit_event({
                "line_number": line_idx + 1,
                "run_status": "error",
                "reason": str(e),
            })
            return None if self.delete_on_fail else line_str
"""

    pattern = re.compile(r"(\s+sem = asyncio\.Semaphore\(PURIFY_CONCURRENCY\)\s+purified_diff_logs = \[\]\s+audit_events = \[\]\s+force_repurify = .*?)\s+purify_counter = 0", re.DOTALL)
    
    match = pattern.search(content)
    if not match:
        print("Could not find the block to replace!")
        sys.exit(1)
        
    original_block = match.group(1)
    
    content = content.replace("async def main():", class_def + "\nasync def main():")
    
    replacement = """
    sem = asyncio.Semaphore(PURIFY_CONCURRENCY)
    force_repurify = os.getenv("PURIFY_FORCE_REPURIFY", "").strip().lower() in {"1", "true", "yes", "on"}
    allow_prune = os.getenv("PURIFY_ALLOW_PRUNE", "").strip().lower() in {"1", "true", "yes", "on"}
    
    runner = PurifierTaskRunner(
        client=client,
        engine=engine,
        sem=sem,
        raw_rows=raw_rows,
        raw_q_index=raw_q_index,
        force_repurify=force_repurify,
        allow_prune=allow_prune,
        delete_on_fail=PURIFY_DELETE_ON_FAIL
    )
"""
    content = content.replace(original_block, replacement)
    
    content = content.replace("tasks.append(process_record(i, line, should_purify, skip_reason))", "tasks.append(runner.process_record(i, line, should_purify, skip_reason))")
    
    content = content.replace("for item in purified_diff_logs:", "for item in runner.purified_diff_logs:")
    content = content.replace("for event in audit_events:", "for event in runner.audit_events:")
    content = content.replace("if purified_diff_logs:", "if runner.purified_diff_logs:")
    content = content.replace("if audit_events:", "if runner.audit_events:")
    content = content.replace("json.dump(purified_diff_logs,", "json.dump(runner.purified_diff_logs,")
    content = content.replace("json.dump(audit_events,", "json.dump(runner.audit_events,")

    with open('scripts/medicalqa_purifier_new.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Refactored into scripts/medicalqa_purifier_new.py successfully!")

if __name__ == "__main__":
    main()
