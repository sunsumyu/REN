import asyncio
import json
import os
import sys
import logging
from datetime import datetime

# Reconfigure stdout/stderr to use UTF-8 under Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from api_client import APIClient
from pipeline import MedicalQAPipeline
from core.pipeline_workflow import SampleQuarantineException
from tests.run_evals import JUDGE_COMPREHENSIVE_PROMPT
from tests.eval_models import ComprehensiveJudgeMetrics
from dataset_db import save_dataset_record
import config

from utils.logging_config import setup_logging
setup_logging(log_file=os.path.join(os.path.dirname(__file__), "pipeline_execution.log"))
logger = logging.getLogger("MedicalQA.Main")

QUALITY_METRIC_LABELS = [
    ("success", "成功度"),
    ("recall", "查全率"),
    ("precision", "精确度"),
    ("faithfulness", "事实忠实度"),
    ("relevance", "相关性"),
    ("professionalism", "专业度"),
    ("interpretability", "可解释性"),
    ("isolation", "领域隔离度"),
    ("complexity", "推演复杂度"),
]

# Global counter stats
stats = {
    "total_attempted": 0,
    "total_passed": 0,
    "total_failed": 0
}


def count_file_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def append_dataset_with_raw_backup(output_file: str, raw_backup_file: str, dataset: dict) -> None:
    """
    Append the accepted raw dataset row to current and raw backup together.
    The raw backup must be line-aligned before appending; otherwise writing is blocked.
    """
    current_count = count_file_lines(output_file)
    raw_exists = os.path.exists(raw_backup_file)
    raw_count = count_file_lines(raw_backup_file) if raw_exists else 0

    if not raw_exists and current_count > 0:
        raise RuntimeError(
            f"Raw backup missing while current dataset already has {current_count} rows; refusing to append unbacked QA."
        )
    if raw_exists and raw_count != current_count:
        raise RuntimeError(
            f"Raw/current line count mismatch before append: raw={raw_count}, current={current_count}; refusing to append."
        )

    payload = json.dumps(dataset, ensure_ascii=False) + "\n"
    output_size_before = os.path.getsize(output_file) if os.path.exists(output_file) else 0
    raw_size_before = os.path.getsize(raw_backup_file) if os.path.exists(raw_backup_file) else 0
    try:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        with open(raw_backup_file, "a", encoding="utf-8") as rf:
            rf.write(payload)
            rf.flush()
            os.fsync(rf.fileno())
    except Exception:
        try:
            with open(output_file, "rb+") as f:
                f.truncate(output_size_before)
        except Exception as rollback_error:
            logger.critical(f"Failed to rollback current dataset append after raw backup write failure: {rollback_error}")
        try:
            if os.path.exists(raw_backup_file):
                with open(raw_backup_file, "rb+") as rf:
                    rf.truncate(raw_size_before)
        except Exception as rollback_error:
            logger.critical(f"Failed to rollback raw backup append after write failure: {rollback_error}")
        raise


def _preview_text(text: str, limit: int = 500) -> str:
    text = (text or "").replace("\r", "").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _extract_answer_body(answer: str) -> str:
    if not answer:
        return ""
    import re
    match = re.match(r"^\s*<think>[\s\S]*?</think>\s*([\s\S]*)$", answer)
    return match.group(1).strip() if match else answer.strip()


def format_dataset_for_quality_judge(dataset: dict) -> str:
    parts = ["【最终 summary】", dataset.get("summary", "").strip()]
    parts.append("\n【各 facet 回答正文】")
    for idx, planner in enumerate(dataset.get("planners", []), start=1):
        facet = planner.get("planner", "")
        answer_body = _extract_answer_body(planner.get("answer", ""))
        parts.append(f"\n[{idx}] facet={facet}\n{_preview_text(answer_body, 1800)}")
    return "\n".join(parts).strip()


def build_quality_gate_audit(
    task_label: str,
    dataset: dict,
    metrics: ComprehensiveJudgeMetrics,
    avg_score: float,
    refusal_avoided: bool,
    is_success: bool,
) -> dict:
    metric_details = {}
    failed_dimensions = []
    for attr, label in QUALITY_METRIC_LABELS:
        metric = getattr(metrics, attr)
        detail = {
            "label": label,
            "score": metric.score,
            "reason": metric.reason,
            "passed": metric.score >= 6.0,
        }
        metric_details[attr] = detail
        if metric.score < 6.0:
            failed_dimensions.append(detail)

    planners = []
    for idx, planner in enumerate(dataset.get("planners", []), start=1):
        answer = planner.get("answer", "")
        planners.append({
            "index": idx,
            "facet": planner.get("planner", ""),
            "has_think": "<think>" in answer and "</think>" in answer,
            "answer_preview": _preview_text(_extract_answer_body(answer), 700),
        })

    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "task_label": task_label,
        "status": "passed" if is_success else "rejected",
        "query": dataset.get("Q", ""),
        "facets": [p["facet"] for p in planners],
        "planners": planners,
        "summary_preview": _preview_text(dataset.get("summary", ""), 1000),
        "refs_count": len(dataset.get("refs", []) or []),
        "avg_score": round(avg_score, 2),
        "refusal_avoided": refusal_avoided,
        "failed_dimensions": failed_dimensions,
        "metrics": metric_details,
    }


def write_generation_rejection_report(audit: dict) -> None:
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    jsonl_path = os.path.join(logs_dir, "generation_rejections.jsonl")
    latest_md_path = os.path.join(logs_dir, "generation_rejections.md")

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit, ensure_ascii=False) + "\n")

    with open(latest_md_path, "a", encoding="utf-8") as f:
        f.write(f"## {audit['time']} | {audit['task_label']} | {audit['status']}\n\n")
        f.write(f"- 主问题: `{audit['query']}`\n")
        f.write(f"- 平均分: {audit['avg_score']}/10\n")
        f.write(f"- 防拒答通过: {audit['refusal_avoided']}\n")
        f.write(f"- refs 数量: {audit['refs_count']}\n")
        f.write(f"- facets: {', '.join(audit['facets']) or 'N/A'}\n\n")

        f.write("### 不达标维度\n\n")
        if audit["failed_dimensions"]:
            for item in audit["failed_dimensions"]:
                f.write(f"- {item['label']}: {item['score']}/10 | {item['reason']}\n")
        elif not audit["refusal_avoided"]:
            f.write("- 防拒答失败: summary 命中拒答/免责声明模式。\n")
        else:
            f.write("- 未发现低于 6 分维度，但综合规则判定未通过。\n")
        f.write("\n### 全量评分理由\n\n")
        for _, label in QUALITY_METRIC_LABELS:
            item = next(v for v in audit["metrics"].values() if v["label"] == label)
            f.write(f"- {label}: {item['score']}/10 | {item['reason']}\n")

        f.write("\n### Planner / Facet 预览\n\n")
        for planner in audit["planners"]:
            f.write(f"- [{planner['index']}] {planner['facet']} | think={planner['has_think']}\n")
            f.write(f"  - answer_preview: {planner['answer_preview']}\n")
        if audit.get("facet_diagnostics"):
            f.write("\n### 逐 Facet 质检定位\n\n")
            for item in audit["facet_diagnostics"]:
                f.write(f"- [{item['index']}] {item['facet']} | avg={item.get('avg_score', 'N/A')}/10 | status={item['status']}\n")
                if item.get("error"):
                    f.write(f"  - error: {item['error']}\n")
                for failed in item.get("failed_dimensions", []):
                    f.write(f"  - {failed['label']}: {failed['score']}/10 | {failed['reason']}\n")
        f.write("\n### Summary 预览\n\n")
        f.write(audit["summary_preview"] + "\n\n---\n\n")


async def evaluate_facets_for_rejected_sample(
    client: APIClient,
    task_label: str,
    query: str,
    refs_text: str,
    dataset: dict,
) -> list:
    diagnostics = []
    for idx, planner in enumerate(dataset.get("planners", []), start=1):
        facet = planner.get("planner", "")
        answer_body = _extract_answer_body(planner.get("answer", ""))
        prompt_answer = f"【单个 facet 待评估】\nfacet={facet}\n{answer_body}"
        judge_prompt = (
            JUDGE_COMPREHENSIVE_PROMPT
            .replace("{{ query }}", query)
            .replace("{{ answer }}", prompt_answer.replace('"', '\\"'))
            .replace("{{ refs_text }}", refs_text)
        )
        try:
            metric: ComprehensiveJudgeMetrics = await client.call_llm_structured(
                [{"role": "user", "content": judge_prompt}],
                ComprehensiveJudgeMetrics,
                model_pool="judge",
                stage=f"[{task_label}] 失败样本逐facet定位-{idx}-{facet}"
            )
            scores = [getattr(metric, attr).score for attr, _ in QUALITY_METRIC_LABELS]
            failed = []
            details = {}
            for attr, label in QUALITY_METRIC_LABELS:
                item = getattr(metric, attr)
                detail = {"label": label, "score": item.score, "reason": item.reason}
                details[attr] = detail
                if item.score < 6.0:
                    failed.append(detail)
            diagnostics.append({
                "index": idx,
                "facet": facet,
                "status": "failed" if failed else "passed",
                "avg_score": round(sum(scores) / len(scores), 2),
                "failed_dimensions": failed,
                "metrics": details,
            })
        except Exception as e:
            diagnostics.append({
                "index": idx,
                "facet": facet,
                "status": "judge_error",
                "error": str(e),
                "failed_dimensions": [],
            })
    return diagnostics

async def generate_and_save_single_task(
    task_idx: int,
    client: APIClient,
    pipeline: MedicalQAPipeline,
    db_write_lock: asyncio.Lock,
    task_semaphore: asyncio.Semaphore
):
    """
    单个数据集生成协程：执行多轮问答生成、质量网关评估、通过互斥锁进行 JSONL/SQLite3 串行双写落盘。
    """
    task_label = f"Task-{task_idx}"
    log_prefix = f"[{task_label}] "
    
    async with task_semaphore:
        logger.info(f"{log_prefix}开始运行数据集生成任务...")
        
        try:
            # 1. 启动多轮对话生成流水线 (传 task_id_label 实现日志标记隔离)
            dataset = await pipeline.generate_multi_round_dataset(task_id_label=task_label)
            
            logger.info(f"{log_prefix}生成完毕，开始进行 Quality Gate 综合质检评估...")
            
            # 2. 从数据集中提取关键字段进行评估
            query = dataset.get('Q', '')
            summary_text = dataset.get('summary', '')
            judge_answer_text = format_dataset_for_quality_judge(dataset)
            refs = dataset.get('refs', []) # 不在此处 pop 掉，保留以供存盘
            
            refs_text = json.dumps(refs, ensure_ascii=False)
            
            judge_prompt = JUDGE_COMPREHENSIVE_PROMPT.replace("{{ query }}", query).replace("{{ answer }}", judge_answer_text.replace('"', '\\"')).replace("{{ refs_text }}", refs_text)
            judge_messages = [{"role": "user", "content": judge_prompt}]
            
            # 3. 大模型裁判打分 (注入 task_label 标签，实现终端耗时表格行号大一统)
            metrics: ComprehensiveJudgeMetrics = await client.call_llm_structured(
                judge_messages, 
                ComprehensiveJudgeMetrics, 
                model_pool="judge",
                stage=f"[{task_label}] 生成数据集质检打分"
            )
            all_scores = [
                metrics.success.score,
                metrics.recall.score,
                metrics.precision.score,
                metrics.faithfulness.score,
                metrics.relevance.score,
                metrics.professionalism.score,
                metrics.interpretability.score,
                metrics.isolation.score,
                metrics.complexity.score
            ]
            avg_score = sum(all_scores) / 9.0
            
            # 4. 防拒答与主观打分双向过关判定
            import re
            refusal_pattern = re.compile(r"(抱歉|无法协助|不符合安全规定|作为一个AI|不能回答|作为AI|未获得授权)")
            refusal_avoided = not bool(refusal_pattern.search(summary_text))
            
            is_success = refusal_avoided and all(s >= 6.0 for s in all_scores)
            
            logger.info(f"{log_prefix}质量网关评估完成 - 平均分: {avg_score:.1f}/10 (通过状态: {'✅ 通过' if is_success else '❌ 拦截'})")
            logger.info(f"{log_prefix}[评分明细] 成功: {metrics.success.score}, 查全: {metrics.recall.score}, 精确: {metrics.precision.score}, 忠实: {metrics.faithfulness.score}, 相关: {metrics.relevance.score}, 专业: {metrics.professionalism.score}, 解释: {metrics.interpretability.score}, 隔离: {metrics.isolation.score}, 复杂: {metrics.complexity.score}")
            quality_audit = build_quality_gate_audit(task_label, dataset, metrics, avg_score, refusal_avoided, is_success)
            
            if not is_success:
                quality_audit["facet_diagnostics"] = await evaluate_facets_for_rejected_sample(
                    client,
                    task_label,
                    query,
                    refs_text,
                    dataset,
                )
                write_generation_rejection_report(quality_audit)
                logger.warning(f"{log_prefix}❌ 质量不达标或触发防拒答拦截，本次生成语料已被丢弃！")
                logger.warning(f"{log_prefix}拦截问题 Q: {query}")
                logger.warning(f"{log_prefix}候选 facets: {', '.join(quality_audit['facets']) or 'N/A'}")
                if not refusal_avoided:
                    logger.warning(f"{log_prefix}失败原因: summary 命中拒答/免责声明模式")
                for item in quality_audit["failed_dimensions"]:
                    logger.warning(
                        f"{log_prefix}不达标维度: {item['label']}={item['score']}/10 | 原因: {item['reason']}"
                    )
                for facet_item in quality_audit.get("facet_diagnostics", []):
                    if facet_item["status"] != "passed":
                        failed_bits = "; ".join(
                            f"{fd['label']}={fd['score']}/10: {fd['reason']}"
                            for fd in facet_item.get("failed_dimensions", [])
                        )
                        logger.warning(
                            f"{log_prefix}可疑 facet[{facet_item['index']}:{facet_item['facet']}] "
                            f"状态={facet_item['status']} avg={facet_item.get('avg_score', 'N/A')}/10 | {failed_bits or facet_item.get('error', '')}"
                        )
                logger.warning(f"{log_prefix}失败审计已写入 logs/generation_rejections.md 和 logs/generation_rejections.jsonl")
                stats["total_failed"] += 1
                return
                
            metrics_dict = json.loads(metrics.model_dump_json())
            
            # 5. 安全排他写锁 (asyncio.Lock)：防止多线程 SQLite 锁死及 JSONL 写入错乱
            async with db_write_lock:
                today_str = datetime.now().strftime("%Y-%m-%d")
                output_file = os.path.join(os.path.dirname(__file__), "medical_qa_dataset.jsonl")
                raw_backup_file = os.path.join(os.path.dirname(__file__), "medical_qa_dataset_raw.jsonl")
                
                # 追加写入 current JSONL 与 raw backup；raw 必须与 current 行号对齐。
                append_dataset_with_raw_backup(output_file, raw_backup_file, dataset)
                    
                # 双写写入 SQLite3 数据库
                save_dataset_record(today_str, query, dataset, metrics_dict)
                
            logger.info(f"{log_prefix}🎉 质检通过！数据已成功写盘入库 (current/raw JSONL / qa_datasets.db)")
            stats["total_passed"] += 1
            
        except SampleQuarantineException as sqe:
            logger.warning(f"{log_prefix}⚠️ 样本已安全隔离并跳过: {sqe}")
            stats["total_failed"] += 1
        except Exception as e:
            logger.error(f"{log_prefix}❌ 任务运行期发生异常: {e}", exc_info=True)
            stats["total_failed"] += 1

async def run_generator():
    start_time = datetime.now()
    logger.info("=============================================================")
    logger.info("🚀 启动企业级高并发批处理问答数据生成流程...")
    logger.info(f"   - 总样本任务数 BATCH_SIZE: {config.BATCH_SIZE}")
    logger.info(f"   - 任务并发度限制 BATCH_CONCURRENCY_LIMIT: {config.BATCH_CONCURRENCY_LIMIT}")
    logger.info(f"   - 全局 API 信号量限制 GLOBAL_API_SEMAPHORE: {config.GLOBAL_API_SEMAPHORE}")
    logger.info("=============================================================")
    
    # 检查环境变量是否包含 API Key
    if not config.LLM_API_KEY:
        logger.warning("警告: 您的 LLM_API_KEY 环境变量未设置！")
        
    client = APIClient()
    pipeline = MedicalQAPipeline(client)
    
    # 初始化串行持久化互斥写锁
    db_write_lock = asyncio.Lock()
    # 初始化任务并发度信号量限制
    task_semaphore = asyncio.Semaphore(config.BATCH_CONCURRENCY_LIMIT)
    
    # 装配批量生成协程任务
    tasks = [
        generate_and_save_single_task(i + 1, client, pipeline, db_write_lock, task_semaphore)
        for i in range(config.BATCH_SIZE)
    ]
    
    stats["total_attempted"] = config.BATCH_SIZE
    
    try:
        # 并发执行全部任务
        await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 打印完美的批处理执行报告
        logger.info("=============================================================")
        logger.info("📊 批量高并发数据集生成完毕 - 执行报告汇总:")
        logger.info(f"   - 总运行耗时: {duration:.1f} 秒")
        logger.info(f"   - 成功生成入库样本数 (Passed): {stats['total_passed']} / {stats['total_attempted']}")
        logger.info(f"   - 丢弃/失败样本数 (Failed): {stats['total_failed']} / {stats['total_attempted']}")
        logger.info(f"   - 最终质量网关通过率: {(stats['total_passed']/stats['total_attempted'])*100:.1f}%")
        logger.info("=============================================================")
        
    except Exception as e:
        logger.critical(f"批处理高并发执行遇到未捕获的严重异常: {e}", exc_info=True)
    finally:
        await client.close()

if __name__ == "__main__":
    try:
        from utils.process_lock import acquire_process_lock
        acquire_process_lock("main_pipeline")
    except Exception as e:
        print(f"⚠️ [ProcessLock Error] Failed to initialize lock: {e}")

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_generator())
