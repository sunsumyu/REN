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
from tests.run_evals import JUDGE_COMPREHENSIVE_PROMPT
from tests.eval_models import ComprehensiveJudgeMetrics
from dataset_db import save_dataset_record
import config

from utils.logging_config import setup_logging
setup_logging(log_file=os.path.join(os.path.dirname(__file__), "pipeline_execution.log"))
logger = logging.getLogger("MedicalQA.Main")

# Global counter stats
stats = {
    "total_attempted": 0,
    "total_passed": 0,
    "total_failed": 0
}

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
            refs = dataset.get('refs', []) # 不在此处 pop 掉，保留以供存盘
            
            refs_text = json.dumps(refs, ensure_ascii=False)
            
            judge_prompt = JUDGE_COMPREHENSIVE_PROMPT.replace("{{ query }}", query).replace("{{ answer }}", summary_text.replace('"', '\\"')).replace("{{ refs_text }}", refs_text)
            judge_messages = [{"role": "user", "content": judge_prompt}]
            
            # 3. 大模型裁判打分 (注入 task_label 标签，实现终端耗时表格行号大一统)
            metrics: ComprehensiveJudgeMetrics = await client.call_llm_structured(
                judge_messages, 
                ComprehensiveJudgeMetrics, 
                model_pool="judge",
                stage=f"[{task_label}] 生成数据集质检打分"
            )
            all_scores = [
                metrics.grounding.score,
                metrics.isolation.score,
                metrics.explainability.score,
                metrics.professionalism.score,
                metrics.relevance.score
            ]
            avg_score = sum(all_scores) / 5.0
            
            # 4. 防拒答与主观打分双向过关判定
            import re
            refusal_pattern = re.compile(r"(抱歉|无法协助|不符合安全规定|作为一个AI|不能回答|作为AI|未获得授权)")
            refusal_avoided = not bool(refusal_pattern.search(summary_text))
            
            is_success = refusal_avoided and all(s >= 6.0 for s in all_scores)
            
            logger.info(f"{log_prefix}质量网关评估完成 - 平均分: {avg_score:.1f}/10 (通过状态: {'✅ 通过' if is_success else '❌ 拦截'})")
            logger.info(f"{log_prefix}[评分明细] 忠实度: {metrics.grounding.score}, 隔离: {metrics.isolation.score}, 可解释: {metrics.explainability.score}, 专业: {metrics.professionalism.score}, 相关: {metrics.relevance.score}")
            
            if not is_success:
                logger.warning(f"{log_prefix}❌ 质量不达标或触发防拒答拦截，本次生成语料已被丢弃！")
                stats["total_failed"] += 1
                return
                
            metrics_dict = json.loads(metrics.model_dump_json())
            
            # 5. 安全排他写锁 (asyncio.Lock)：防止多线程 SQLite 锁死及 JSONL 写入错乱
            async with db_write_lock:
                today_str = datetime.now().strftime("%Y-%m-%d")
                output_file = os.path.join(os.path.dirname(__file__), "medical_qa_dataset.jsonl")
                
                # 追加写入 JSONL 文件
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dataset, ensure_ascii=False) + "\n")
                    
                # 双写写入 SQLite3 数据库
                save_dataset_record(today_str, query, dataset, metrics_dict)
                
            logger.info(f"{log_prefix}🎉 质检通过！数据已成功双向写盘入库 (medical_qa_dataset.jsonl / qa_datasets.db)")
            stats["total_passed"] += 1
            
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
