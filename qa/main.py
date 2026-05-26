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
from run_evals import JUDGE_COMPREHENSIVE_PROMPT
from eval_models import ComprehensiveJudgeMetrics
from dataset_db import save_dataset_record

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "pipeline_execution.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("MedicalQA.Main")

async def run_generator():
    logger.info("Initializing Medical QA Multi-Round Dataset Generator...")
    
    # Check if API Key is set
    import config
    if not config.LLM_API_KEY:
        print("="*60)
        print("警告: 您的 LLM_API_KEY 环境变量未设置！")
        print("="*60)
        
    client = APIClient()
    pipeline = MedicalQAPipeline(client)
    
    try:
        # Generate the multi-round trajectory dataset
        dataset = await pipeline.generate_multi_round_dataset()
        
        print("\n" + "="*60)
        print("🚀 数据生成完毕，准备进入 Quality Gate (运行期综合质量网关打分)...")
        print(f"⏳ 触发限流主动冷切避让机制 - 冷却休眠 {config.SOFT_DELAY_BEFORE_JUDGE} 秒...")
        await asyncio.sleep(config.SOFT_DELAY_BEFORE_JUDGE)
        
        # QUALITY GATE EVALUATION
        query = dataset.get('Q', '')
        summary_text = dataset.get('summary', '')
        
        # 提取并解析该轮次的真实权威文献事实 (Guideline & Graph Grounding Refs)
        refs = dataset.pop('refs', [])
        refs_text = json.dumps(refs, ensure_ascii=False)
        
        judge_prompt = JUDGE_COMPREHENSIVE_PROMPT.replace("{{ query }}", query).replace("{{ answer }}", summary_text.replace('"', '\\"')).replace("{{ refs_text }}", refs_text)
        judge_messages = [{"role": "user", "content": judge_prompt}]
        
        try:
            metrics: ComprehensiveJudgeMetrics = await client.call_llm_structured(judge_messages, ComprehensiveJudgeMetrics, model_pool="judge")
            all_scores = [
                metrics.grounding.score,
                metrics.isolation.score,
                metrics.explainability.score,
                metrics.professionalism.score,
                metrics.relevance.score
            ]
            avg_score = sum(all_scores) / 5.0
            
            # 判决逻辑: 防拒答必须成功（摘要不含抱歉/无法回答等），且所有主观打分>=6.0
            import re
            refusal_pattern = re.compile(r"(抱歉|无法协助|不符合安全规定|作为一个AI|不能回答|作为AI|未获得授权)")
            refusal_avoided = not bool(refusal_pattern.search(summary_text))
            
            is_success = refusal_avoided and all(s >= 6.0 for s in all_scores)
            
            print(f"📊 质量网关评估完成 - 平均分: {avg_score:.1f}/10 (通过状态: {'✅ 通过' if is_success else '❌ 拦截'})")
            print(f"   [详尽得分] 忠实度: {metrics.grounding.score}, 隔离: {metrics.isolation.score}, 可解释: {metrics.explainability.score}, 专业: {metrics.professionalism.score}, 相关: {metrics.relevance.score}")
            
            if not is_success:
                logger.warning("Generated dataset failed the Quality Gate. It will not be saved.")
                print("⚠️  当前生成语料质量不达标，已被丢弃。")
                return
                
            metrics_dict = json.loads(metrics.model_dump_json())
            
        except Exception as e:
            logger.error(f"Quality Gate Error: {e}")
            print("⚠️  质量网关发生异常，为了安全起见，本次数据将不予入库。")
            return
            
        # If success, proceed to save (Append to medical_qa_dataset.jsonl)
        today_str = datetime.now().strftime("%Y-%m-%d")
        output_file = os.path.join(os.path.dirname(__file__), "medical_qa_dataset.jsonl")
        
        # Append as a single line in JSONL format
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(dataset, ensure_ascii=False) + "\n")
            
        # Double-write to SQLite database
        save_dataset_record(today_str, query, dataset, metrics_dict)
            
        print("\n" + "="*60)
        print("🎉 医药知识图谱多视角多轮问答数据集通过质检，成功入库！")
        print(f"📁 本地追加文件: {output_file}")
        print(f"💾 数据库入库: qa_datasets.db")
        print("="*60)
        
    except Exception as e:
        logger.critical(f"Pipeline execution encountered an unhandled exception: {e}", exc_info=True)
        print(f"\n❌ 生成失败: {e}", file=sys.stderr)
    finally:
        await client.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_generator())
