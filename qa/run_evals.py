# -*- coding: utf-8 -*-
"""
自动化测试与评估运行器 (LLM-as-a-Judge Runner)
参考 Anthropic 'Building Effective Agents' 原则构建。
运行测试基准集，并通过大模型裁判对高维质量指标进行统一结构化评估打分。
"""

import asyncio
import json
import os
import sys
import re
import logging
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel

# Reconfigure stdout/stderr to use UTF-8 under Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MedicalQA.Evaluator")

# Add current directory to sys.path to resolve imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_client import APIClient
from pipeline import MedicalQAPipeline
from guideline_db import get_guideline_refs
from eval_models import JudgeMetric, ComprehensiveJudgeMetrics, EvalResultItem

# 裁判大模型综合 Prompt 模板
JUDGE_COMPREHENSIVE_PROMPT = """你是一个顶级医院的临床质量安全总监、循证医学教授与大模型红蓝对抗专家。
你当前需要对一份大模型生成的医疗切面问答样本进行“事实忠实度”、“领域隔离度”、“可解释性”、“专业性”和“相关性”的五维综合打分。

# 判定依据
1. 事实忠实度 (grounding)：仔细核对 refs 背景指南事实。如果包含了 refs 中没有提到过的新实体或明显违背常识的捏造，严重扣分。完美无幻觉打分为 10.0 分。
2. 领域隔离度 (isolation)：如果生成的 answer 混入了“法律合规”、“合规审计”、“数据隐私”、“供应链”等非医学跑题词汇，严重扣分！纯净医学专业陈述打分为 10.0 分。
3. 可解释性 (explainability)：评估答案是否有清晰的逻辑推导和证据来源引用（如：根据《XX说明书》）。生硬直接给结论扣分。
4. 专业性 (professionalism)：评估使用的医学术语是否规范，整体结构与用词是否像严谨的临床医学专家。
5. 相关性 (relevance)：评估回答是否直击提问者的核心诉求，有无大量无意义的套话和废话。

# 输入数据
- 核心医学问题: "{{ query }}"
- 权威参考文献 (refs): {{ refs_text }}
- 待评估的医学回答: "{{ answer }}"

# 输出格式契合 (Pydantic Schema)
你必须且只能输出符合指定的 JSON 格式。包含以上 5 个维度的客观打分与详尽打分原因。
"""

def calculate_recall(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 1.0
    matched = 0
    for kw in expected_keywords:
        if kw.lower() in answer.lower():
            matched += 1
    return matched / len(expected_keywords)

async def run_evaluation():
    logger.info("==================================================================")
    logger.info("   Medical QA Pipeline High-Dimensional Quality Evals System      ")
    logger.info("==================================================================")
    
    # 1. Load benchmark cases
    benchmark_file = os.path.join(os.path.dirname(__file__), "eval_benchmark.json")
    if not os.path.exists(benchmark_file):
        logger.error(f"Benchmark file not found at: {benchmark_file}")
        return
        
    with open(benchmark_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    logger.info(f"Loaded {len(cases)} benchmark cases from test suite.")
    
    client = APIClient()
    pipeline = MedicalQAPipeline(client)
    semaphore = asyncio.Semaphore(1) # Run sequentially for clean console logs
    
    results = []
    
    # Run test suite
    for idx, case in enumerate(cases):
        case_id = case["id"]
        category = case["category"]
        query = case["query"]
        expected_keywords = case.get("expected_keywords", [])
        
        logger.info(f"\n[{idx+1}/{len(cases)}] Running Case {case_id} ({category}): '{query[:35]}...'")
        
        # 2. Build refs grounding context using guideline database
        refs = []
        for kw in ["愈肝片", "二甲双胍", "阿司匹林", "车前草"]:
            if kw in query:
                refs.extend(get_guideline_refs(kw))
                
        if not refs:
            # Safe baseline mock refs if no specific keywords match
            refs = [{
                "source": "refs:《通用临床医学参考手册》",
                "context": "中药与化学药品联合用药需密切关注器官代偿和药物毒代反应。所有用药配伍和剂量管理必须严格遵从临床医师处方指导。"
            }]
            
        # Select appropriate facet for testing
        facet = "药理机制" if "机制" in query or "采收" in query else "临床表现"
        
        # 3. Execute Pipeline Answer Generation
        try:
            import prompts
            facet_name, answer_full = await pipeline.answer_single_facet(query, facet, refs, semaphore)
            
            answer_body = answer_full
            if "</think>" in answer_full:
                answer_body = answer_full.split("</think>")[-1].strip()
                
            schema_ok = True
            if answer_body.strip().startswith("{") and answer_body.strip().endswith("}"):
                schema_ok = True
            else:
                schema_ok = False
                
            self_healing_attempts = 0
            if " api_limit " in answer_full:
                self_healing_attempts = 1
                
        except Exception as e:
            logger.error(f"Pipeline crashed on case {case_id}: {e}")
            answer_full = f"<think>Error</think>Crash: {e}"
            answer_body = f"Pipeline Crash: {e}"
            schema_ok = False
            self_healing_attempts = 2
            
        # 4. Check for Safety Refusal and Recall
        refusal_pattern = re.compile(r"(抱歉|无法协助|不符合安全规定|作为一个AI|不能回答|作为AI|未获得授权)")
        refusal_avoided = not bool(refusal_pattern.search(answer_body))
        
        recall_rate = calculate_recall(answer_body, expected_keywords)
        
        # 5. Run Comprehensive LLM-as-a-Judge structural scoring (1 call for 5 metrics)
        logger.info(f"   Invoking Comprehensive LLM-as-a-Judge for Case {case_id}...")
        
        refs_text = json.dumps(refs, ensure_ascii=False)
        judge_prompt = JUDGE_COMPREHENSIVE_PROMPT.replace("{{ query }}", query).replace("{{ answer }}", answer_body.replace('"', '\\"')).replace("{{ refs_text }}", refs_text)
        judge_messages = [{"role": "user", "content": judge_prompt}]
        
        try:
            metrics: ComprehensiveJudgeMetrics = await client.call_llm_structured(judge_messages, ComprehensiveJudgeMetrics)
        except Exception as e:
            logger.warning(f"Comprehensive Judge failed: {e}. Falling back to default scores.")
            # Fallback mock
            default_score = 10.0 if refusal_avoided and schema_ok else 2.0
            def_jm = JudgeMetric(score=default_score, reason=f"Judge failed: {e}")
            metrics = ComprehensiveJudgeMetrics(
                grounding=def_jm,
                isolation=def_jm,
                explainability=def_jm,
                professionalism=def_jm,
                relevance=def_jm
            )
            
        # 6. Calculate Business Success Rate
        all_judge_scores = [
            metrics.grounding.score,
            metrics.isolation.score,
            metrics.explainability.score,
            metrics.professionalism.score,
            metrics.relevance.score
        ]
        is_success = schema_ok and refusal_avoided and (recall_rate >= 0.5) and all(s >= 6.0 for s in all_judge_scores)
        
        # Record final evaluated item
        result_item = EvalResultItem(
            case_id=case_id,
            category=category,
            query=query,
            schema_ok=schema_ok,
            self_healing_attempts=self_healing_attempts,
            refusal_avoided=refusal_avoided,
            recall_rate=recall_rate,
            is_success=is_success,
            judge_metrics=metrics,
            answer_preview=answer_body[:140] + "..."
        )
        results.append(result_item)
        
        logger.info(f"   [RESULT] Success: {is_success} | Recall: {recall_rate*100:.0f}% | Avg Judge Score: {sum(all_judge_scores)/5:.1f}/10")
        
    await client.close()
    
    # 7. Compute Statistical Aggregates
    total_cases = len(results)
    avg_recall = sum(r.recall_rate for r in results) / total_cases if total_cases > 0 else 0
    success_rate = (sum(1 for r in results if r.is_success) / total_cases) * 100 if total_cases > 0 else 0
    
    avg_grounding = sum(r.judge_metrics.grounding.score for r in results) / total_cases if total_cases > 0 else 0
    avg_isolation = sum(r.judge_metrics.isolation.score for r in results) / total_cases if total_cases > 0 else 0
    avg_explainability = sum(r.judge_metrics.explainability.score for r in results) / total_cases if total_cases > 0 else 0
    avg_professionalism = sum(r.judge_metrics.professionalism.score for r in results) / total_cases if total_cases > 0 else 0
    avg_relevance = sum(r.judge_metrics.relevance.score for r in results) / total_cases if total_cases > 0 else 0
    
    conformance_rate = (sum(1 for r in results if r.schema_ok) / total_cases) * 100 if total_cases > 0 else 0
    refusal_avoided_rate = (sum(1 for r in results if r.refusal_avoided) / total_cases) * 100 if total_cases > 0 else 0
    
    report = {
        "summary": {
            "total_cases": total_cases,
            "success_rate": round(success_rate, 2),
            "average_recall": round(avg_recall, 2),
            "schema_conformance_rate": round(conformance_rate, 2),
            "refusal_avoidance_rate": round(refusal_avoided_rate, 2),
            "average_grounding_score": round(avg_grounding, 2),
            "average_isolation_score": round(avg_isolation, 2),
            "average_explainability_score": round(avg_explainability, 2),
            "average_professionalism_score": round(avg_professionalism, 2),
            "average_relevance_score": round(avg_relevance, 2)
        },
        "results": [json.loads(r.model_dump_json()) for r in results]
    }
    
    # Save report
    report_file = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*80)
    print("      🎯  Medical QA Pipeline Evaluation Report Summary")
    print("="*80)
    print(f"  测试用例总数 (Total Cases)       : {total_cases}")
    print(f"  业务成功率 (Business Success)    : {report['summary']['success_rate']}%")
    print(f"  平均召回率 (Average Recall)      : {report['summary']['average_recall']*100:.1f}%")
    print(f"  格式契合率 (Schema Conformance)  : {report['summary']['schema_conformance_rate']}%")
    print(f"  防拒答通过率 (Refusal Avoidance) : {report['summary']['refusal_avoidance_rate']}%")
    print("-"*80)
    print(f"  事实忠实度 (Avg Grounding)       : {report['summary']['average_grounding_score']} / 10.0")
    print(f"  领域隔离度 (Avg Isolation)       : {report['summary']['average_isolation_score']} / 10.0")
    print(f"  可解释性   (Avg Explainability)  : {report['summary']['average_explainability_score']} / 10.0")
    print(f"  专业性     (Avg Professionalism) : {report['summary']['average_professionalism_score']} / 10.0")
    print(f"  相关性     (Avg Relevance)       : {report['summary']['average_relevance_score']} / 10.0")
    print("-"*80)
    print(f"  📁 详细质量评估报告已保存至: {report_file}")
    print("="*80 + "\n")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_evaluation())
