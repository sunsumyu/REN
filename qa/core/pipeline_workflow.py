# -*- coding: utf-8 -*-
import asyncio
import json
import random
import re
import logging
from typing import List, Dict, Any, Tuple
import config
import prompts
from models import FacetPlan, FacetQAOutput
from services.llm_service import ILLMService
from services.graph_service import IGraphService
from strategies.redundancy_filter.llm_filter import IRedundancyFilterStrategy
from core.prompt_renderer import PromptRenderer

logger = logging.getLogger("MedicalQA.PipelineWorkflow")

class PipelineWorkflow:
    """
    Surgically re-engineered pipeline workspace representing the 
    core Multi-round Generation Orchestrator in a clear Workflow architectural style.
    """
    def __init__(
        self, 
        llm_service: ILLMService, 
        graph_service: IGraphService, 
        redundancy_filter: IRedundancyFilterStrategy
    ):
        self.llm_service = llm_service
        self.graph_service = graph_service
        self.redundancy_filter = redundancy_filter

    async def _prepare_context_and_refs(self, graph_data: Dict[str, Any], query: str = "") -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Converts Knowledge Graph response into context and reference blocks, with Phase 4 Guideline Grounding.
        """
        entities = graph_data.get("entities", []) or []
        relationships = graph_data.get("relationships", []) or []
        
        context_list = []
        refs = []
        
        from retrieval.retrieval_manager import RetrievalManager
        retrieval_mgr = RetrievalManager()
        
        # Format entities
        for entity in entities:
            name = entity.get("name", "未命名实体")
            ent_type = entity.get("type", "普通实体")
            description = entity.get("description", "暂无描述").strip()
            ent_id = entity.get("id", "")
            
            context_str = f"医疗实体【{name}】（类型: {ent_type}）：{description}"
            source_str = f"《图谱实体集:{name}》"
            
            item = {
                "context": context_str,
                "source": source_str,
                "entity_id": str(ent_id)
            }
            context_list.append(item)
            refs.append({
                "context": f"概念定义: {name} (类型: {ent_type}) - {description}",
                "source": f"refs:《实体库:{name}》"
            })
            
            try:
                tiered_refs, tier_label = await retrieval_mgr.get_grounding_references(query, name)
                logger.info(f"Tiered Grounding match found for '{name}' via [{tier_label}]: injecting {len(tiered_refs)} reference items.")
                for r_item in tiered_refs:
                    refs.append(r_item)
            except Exception as e:
                logger.error(f"Failed to fetch tiered retrieval grounding for entity '{name}': {e}")
            
        # Format relationships
        for rel in relationships:
            src = rel.get("sourceName", "")
            tgt = rel.get("targetName", "")
            relation = rel.get("relationship", "")
            strength = rel.get("relationshipStrength", 10)
            
            context_str = f"关联关系：【{src}】与【{tgt}】之间存在关联【{relation}】（强度: {strength}）"
            source_str = f"《图谱关系集:{src}-{tgt}》"
            
            item = {
                "context": context_str,
                "source": source_str
            }
            context_list.append(item)
            refs.append({
                "context": f"知识关联: 【{src}】 --({relation})--> 【{tgt}】",
                "source": f"refs:《图谱关系:{src}-{tgt}》"
            })
            
        retrieval_mgr.close()
        return context_list, refs

    async def generate_initial_question(self, context_list: List[Dict[str, str]], task_id_label: str = "") -> str:
        """
        Generate multiple questions from context and pick one randomly (Routed to lightweight).
        """
        prompt = PromptRenderer.render(prompts.QUESTION_CREATOR_TEMPLATE, context_list=context_list)
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        response = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}初始问题生成")
        
        from pipeline import parse_json_safely
        questions = parse_json_safely(response, [])
        if not questions:
            raise Exception("Failed to generate questions from context.")
            
        selected_q = random.choice(questions)
        logger.info(f"Generated {len(questions)} questions. Selected: '{selected_q}'")
        return selected_q

    async def plan_facets(self, query: str, task_id_label: str = "") -> List[str]:
        """
        Plan answering perspectives (facets) using lightweight structured constraints.
        """
        prompt = PromptRenderer.render(prompts.FACET_PLANNER_TEMPLATE, query=query)
        messages = [{"role": "user", "content": prompt}]
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        try:
            result: FacetPlan = await self.llm_service.call_llm_structured(
                messages, 
                FacetPlan, 
                model_pool="lightweight", 
                stage=f"{stage_prefix}视角切面规划"
            )
            facets = [f for f in result.facets]
            logger.info(f"Planned initial facets: {facets}")
            return facets
        except Exception as e:
            logger.error(f"Failed to plan facets: {e}. Falling back to default list.")
            return ["临床表现", "药理学机制"]

    async def preprocess_facets(self, query: str, facets: List[str], task_id_label: str = "") -> List[str]:
        """
        Preprocess facets (Routed to lightweight).
        """
        count = len(facets)
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        
        from pipeline import parse_json_safely
        if count > 8:
            logger.info(f"Facet count {count} > 8. Running Facet Reducer...")
            prompt = PromptRenderer.render(prompts.FACET_REDUCER_TEMPLATE, query=query, facets=facets)
            response = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}切面视角缩减过滤")
            reduced_facets = parse_json_safely(response, [])
            if len(reduced_facets) == 8:
                return reduced_facets
            else:
                return facets[:8]
        elif 2 < count < 8:
            logger.info(f"Facet count {count} is between 2 and 8. Running Facet Expander...")
            prompt = PromptRenderer.render(prompts.FACET_EXPANDER_TEMPLATE, query=query, facets=facets)
            response = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}切面视角丰富扩充")
            expanded_new = parse_json_safely(response, [])
            
            combined = list(facets)
            for f in expanded_new:
                if f not in combined:
                    combined.append(f)
            return combined
        else:
            return facets

    async def answer_single_facet(self, query: str, facet: str, refs: List[Dict[str, str]], semaphore: asyncio.Semaphore, task_id_label: str = "") -> Tuple[str, str]:
        """
        Call the FacetGraph-QA Agent for a single perspective (Core generation: MUST stay in premium).
        """
        from strategies.quality_gate.answer_guard import check_answer_quality
        
        async with semaphore:
            system_prompt = PromptRenderer.get_l1_meta()
            task_prompt = PromptRenderer.get_l2_execution(facet)
            context_prompt = PromptRenderer.get_l3_context(query, refs, [])
            user_prompt = f"{task_prompt}\n\n{context_prompt}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            max_quality_attempts = 2
            stage_prefix = f"[{task_id_label}] " if task_id_label else ""
            
            try:
                result = None
                is_passed = False
                reason = ""
                
                for q_attempt in range(max_quality_attempts + 1):
                    result: FacetQAOutput = await self.llm_service.call_llm_structured(
                        messages, 
                        FacetQAOutput, 
                        model_pool="premium", 
                        stage=f"{stage_prefix}切面深度问答 - {facet}"
                    )
                    
                    is_passed, reason = check_answer_quality(
                        result.answer_body, 
                        getattr(result, "_reasoning_content", "")
                    )
                    if is_passed:
                        break
                    logger.warning(f"Quality Guardrail FAILED on structured QA attempt {q_attempt} for facet '{facet}': {reason}. Retrying...")
                    
                if not is_passed:
                    raise ValueError(f"Quality Guardrail failed repeatedly for structured output: {reason}")
                
                reasoning_content = getattr(result, "_reasoning_content", "")
                if reasoning_content:
                    cleaned_reasoning = reasoning_content.replace("<think>", "").replace("</think>", "").strip()
                    think_block = f"<think><facet = {facet}>\n{cleaned_reasoning}\n</think>\n"
                else:
                    evidences_str = "\n".join([
                        f"[证据R{i+1}：来源={e.source}，定位={e.location}，要点={e.summary}]"
                        for i, e in enumerate(result.evidences)
                    ])
                    reasoning_str = "\n".join([
                        f"- {step.step_id}: {step.logic}"
                        for step in result.reasoning_chains
                    ])
                    sub_questions_str = "\n".join([f"- {q}" for q in result.sub_questions])
                    
                    think_block = (
                        f"<think><facet = {facet}>\n"
                        f"问题拆解：\n{sub_questions_str}\n"
                        f"证据清单：\n{evidences_str}\n"
                        f"推理链：\n{reasoning_str}\n"
                        f"最终结论摘要：\n- {result.final_conclusion_summary}\n"
                        f"</think>\n"
                    )
                
                response = think_block + result.answer_body
                logger.info(f"Successfully generated structured and layered QA output for facet: {facet}")
                
            except Exception as e:
                logger.error(f"Structured QA call failed for facet '{facet}': {e}. Triggering fallback.")
                fallback_prompt = f"{system_prompt}\n\n{user_prompt}"
                raw_response = ""
                reasoning_content = ""
                is_passed = False
                reason = ""
                
                for fb_attempt in range(max_quality_attempts + 1):
                    raw_response, reasoning_content = await self.llm_service.call_llm_with_reasoning(
                        fallback_prompt, 
                        model_pool="premium", 
                        stage=f"{stage_prefix}切面深度问答降级 - {facet}"
                    )
                    
                    check_body = raw_response
                    if "</think>" in raw_response:
                        check_body = raw_response.split("</think>")[-1].strip()
                        if not reasoning_content and "<think>" in raw_response:
                            parts = raw_response.split("</think>")[0].split("<think>")
                            if len(parts) > 1:
                                reasoning_content = parts[1].strip()
                        
                    is_passed, reason = check_answer_quality(check_body, reasoning_content)
                    if is_passed:
                        break
                    logger.warning(f"Quality Guardrail FAILED on fallback attempt {fb_attempt}: {reason}. Retrying...")
                
                if not is_passed:
                    logger.critical(f"Quality guard failed repeatedly for facet '{facet}'. DROPPING this facet.")
                    return facet, None
                
                if "<think>" in raw_response and f"facet =" not in raw_response:
                    response = raw_response.replace("<think>", f"<think><facet = {facet}>")
                elif not raw_response.startswith("<think"):
                    if reasoning_content:
                        cleaned_reasoning = reasoning_content.replace("<think>", "").replace("</think>", "").strip()
                        response = f"<think><facet = {facet}>\n{cleaned_reasoning}\n</think>\n" + raw_response
                    else:
                        mock_think = f"<think><facet = {facet}>\n{facet}切面推理过程\n</think>\n"
                        response = mock_think + raw_response
                else:
                    response = raw_response
                    
            return facet, response

    async def run_parallel_answers(self, query: str, facets: List[str], refs: List[Dict[str, str]], task_id_label: str = "") -> List[Dict[str, str]]:
        """
        Runs the QA Agent in parallel for all facets.
        """
        semaphore = asyncio.Semaphore(config.CONCURRENT_QA_LIMIT)
        tasks = [self.answer_single_facet(query, facet, refs, semaphore, task_id_label=task_id_label) for facet in facets]
        results = await asyncio.gather(*tasks)
        
        planners = []
        for facet, answer in results:
            if answer is None:
                continue
            planners.append({
                "planner": facet,
                "answer": answer
            })
        return planners

    async def synthesize_answers(self, query: str, planners: List[Dict[str, str]], task_id_label: str = "") -> str:
        """
        Synthesize filtered answers into a single cohesive final summary answer (Routed to lightweight).
        """
        answers_clean = []
        for p in planners:
            ans = p["answer"]
            if "</think>" in ans:
                parts = ans.split("</think>")
                ans_body = parts[-1].strip()
            else:
                ans_body = ans.strip()
            answers_clean.append(ans_body)
            
        prompt = PromptRenderer.render(prompts.MULTI_ANSWER_SYNTHESIS_TEMPLATE, query=query, answers=answers_clean)
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        summary = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}切面问答综合凝练")
        logger.info("Successfully synthesized final answer summary.")
        return summary

    async def generate_next_question(self, context_list: List[Dict[str, str]], history: List[Dict[str, Any]], previous_summary: str, task_id_label: str = "") -> str:
        """
        Generate the next question in a multi-round dialog (Routed to lightweight).
        """
        prompt = PromptRenderer.render(
            prompts.NEXT_QUESTION_TEMPLATE,
            context_list=context_list,
            history=history,
            summary=previous_summary
        )
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        next_q = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}多轮对话下一问生成")
        next_q = next_q.strip().strip('"').strip("'")
        return next_q

    async def generate_single_round(
        self, 
        query: str, 
        refs: List[Dict[str, str]], 
        history: List[Dict[str, Any]] = None,
        task_id_label: str = ""
    ) -> Dict[str, Any]:
        """
        Execute a complete single round QA generation.
        """
        logger.info(f"--- Starting Round QA for Q: '{query}' ---")
        
        initial_facets = await self.plan_facets(query, task_id_label=task_id_label)
        if not initial_facets:
            initial_facets = ["临床表现", "药理学机制"]
            
        final_facets = await self.preprocess_facets(query, initial_facets, task_id_label=task_id_label)
        planners = await self.run_parallel_answers(query, final_facets, refs, task_id_label=task_id_label)
        
        # Strategies execution under strategy pattern
        non_redundant_planners = await self.redundancy_filter.filter_redundancy(
            query, 
            planners, 
            task_id_label=task_id_label
        )
        if not non_redundant_planners:
            non_redundant_planners = planners
            
        summary = await self.synthesize_answers(query, non_redundant_planners, task_id_label=task_id_label)
        
        round_data = {
            "Q": query,
            "planners": non_redundant_planners,
            "history": list(history) if history else [],
            "summary": summary
        }
        return round_data

    async def generate_multi_round_dataset(self, intent: Dict[str, Any] = None, task_id_label: str = "") -> Dict[str, Any]:
        """
        Orchestrates a robust Dialogue Generation Pipeline based on Graph-RAG Intention Guiding.
        """
        log_prefix = f"[{task_id_label}] " if task_id_label else ""
        num_rounds = int(config.NUM_ROUNDS) if hasattr(config, "NUM_ROUNDS") else 1
        logger.info(f"{log_prefix}=== Starting Multi-Round Generation: {num_rounds} Rounds ===")
        
        try:
            graph_data = await self.graph_service.fetch_random_knowledge_graph(count=1)
        except Exception as e:
            logger.critical(f"{log_prefix}Failed to fetch random KG: {e}")
            graph_data = {"entities": [], "relationships": []}
            
        if "entities" not in graph_data or not graph_data["entities"]:
            graph_data["entities"] = []
        if "relationships" not in graph_data or not graph_data["relationships"]:
            graph_data["relationships"] = []
            
        if intent is not None:
            selected_theme = intent.get("theme", "临床医学研究")
            for seed in intent.get("seeds", []):
                if not any(e.get("name") == seed["name"] for e in graph_data["entities"]):
                    graph_data["entities"].append({
                        "id": random.randint(10000, 99999),
                        "name": seed["name"],
                        "type": seed["type"],
                        "description": seed["description"]
                    })
            for rel in intent.get("rels", []):
                if not any(r.get("sourceName") == rel["sourceName"] and r.get("targetName") == rel["targetName"] for r in graph_data["relationships"]):
                    graph_data["relationships"].append(rel)
        else:
            if graph_data.get("relationships"):
                rel = random.choice(graph_data["relationships"])
                src = rel.get("sourceName", "")
                tgt = rel.get("targetName", "")
                relation = rel.get("relationship", "")
                selected_theme = f"{src}与{tgt}的{relation}临床诊疗规范与医学循证"
            elif graph_data.get("entities"):
                names = [e.get("name") for e in graph_data["entities"][:2]]
                selected_theme = "与".join(names) + "的临床应用研究与用药指南"
            else:
                selected_theme = "常见疾病的循证医学用药指南"
                
        logger.info(f"{log_prefix}--- Intention-Guided Graph-RAG Theme: '{selected_theme}' ---")
        
        context_list, refs = await self._prepare_context_and_refs(graph_data, query=selected_theme)
        q1 = await self.generate_initial_question(context_list, task_id_label=task_id_label)
        
        history = []
        current_q = q1
        
        for r in range(1, num_rounds + 1):
            logger.info(f"{log_prefix}=== Running Round {r} / {num_rounds} ===")
            round_result = await self.generate_single_round(current_q, refs, history, task_id_label=task_id_label)
            history.append(round_result)
            
            if r < num_rounds:
                if random.random() < 0.3:
                    try:
                        logger.info(f"{log_prefix}Expanding dialog horizon...")
                        additional_graph = await self.graph_service.fetch_random_knowledge_graph(count=1)
                        new_context, new_refs = await self._prepare_context_and_refs(additional_graph, query=current_q)
                        context_list.extend(new_context)
                        refs.extend(new_refs)
                    except Exception as e:
                        logger.warning(f"Failed to fetch extra entities: {e}")
                        
                current_q = await self.generate_next_question(context_list, history, round_result["summary"], task_id_label=task_id_label)
                
        logger.info(f"{log_prefix}=== Multi-Round Dialog Generation Completed! ===")
        history[-1]["refs"] = refs
        return history[-1]
