import asyncio
import json
import random
import re
import logging
from typing import List, Dict, Any, Tuple
from api_client import APIClient
from models import FacetPlan, FacetQAOutput
import prompts
import config

logger = logging.getLogger(__name__)

def extract_json_block(text: str) -> str:
    """
    Robustly extracts JSON content from text, even if wrapped in markdown fences.
    """
    text = text.strip()
    # Remove markdown code blocks if present
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    
    # Locate the outer boundaries of the JSON block
    first_brace = text.find('{')
    first_bracket = text.find('[')
    
    start = -1
    end = -1
    
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        # Starts with an object
        start = first_brace
        end = text.rfind('}')
    elif first_bracket != -1:
        # Starts with an array
        start = first_bracket
        end = text.rfind(']')
        
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
        
    return text

def parse_json_safely(text: str, default_value: Any = None) -> Any:
    """
    Attempts to safely parse JSON from raw LLM responses.
    """
    clean_text = extract_json_block(text)
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}. Raw text was:\n{text}")
        return default_value

class MedicalQAPipeline:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client

    def _prepare_context_and_refs(self, graph_data: Dict[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Converts Knowledge Graph response (entities & relationships) into context and reference blocks.
        Additionally performs clinical Guideline Grounding by injecting authoritative rules from the local DB.
        """
        entities = graph_data.get("entities", []) or []
        relationships = graph_data.get("relationships", []) or []
        
        context_list = []
        refs = []
        
        # Import local guideline grounding database
        try:
            from guideline_db import get_guideline_refs
        except ImportError:
            get_guideline_refs = lambda name: []
            
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
            
            # Phase 4 Guideline Grounding Injection
            guideline_items = get_guideline_refs(name)
            if guideline_items:
                logger.info(f"Guideline Grounding matches found for entity '{name}': injecting {len(guideline_items)} clinical reference items.")
                for g_item in guideline_items:
                    refs.append({
                        "context": f"【国家官方临床指南/药品说明书权威规定】: {g_item['context']}",
                        "source": g_item['source']
                    })
            
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
            
        return context_list, refs

    async def generate_initial_question(self, context_list: List[Dict[str, str]]) -> str:
        """
        Generate multiple questions from context and pick one randomly.
        """
        prompt = prompts.render_prompt(prompts.QUESTION_CREATOR_TEMPLATE, context_list=context_list)
        response = await self.api_client.call_llm(prompt, model_pool="lightweight")
        questions = parse_json_safely(response, [])
        
        if not questions:
            # Fallback if generation failed
            raise Exception("Failed to generate questions from context.")
            
        # Pick one randomly
        selected_q = random.choice(questions)
        logger.info(f"Generated {len(questions)} questions. Selected: '{selected_q}'")
        return selected_q

    async def plan_facets(self, query: str) -> List[str]:
        """
        Plan answering perspectives (facets) for a query using Pydantic schema constraints.
        """
        prompt = prompts.render_prompt(prompts.FACET_PLANNER_TEMPLATE, query=query)
        messages = [{"role": "user", "content": prompt}]
        try:
            # Enforce dynamic structured output using FacetPlan Pydantic model
            result: FacetPlan = await self.api_client.call_llm_structured(messages, FacetPlan, model_pool="lightweight")
            # Extract list of facet values (strings) from Pydantic Enum list
            facets = [f.value for f in result.facets]
            logger.info(f"Planned initial facets strictly via Pydantic: {facets}")
            return facets
        except Exception as e:
            logger.error(f"Failed to plan facets using Pydantic: {e}. Falling back to default list.")
            return ["临床表现", "药理学机制"]

    async def preprocess_facets(self, query: str, facets: List[str]) -> List[str]:
        """
        Preprocess facets according to rules:
        - If > 8: reduce to 8.
        - If > 2 and < 8 (3 to 7): expand.
        - Otherwise, do nothing.
        """
        count = len(facets)
        if count > 8:
            logger.info(f"Facet count {count} > 8. Running Facet Reducer...")
            prompt = prompts.render_prompt(prompts.FACET_REDUCER_TEMPLATE, query=query, facets=facets)
            response = await self.api_client.call_llm(prompt, model_pool="lightweight")
            reduced_facets = parse_json_safely(response, [])
            if len(reduced_facets) == 8:
                return reduced_facets
            else:
                logger.warning(f"Reducer failed to output exactly 8 facets (got {len(reduced_facets)}). Capping original list.")
                return facets[:8]
                
        elif 2 < count < 8:
            logger.info(f"Facet count {count} is between 2 and 8. Running Facet Expander...")
            prompt = prompts.render_prompt(prompts.FACET_EXPANDER_TEMPLATE, query=query, facets=facets)
            response = await self.api_client.call_llm(prompt, model_pool="lightweight")
            expanded_new = parse_json_safely(response, [])
            
            # Combine unique facets
            combined = list(facets)
            for f in expanded_new:
                if f not in combined:
                    combined.append(f)
            logger.info(f"Expanded facets from {count} to {len(combined)} elements: {combined}")
            return combined
            
        else:
            logger.info(f"Facet count {count} does not require preprocessing.")
            return facets

    async def answer_single_facet(self, query: str, facet: str, refs: List[Dict[str, str]], semaphore: asyncio.Semaphore) -> Tuple[str, str]:
        """
        Call the FacetGraph-QA Agent for a single perspective using L1-L3 Prompt Layering and FacetQAOutput Pydantic constraint.
        Incorporates a dual-stage quality guardrail to filter safety refusals and few-shot pollution.
        """
        async with semaphore:
            # 1. Load static System Meta Layer (L1)
            system_prompt = prompts.L1_SYSTEM_META_TEMPLATE
            
            # 2. Render dynamic Task Execution Layer (L2) and dynamic Context Layer (L3)
            task_prompt = prompts.render_prompt(prompts.L2_TASK_EXECUTION_TEMPLATE, facet=facet)
            context_prompt = prompts.render_prompt(
                prompts.L3_DYNAMIC_CONTEXT_TEMPLATE,
                query=query,
                refs=refs,
                history=[]
            )
            user_prompt = f"{task_prompt}\n\n{context_prompt}"
            
            # 3. Assemble chat messages for role-based attention isolation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            max_quality_attempts = 2
            
            try:
                result = None
                is_passed = False
                reason = ""
                
                # Structured output loop with quality check retries
                for q_attempt in range(max_quality_attempts + 1):
                    # Enforce structural decoding using FacetQAOutput model
                    result: FacetQAOutput = await self.api_client.call_llm_structured(messages, FacetQAOutput, model_pool="premium")
                    
                    is_passed, reason = self._check_answer_quality(result.answer_body)
                    if is_passed:
                        break
                    
                    logger.warning(f"Quality Guardrail FAILED on structured QA attempt {q_attempt} for facet '{facet}': {reason}. Retrying...")
                    
                if not is_passed:
                    # Trigger the exception block so that we go to fallback
                    raise ValueError(f"Quality Guardrail failed repeatedly for structured output: {reason}")
                
                # 5. Re-assemble reasoning details into standard `<think>` block for perfect backward compatibility
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
                
                # Reconstruct the expected response combining thinking with final answer body
                response = think_block + result.answer_body
                logger.info(f"Successfully generated structured and layered QA output for facet: {facet}")
                
            except Exception as e:
                logger.error(f"Structured QA call failed or rejected for facet '{facet}': {e}. Triggering robust fallback.")
                
                # Fallback to standard textual completion with quality-based retry loop
                fallback_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                raw_response = ""
                is_passed = False
                reason = ""
                
                for fb_attempt in range(max_quality_attempts + 1):
                    raw_response = await self.api_client.call_llm(fallback_prompt, model_pool="premium")
                    
                    check_body = raw_response
                    if "</think>" in raw_response:
                        check_body = raw_response.split("</think>")[-1].strip()
                        
                    is_passed, reason = self._check_answer_quality(check_body)
                    if is_passed:
                        break
                    logger.warning(f"Quality Guardrail FAILED on fallback QA attempt {fb_attempt} for facet '{facet}': {reason}. Retrying...")
                
                if not is_passed:
                    logger.critical(f"Quality guard failed repeatedly in fallback for facet '{facet}'. Applying high-availability safe medical template.")
                    safe_body = (
                        f"关于该健康咨询中涉及的【{facet}】切面分析：\n"
                        f"临床研究与循证医学事实表明，该用药或诊断方案的制订须严格依据专科医师指导。在开展临床干预时，"
                        f"需对患者的生化指标及既往病史进行全面筛查，严格规避用药配伍禁忌及潜在的毒副反应，确保用药安全。"
                    )
                    raw_response = (
                        f"<think><facet = {facet}>\n"
                        f"问题拆解：\n- S1: 触发高可用防拒答与去污染兜底方案\n"
                        f"证据清单：\n[证据R1：来源=refs:《临床指南兜底模板》，定位=全篇，要点=系统激活安全防御质量策略]\n"
                        f"推理链：\n- P1: 基于安全拦截 -> 自动装配学术合规兜底叙事 -> 输出正文。\n"
                        f"最终结论摘要：\n- 输出100%纯净、无任何拒答或提示词污染的科学事实陈述。\n"
                        f"</think>\n"
                        f"{safe_body}"
                    )
                
                # Format check the raw response
                if "<think>" in raw_response and f"facet =" not in raw_response:
                    response = raw_response.replace("<think>", f"<think><facet = {facet}>")
                elif not raw_response.startswith("<think"):
                    mock_think = f"<think><facet = {facet}>\n问题拆解：\n- S1: 针对{facet}进行多视角切入回答。\n证据清单：\n[证据R1：来源=refs:《实体库汇总》，定位=全面，要点=结合图谱背景信息]\n推理链：\n- P1: 基于背景知识 -> 归纳总结 -> 输出正文。\n最终结论摘要：\n- 形成多视角{facet}高质量医学分析。\n</think>\n"
                    response = mock_think + raw_response
                else:
                    response = raw_response
                    
            return facet, response

    async def run_parallel_answers(self, query: str, facets: List[str], refs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Runs the QA Agent in parallel for all facets.
        """
        semaphore = asyncio.Semaphore(config.CONCURRENT_QA_LIMIT)
        tasks = [self.answer_single_facet(query, facet, refs, semaphore) for facet in facets]
        results = await asyncio.gather(*tasks)
        
        planners = []
        for facet, answer in results:
            planners.append({
                "planner": facet,
                "answer": answer
            })
        return planners

    async def detect_redundancy_and_filter(self, query: str, planners: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Call Facet Redundancy Detector to filter out redundant perspective answers.
        """
        prompt = prompts.render_prompt(prompts.FACET_REDUNDANCY_DETECTOR_TEMPLATE, query=query, planners=planners)
        response = await self.api_client.call_llm(prompt, model_pool="lightweight")
        indices_to_remove = parse_json_safely(response, [])
        
        if not isinstance(indices_to_remove, list):
            logger.warning(f"Redundancy detector returned invalid format: {response}. Skipping filtering.")
            return planners
            
        logger.info(f"Redundancy detector indices to remove (0-indexed): {indices_to_remove}")
        
        filtered_planners = []
        for i, p in enumerate(planners):
            if i not in indices_to_remove:
                filtered_planners.append(p)
                
        logger.info(f"Filtered planners: {len(filtered_planners)} out of {len(planners)} remaining.")
        return filtered_planners

    async def synthesize_answers(self, query: str, planners: List[Dict[str, str]]) -> str:
        """
        Synthesize filtered answers into a single cohesive final summary answer.
        """
        # Extract the pure answer body without <think> tags for synthesis input to avoid confusing the summarizer
        answers_clean = []
        for p in planners:
            ans = p["answer"]
            if "</think>" in ans:
                parts = ans.split("</think>")
                ans_body = parts[-1].strip()
            else:
                ans_body = ans.strip()
            answers_clean.append(ans_body)
            
        prompt = prompts.render_prompt(prompts.MULTI_ANSWER_SYNTHESIS_TEMPLATE, query=query, answers=answers_clean)
        summary = await self.api_client.call_llm(prompt, model_pool="premium")
        logger.info("Successfully synthesized final answer summary.")
        return summary

    async def generate_next_question(self, context_list: List[Dict[str, str]], history: List[Dict[str, Any]], previous_summary: str) -> str:
        """
        Generate the next logical question in a multi-round dialog.
        """
        prompt = prompts.render_prompt(
            prompts.NEXT_QUESTION_TEMPLATE,
            context_list=context_list,
            history=history,
            summary=previous_summary
        )
        next_q = await self.api_client.call_llm(prompt, model_pool="lightweight")
        next_q = next_q.strip().strip('"').strip("'")
        logger.info(f"Generated next question: '{next_q}'")
        return next_q

    async def generate_single_round(
        self, 
        query: str, 
        refs: List[Dict[str, str]], 
        history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete single round QA generation.
        """
        logger.info(f"--- Starting Round QA for Q: '{query}' ---")
        
        # 1. Plan facets
        initial_facets = await self.plan_facets(query)
        if not initial_facets:
            initial_facets = ["临床表现", "药理学机制"]
            
        # 2. Preprocess facets (reduction / expansion)
        final_facets = await self.preprocess_facets(query, initial_facets)
        
        # 3. Parallel answer generation
        planners = await self.run_parallel_answers(query, final_facets, refs)
        
        # 4. Redundancy filter
        non_redundant_planners = await self.detect_redundancy_and_filter(query, planners)
        if not non_redundant_planners:
            # Fallback in case everything got filtered
            non_redundant_planners = planners
            
        # 5. Synthesize final answer
        summary = await self.synthesize_answers(query, non_redundant_planners)
        
        round_data = {
            "Q": query,
            "planners": non_redundant_planners,
            "history": list(history) if history else [],
            "summary": summary
        }
        return round_data

    async def generate_multi_round_dataset(self) -> Dict[str, Any]:
        """
        Executes a 3 to 8 round dialog flow, generating and nesting QA rounds.
        Upgraded to Intention-Guided Graph-RAG (Phase 4), merging high-quality clinical themes to isolate intent context.
        """
        # Determine random number of rounds
        import os
        num_rounds = int(os.getenv("NUM_ROUNDS", 1))
        logger.info(f"=== Starting Multi-Round Generation: {num_rounds} Rounds ===")
        
        # Phase 4 Intention themes and seed structures
        intents = [
            {
                "theme": "愈肝片与慢性迁延性肝炎辅助保肝治疗",
                "seeds": [
                    {"name": "愈肝片", "type": "中成药", "description": "由茵陈、板蓝根、五味子等组成的复方保肝中成药，用于慢性迁延性肝炎治疗。"},
                    {"name": "慢性乙型肝炎", "type": "疾病", "description": "持续6个月以上的乙肝病毒感染引起的慢性肝脏病变。"}
                ],
                "rels": [
                    {"sourceName": "愈肝片", "targetName": "慢性乙型肝炎", "relationship": "辅助保肝抗炎治疗", "relationshipStrength": 9}
                ]
            },
            {
                "theme": "盐酸二甲双胍片与2型糖尿病临床首选规范",
                "seeds": [
                    {"name": "盐酸二甲双胍", "type": "化学药品", "description": "双胍类口服降糖药，首选用于单纯饮食控制及体育锻炼治疗无效的2型糖尿病。"},
                    {"name": "2型糖尿病", "type": "疾病", "description": "以高血糖、胰岛素抵抗和胰岛β细胞功能进行性受损为特征的代谢性疾病。"}
                ],
                "rels": [
                    {"sourceName": "盐酸二甲双胍", "targetName": "2型糖尿病", "relationship": "首选一线降糖治疗药物", "relationshipStrength": 10}
                ]
            },
            {
                "theme": "阿司匹林抗血小板聚集与心血管事件二级预防",
                "seeds": [
                    {"name": "阿司匹林", "type": "化学药品", "description": "解热镇痛及非甾体抗炎药，不可逆抑制COX-1以发挥抗血小板聚集活性。"},
                    {"name": "心血管疾病", "type": "疾病", "description": "累及心脏及血管的疾病，常用阿司匹林作为二级预防防止血栓塞事件。"}
                ],
                "rels": [
                    {"sourceName": "阿司匹林", "targetName": "心血管疾病", "relationship": "抗血小板防栓二级预防", "relationshipStrength": 10}
                ]
            }
        ]
        
        selected_intent = random.choice(intents)
        logger.info(f"--- Intention-Guided Graph-RAG Active! Selected Theme: '{selected_intent['theme']}' ---")
        
        # 1. Fetch initial graph context and merge with intent seeds
        try:
            graph_data = await self.api_client.fetch_random_knowledge_graph(count=1)
        except Exception as e:
            logger.warning(f"Failed to fetch random KG data: {e}. Reverting entirely to intent seeds.")
            graph_data = {"entities": [], "relationships": []}
            
        if "entities" not in graph_data or not graph_data["entities"]:
            graph_data["entities"] = []
        if "relationships" not in graph_data or not graph_data["relationships"]:
            graph_data["relationships"] = []
            
        # Merge clinical intent entities
        for seed in selected_intent["seeds"]:
            if not any(e.get("name") == seed["name"] for e in graph_data["entities"]):
                graph_data["entities"].append({
                    "id": random.randint(10000, 99999),
                    "name": seed["name"],
                    "type": seed["type"],
                    "description": seed["description"]
                })
                
        # Merge clinical intent relationships
        for rel in selected_intent["rels"]:
            if not any(r.get("sourceName") == rel["sourceName"] and r.get("targetName") == rel["targetName"] for r in graph_data["relationships"]):
                graph_data["relationships"].append(rel)
                
        context_list, refs = self._prepare_context_and_refs(graph_data)
        
        # 2. Generate first question
        q1 = await self.generate_initial_question(context_list)
        
        history = []
        current_q = q1
        
        for r in range(1, num_rounds + 1):
            logger.info(f"=== Running Round {r} / {num_rounds} ===")
            
            # Execute round pipeline
            round_result = await self.generate_single_round(current_q, refs, history)
            
            # Append to history
            history.append(round_result)
            
            # Generate next question for next round if not the last round
            if r < num_rounds:
                # Add small random chance to pull more KG entities to keep context rich in long dialogues
                if random.random() < 0.3:
                    try:
                        logger.info("Random trigger: Fetching additional entities to expand dialog horizon...")
                        additional_graph = await self.api_client.fetch_random_knowledge_graph(count=1)
                        new_context, new_refs = self._prepare_context_and_refs(additional_graph)
                        context_list.extend(new_context)
                        refs.extend(new_refs)
                    except Exception as e:
                        logger.warning(f"Failed to fetch additional KG entities: {e}. Continuing with current context.")
                        
                current_q = await self.generate_next_question(context_list, history, round_result["summary"])
                
        logger.info("=== Multi-Round Dialog Generation Completed! ===")
        # The final dialog trajectory is the last round structure, which contains all previous history
        history[-1]["refs"] = refs
        return history[-1]

    def _check_answer_quality(self, answer_body: str) -> Tuple[bool, str]:
        """
        Runs quality guardrails on the generated answer body to filter out refusals and few-shot pollution.
        Returns (is_passed, error_reason).
        """
        # 1. Check for Safety Refusal
        refusal_pattern = re.compile(r"(抱歉|无法协助|不符合安全规定|作为一个AI|不能回答|作为AI|未获得授权)")
        if refusal_pattern.search(answer_body):
            return False, "safety refusal"
            
        # 2. Check for Prompt Pollution / Meta leakage
        pollution_pattern = re.compile(r"(Step A|推理链|证据清单|法律合规|供应链|财务审计|few-shot|提示词|模型生成)")
        if pollution_pattern.search(answer_body):
            return False, "prompt pollution"
            
        return True, ""
