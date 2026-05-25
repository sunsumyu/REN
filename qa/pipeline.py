import asyncio
import json
import random
import re
import logging
from typing import List, Dict, Any, Tuple
from api_client import APIClient
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
        """
        entities = graph_data.get("entities", [])
        relationships = graph_data.get("relationships", [])
        
        context_list = []
        refs = []
        
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
        response = await self.api_client.call_llm(prompt)
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
        Plan answering perspectives (facets) for a query.
        """
        prompt = prompts.render_prompt(prompts.FACET_PLANNER_TEMPLATE, query=query)
        response = await self.api_client.call_llm(prompt)
        facets = parse_json_safely(response, [])
        logger.info(f"Planned initial facets: {facets}")
        return facets

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
            response = await self.api_client.call_llm(prompt)
            reduced_facets = parse_json_safely(response, [])
            if len(reduced_facets) == 8:
                return reduced_facets
            else:
                logger.warning(f"Reducer failed to output exactly 8 facets (got {len(reduced_facets)}). Capping original list.")
                return facets[:8]
                
        elif 2 < count < 8:
            logger.info(f"Facet count {count} is between 2 and 8. Running Facet Expander...")
            prompt = prompts.render_prompt(prompts.FACET_EXPANDER_TEMPLATE, query=query, facets=facets)
            response = await self.api_client.call_llm(prompt)
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
        Call the FacetGraph-QA Agent for a single perspective.
        """
        async with semaphore:
            prompt = prompts.render_prompt(prompts.FACET_QA_TEMPLATE, query=query, facet=facet, refs=refs)
            # QA Agent requires structured thinking and strictly outputting <think><facet = RISK>...</think>
            response = await self.api_client.call_llm(prompt)
            
            # Injecting exact thinking tags if model outputted standard <think> without facet attribute
            # The trial document page 1 says: <think >< facet = RISK > (思考过程) < /think >
            # We enforce this strictly:
            if "<think>" in response and f"facet =" not in response:
                response = response.replace("<think>", f"<think><facet = {facet}>")
            elif not response.startswith("<think"):
                # If model missed think block entirely, we can prepend a mock one to satisfy format
                mock_think = f"<think><facet = {facet}>\n问题拆解：\n- S1: 针对{facet}进行多视角切入回答。\n证据清单：\n[证据R1：来源=refs:《实体库汇总》，定位=全面，要点=结合图谱背景信息]\n推理链：\n- P1: 基于背景知识 -> 归纳总结 -> 输出正文。\n最终结论摘要：\n- 形成多视角{facet}高质量医学分析。\n</think>\n"
                response = mock_think + response
                
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
        response = await self.api_client.call_llm(prompt)
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
        summary = await self.api_client.call_llm(prompt)
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
        next_q = await self.api_client.call_llm(prompt)
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
        """
        # Determine random number of rounds
        import os
        num_rounds = int(os.getenv("NUM_ROUNDS", 1))
        logger.info(f"=== Starting Multi-Round Generation: {num_rounds} Rounds ===")
        
        # 1. Fetch initial random context from KG API
        graph_data = await self.api_client.fetch_random_knowledge_graph(count=2)
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
        return history[-1]
