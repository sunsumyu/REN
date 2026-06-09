# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger("MedicalQA.VisualPrinter")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.box import ROUNDED
    _has_rich = True
    _console = Console()
except ImportError:
    _has_rich = False

def print_token_usage(stage: str, model: str, duration: float, usage: dict):
    """
    Renders an elegant, high-observability token usage table in the terminal using Rich.
    Falls back to standard logger if Rich is not available.
    """
    stage = stage or "LLM Call"
    prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
    completion_tokens = usage.get("completion_tokens", 0) if usage else 0
    total_tokens = usage.get("total_tokens", 0) if usage else 0

    if _has_rich:
        table = Table(box=ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Stage / 环节", style="bold cyan", min_width=20)
        table.add_column("Model / 模型", style="green")
        table.add_column("Latency / 耗时", style="yellow", justify="right")
        table.add_column("Prompt / 输入", style="blue", justify="right")
        table.add_column("Completion / 输出", style="magenta", justify="right")
        table.add_column("Total / 总消耗", style="bold red", justify="right")
        
        table.add_row(
            stage,
            model,
            f"{duration:.3f}s",
            str(prompt_tokens),
            str(completion_tokens),
            str(total_tokens)
        )
        _console.print(table)
    else:
        logger.info(
            f"[{stage}] Model: {model} | Latency: {duration:.3f}s | "
            f"Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {total_tokens}"
        )

def print_context_and_refs(context_list: list, refs: list):
    """
    在终端中以富文本（Table/Panel）形式优雅地显示图谱上下文和 RAG 参考文献
    """
    import json
    if _has_rich:
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.text import Text
        
        # 1. 构造 Context List 的表格
        ctx_table = Table(box=ROUNDED, show_header=True, header_style="bold cyan", title="[bold cyan]📍 Graph-RAG Context List (图谱上下文)[/bold cyan]")
        ctx_table.add_column("No.", style="dim", width=4, justify="center")
        ctx_table.add_column("Source / 来源", style="green", width=35)
        ctx_table.add_column("Context / 事实上下文", style="white")
        
        for idx, item in enumerate(context_list):
            ctx_table.add_row(
                str(idx + 1),
                item.get("source", "Unknown"),
                item.get("context", "")
            )
            
        # 2. 构造 Refs 的表格
        refs_table = Table(box=ROUNDED, show_header=True, header_style="bold yellow", title="[bold yellow]📚 References & Grounding (依据文献)[/bold yellow]")
        refs_table.add_column("No.", style="dim", width=4, justify="center")
        refs_table.add_column("Source / 参考文献来源", style="gold1", width=40)
        refs_table.add_column("Context / 文献摘要及定义", style="white")
        
        for idx, item in enumerate(refs):
            source = item.get("source", "Unknown")
            source_text = Text(source)
            if "PubMed" in source:
                source_text.stylize("bold red")
            elif "实体库" in source:
                source_text.stylize("bold green")
            elif "图谱关系" in source:
                source_text.stylize("bold blue")
            elif "异常" in source or "未收录" in source:
                source_text.stylize("bold reverse red")
                
            refs_table.add_row(
                str(idx + 1),
                source_text,
                item.get("context", "")
            )
            
        _console.print("\n")
        _console.print(ctx_table)
        _console.print("\n")
        _console.print(refs_table)
        _console.print("\n")
    else:
        print("\n=== [Fallback] Graph-RAG Context List ===")
        print(json.dumps(context_list, indent=2, ensure_ascii=False))
        print("\n=== [Fallback] References & Grounding ===")
        print(json.dumps(refs, indent=2, ensure_ascii=False))

def print_tiered_refs(query: str, name: str, tier_label: str, tiered_refs: list):
    """
    在终端中以富文本形式优雅地显示分层检索（Tiered Grounding）抓取出来的参考文献
    """
    import json
    if not tiered_refs:
        return
        
    if _has_rich:
        from rich.text import Text
        
        # 构造表格
        table = Table(
            box=ROUNDED, 
            show_header=True, 
            header_style="bold magenta", 
            title=f"[bold magenta]🔍 Tiered Grounding Match: '{name}' via [[bold green]{tier_label}[/bold green]][/bold magenta]"
        )
        table.add_column("No.", style="dim", width=4, justify="center")
        table.add_column("Source / 参考文献来源", style="orange1", width=40)
        table.add_column("Context / 文献参考正文", style="white")
        
        for idx, item in enumerate(tiered_refs):
            source = item.get("source", "Unknown")
            source_text = Text(source)
            if "PubMed" in source:
                source_text.stylize("bold red")
            elif "在线" in source or "异常" in source or "未收录" in source:
                source_text.stylize("bold reverse red")
            elif "实体库" in source:
                source_text.stylize("bold green")
                
            table.add_row(
                str(idx + 1),
                source_text,
                item.get("context", "")
            )
            
        _console.print("\n")
        _console.print(table)
        _console.print("\n")
    else:
        print(f"\n=== [Fallback] Tiered Grounding Match for '{name}' via [{tier_label}] ===")
        print(json.dumps(tiered_refs, indent=2, ensure_ascii=False))

def print_graph_data(graph_data: dict):
    """
    在终端中以富文本表格形式优雅地显示从 API 抓取来的原始知识图谱数据 (Entities & Relationships)
    并且同步写入日志文件。
    """
    import json
    import os
    from datetime import datetime

    if not graph_data:
        return
        
    # 写入日志文件
    try:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, "fetched_graph_entities.txt")
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        entities = graph_data.get("entities", [])
        
        with open(log_path, "a", encoding="utf-8") as f:
            for ent in entities:
                entry = {
                    "timestamp": timestamp,
                    "entity_id": ent.get("id"),
                    "name": ent.get("name"),
                    "type": ent.get("type"),
                    "description": ent.get("description")
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to log graph entities to file: {e}")

        
    entities = graph_data.get("entities", [])
    relationships = graph_data.get("relationships", [])
    
    if _has_rich:
        # 1. 构造 Entities 表格
        ent_table = Table(
            box=ROUNDED, 
            show_header=True, 
            header_style="bold magenta", 
            title="[bold magenta]🟢 Graph Database Entities (图谱节点)[/bold magenta]"
        )
        ent_table.add_column("No.", style="dim", width=4, justify="center")
        ent_table.add_column("ID", style="dim cyan", width=12)
        ent_table.add_column("Entity / 实体名", style="bold green", width=25)
        ent_table.add_column("Type / 类型", style="yellow", width=15)
        ent_table.add_column("Description / 描述说明", style="white")
        
        for idx, ent in enumerate(entities):
            ent_table.add_row(
                str(idx + 1),
                str(ent.get("id", "N/A")),
                ent.get("name", "Unknown"),
                ent.get("type", "Unknown"),
                ent.get("description", "N/A")
            )
            
        # 2. 构造 Relationships 表格
        rel_table = Table(
            box=ROUNDED, 
            show_header=True, 
            header_style="bold cyan", 
            title="[bold cyan]🔵 Graph Database Relationships (图谱边关系)[/bold cyan]"
        )
        rel_table.add_column("No.", style="dim", width=4, justify="center")
        rel_table.add_column("Source / 源实体", style="bold green", width=25)
        rel_table.add_column("Relationship / 关联关系", style="yellow", width=20)
        rel_table.add_column("Target / 目标实体", style="bold green", width=25)
        rel_table.add_column("Strength / 强度", style="red", justify="right", width=10)
        
        for idx, rel in enumerate(relationships):
            rel_table.add_row(
                str(idx + 1),
                rel.get("sourceName", rel.get("source", "Unknown")),
                rel.get("relationship", "Unknown"),
                rel.get("targetName", rel.get("target", "Unknown")),
                str(rel.get("relationshipStrength", 10))
            )
            
        _console.print("\n")
        _console.print(ent_table)
        _console.print("\n")
        _console.print(rel_table)
        _console.print("\n")
    else:
        print("\n=== [Fallback] Graph Data (Entities) ===")
        print(json.dumps(entities, indent=2, ensure_ascii=False))
        print("\n=== [Fallback] Graph Data (Relationships) ===")
        print(json.dumps(relationships, indent=2, ensure_ascii=False))

def print_generated_questions(questions: list):
    """
    在终端中以富文本表格形式优雅地显示大模型生成的候选主问题列表
    """
    if not questions:
        return
        
    if _has_rich:
        # 构造表格
        table = Table(
            box=ROUNDED, 
            show_header=True, 
            header_style="bold cyan", 
            title="[bold cyan]❓ Candidate Questions (生成的主问题候选列表)[/bold cyan]"
        )
        table.add_column("No.", style="dim", width=4, justify="center")
        table.add_column("Generated Question / 候选问题文本", style="white")
        
        for idx, q in enumerate(questions):
            table.add_row(
                str(idx + 1),
                q
            )
        _console.print("\n")
        _console.print(table)
        _console.print("\n")
    else:
        print("\n=== [Fallback] Candidate Questions ===")
        for idx, q in enumerate(questions):
            print(f"  [{idx + 1}] {q}")
        print("\n")

def apply_aop_aspects():
    """
    通过 Monkey Patch 动态织入（Weave）AOP 切面，拦截指定方法的执行，
    在方法返回后（After Returning）自动输出富文本排版。
    """
    try:
        from core.pipeline_workflow import PipelineWorkflow
        from retrieval.retrieval_manager import RetrievalManager
        from services.graph_service import GraphService
        import config
        
        # 1. 织入对 _prepare_context_and_refs 的切面拦截
        orig_prepare = PipelineWorkflow._prepare_context_and_refs
        
        async def wrapped_prepare(self, graph_data, query=""):
            res = await orig_prepare(self, graph_data, query)
            try:
                context_list, refs = res
                print_context_and_refs(context_list, refs)
            except Exception as e:
                logger.error(f"[AOP Aspect Error] Failed to print context_list/refs in wrapped_prepare: {e}")
            return res
            
        PipelineWorkflow._prepare_context_and_refs = wrapped_prepare
        logger.info("AOP Aspect successfully woven: PipelineWorkflow._prepare_context_and_refs")
        
        # 2. 织入对 get_grounding_references 的切面拦截
        orig_grounding = RetrievalManager.get_grounding_references
        
        async def wrapped_grounding(self, query, name):
            res = await orig_grounding(self, query, name)
            try:
                tiered_refs, tier_label = res
                print_tiered_refs(query, name, tier_label, tiered_refs)
            except Exception as e:
                logger.error(f"[AOP Aspect Error] Failed to print tiered_refs in wrapped_grounding: {e}")
            return res
            
        RetrievalManager.get_grounding_references = wrapped_grounding
        logger.info("AOP Aspect successfully woven: RetrievalManager.get_grounding_references")
        
        # 3. 织入对 fetch_random_knowledge_graph 的切面拦截
        orig_fetch = GraphService.fetch_random_knowledge_graph
        
        async def wrapped_fetch(self, count=config.DEFAULT_ENTITY_COUNT, kb_id=config.DEFAULT_KNOWLEDGE_BASE_ID, hop_count=config.DEFAULT_HOP_COUNT):
            res = await orig_fetch(self, count, kb_id, hop_count)
            try:
                print_graph_data(res)
            except Exception as e:
                logger.error(f"[AOP Aspect Error] Failed to print graph_data in wrapped_fetch: {e}")
            return res
            
        GraphService.fetch_random_knowledge_graph = wrapped_fetch
        logger.info("AOP Aspect successfully woven: GraphService.fetch_random_knowledge_graph")
        
        # 4. 织入对 generate_initial_question 的展示切面；保留核心工作流的
        # JSON 修复、复杂度网关、重试和隔离逻辑。
        if getattr(PipelineWorkflow.generate_initial_question, "_visual_printer_wrapped", False):
            logger.info("AOP Aspect already woven: PipelineWorkflow.generate_initial_question")
        else:
            orig_gen_q = PipelineWorkflow.generate_initial_question

            async def wrapped_gen_q(self, context_list, task_id_label=""):
                selected_q = await orig_gen_q(self, context_list, task_id_label=task_id_label)

                # 自动渲染候选问题列表
                try:
                    print_generated_questions([selected_q])
                except Exception as e:
                    logger.error(f"[AOP Aspect Error] Failed to print candidate questions in wrapped_gen_q: {e}")
                return selected_q

            wrapped_gen_q._visual_printer_wrapped = True
            wrapped_gen_q._visual_printer_original = orig_gen_q
            PipelineWorkflow.generate_initial_question = wrapped_gen_q
            logger.info("AOP Aspect successfully woven: PipelineWorkflow.generate_initial_question")
        
    except ImportError as e:
        logger.warning(f"Failed to weave AOP aspects due to import error: {e}. Skipping AOP enhancement.")
    except Exception as e:
        logger.error(f"Error occurred during AOP weaving: {e}")
