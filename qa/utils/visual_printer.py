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
