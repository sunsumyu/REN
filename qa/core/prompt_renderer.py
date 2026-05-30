# -*- coding: utf-8 -*-
from jinja2 import Template
import prompts

class PromptRenderer:
    """
    Surgically decouples prompt rendering from execution logic.
    Provides standard methods to fetch and render templates.
    """
    @staticmethod
    def render(template_str: str, **kwargs) -> str:
        return prompts.render_prompt(template_str, **kwargs)

    @staticmethod
    def get_l1_meta() -> str:
        return prompts.L1_SYSTEM_META_TEMPLATE

    @staticmethod
    def get_l2_execution(facet: str) -> str:
        return PromptRenderer.render(prompts.L2_TASK_EXECUTION_TEMPLATE, facet=facet)

    @staticmethod
    def get_l3_context(query: str, refs: list, history: list) -> str:
        return PromptRenderer.render(
            prompts.L3_DYNAMIC_CONTEXT_TEMPLATE,
            query=query,
            refs=refs,
            history=history
        )
