# -*- coding: utf-8 -*-
# ==============================================================================
# DEPRECATED: This file is deprecated to avoid naming collisions with the root
# prompt_manager.py (which manages prompt versioning in SQLite).
#
# Please use `core/prompt_renderer.py` and `PromptRenderer` instead.
# ==============================================================================
raise ImportError(
    "core/prompt_manager.py is DEPRECATED and has been renamed to "
    "core/prompt_renderer.py. Please import PromptRenderer from core.prompt_renderer instead."
)
