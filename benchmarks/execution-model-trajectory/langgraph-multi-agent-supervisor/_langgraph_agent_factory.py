"""Bench-local agent factory for the LangGraph multi-agent supervisor subject.

Loads ``examples/langgraph-multi-agent-supervisor/agent.py`` directly via ``importlib`` rather than ``import agent``, because two installed examples (``langgraph-multi-agent-supervisor`` and ``custom-converter-python-agent``) both ship a top-level ``agent`` py-module and the import order is not deterministic.

The model is pinned to ``gpt-4o-mini`` to match the example's recorded cassette and the example README's documented setup. The LangGraph supervisor is sensitive to non-determinism even at ``temperature=0``; if a run produces a malformed trajectory (missing handoff, missing arithmetic step), the test surfaces that as a normal assertion failure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

USER_PROMPT = "What's the combined headcount of the FAANG companies in 2024?"
MODEL_NAME = "gpt-4o-mini"

_EXAMPLE_AGENT_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "langgraph-multi-agent-supervisor"
    / "agent.py"
)


def _load_example_agent_module():
    """Import ``examples/langgraph-multi-agent-supervisor/agent.py`` under a unique name."""
    module_name = "agentverify_bench_langgraph_supervisor_example_agent"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _EXAMPLE_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load LangGraph example agent from {_EXAMPLE_AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_supervisor():
    """Construct the LangGraph supervisor app with the bench-pinned model name."""
    example_agent = _load_example_agent_module()
    return example_agent.build_supervisor_app(model_name=MODEL_NAME)
