"""Async agent on the OpenAI Agents SDK, using LiteLLM to reach any model provider.

Adapted from the official ``examples/model_providers/litellm_provider.py`` sample (https://github.com/openai/openai-agents-python/blob/main/examples/model_providers/litellm_provider.py). ``Runner.run`` is async, and ``LitellmModel`` calls ``litellm.acompletion``, so this is the example that exercises agentverify's async LiteLLM recording.
"""

from __future__ import annotations

import asyncio
import os

from agents import Agent, Runner, RunResult, set_tracing_disabled
from agents.decorators import tool
from agents.extensions.models.litellm_model import LitellmModel

# The official sample defaults to OpenRouter. This example defaults to Claude Haiku 5.5 on Amazon
# Bedrock, which LiteLLM reaches with the standard AWS credential chain (no API key argument).
DEFAULT_MODEL = "bedrock/global.anthropic.claude-haiku-5-5"

set_tracing_disabled(disabled=True)


@tool
def get_weather(city: str):
    return f"The weather in {city} is sunny."


def build_agent(model: str = DEFAULT_MODEL) -> Agent:
    return Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=LitellmModel(model=model),
        tools=[get_weather],
    )


async def run_agent(question: str = "What's the weather in Tokyo?", model: str = DEFAULT_MODEL) -> RunResult:
    return await Runner.run(build_agent(model), question)


if __name__ == "__main__":
    result = asyncio.run(run_agent(model=os.environ.get("LITELLM_MODEL", DEFAULT_MODEL)))
    print(result.final_output)
