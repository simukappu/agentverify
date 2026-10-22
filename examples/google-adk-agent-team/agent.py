"""A Google ADK agent team: a weather agent that delegates greetings and farewells to two sub-agents.

Adapted from Step 3 of the official ADK tutorial "Build Your First Intelligent Agent Team" (https://google.github.io/adk-docs/tutorials/agent-team/). The tools, instructions, and the three-turn conversation are the tutorial's. The models run on Amazon Bedrock through ADK's ``LiteLlm`` wrapper, which the tutorial's Step 2 uses for other providers.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# The tutorial defaults to Gemini. This example uses Claude Haiku 5.5 on Amazon Bedrock, which
# LiteLLM reaches with the standard AWS credential chain (no API key argument).
DEFAULT_MODEL = "bedrock/global.anthropic.claude-haiku-5-5"

APP_NAME = "weather_tutorial_agent_team"
USER_ID = "user_1_agent_team"
SESSION_ID = "session_001_agent_team"

# The tutorial's conversation: a greeting, a weather request, a farewell.
CONVERSATION = ["Hello there!", "What is the weather in New York?", "Thanks, bye!"]


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    city_normalized = city.lower().replace(" ", "")  # Basic normalization

    # Mock weather data
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}


def say_hello(name: Optional[str] = None) -> str:
    """Provides a simple greeting. If a name is provided, it will be used.

    Args:
        name (str, optional): The name of the person to greet. Defaults to a generic greeting if not provided.

    Returns:
        str: A friendly greeting message.
    """
    if name:
        return f"Hello, {name}!"
    return "Hello there!"  # Default greeting if name is None or not explicitly passed


def say_goodbye() -> str:
    """Provides a simple farewell message to conclude the conversation."""
    return "Goodbye! Have a great day."


def build_team(model: str = DEFAULT_MODEL) -> Agent:
    greeting_agent = Agent(
        model=LiteLlm(model=model),
        name="greeting_agent",
        instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
        "Use the 'say_hello' tool to generate the greeting. "
        "If the user provides their name, make sure to pass it to the tool. "
        "Do not engage in any other conversation or tasks.",
        description="Handles simple greetings and hellos using the 'say_hello' tool.",
        tools=[say_hello],
    )
    farewell_agent = Agent(
        model=LiteLlm(model=model),
        name="farewell_agent",
        instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message. "
        "Use the 'say_goodbye' tool when the user indicates they are leaving or ending the conversation "
        "(e.g., using words like 'bye', 'goodbye', 'thanks bye', 'see you'). "
        "Do not perform any other actions.",
        description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.",
        tools=[say_goodbye],
    )
    return Agent(
        name="weather_agent_v2",
        model=LiteLlm(model=model),
        description="The main coordinator agent. Handles weather requests and delegates greetings/farewells to specialists.",
        instruction="You are the main Weather Agent coordinating a team. Your primary responsibility is to provide weather information. "
        "Use the 'get_weather' tool ONLY for specific weather requests (e.g., 'weather in London'). "
        "You have specialized sub-agents: "
        "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
        "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
        "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'. "
        "If it's a weather request, handle it yourself using 'get_weather'. "
        "For anything else, respond appropriately or state you cannot handle it.",
        tools=[get_weather],
        sub_agents=[greeting_agent, farewell_agent],
    )


async def run_conversation(queries: list[str] = CONVERSATION, model: str = DEFAULT_MODEL) -> tuple[list[Any], list[str]]:
    """Run the queries in one session; return every event the runner yielded and each turn's final response."""
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=build_team(model), app_name=APP_NAME, session_service=session_service)
    events: list[Any] = []
    answers: list[str] = []
    for query in queries:
        content = types.Content(role="user", parts=[types.Part(text=query)])
        answer = "Agent did not produce a final response."
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            events.append(event)
            text = "".join(p.text for p in (event.content.parts if event.content else None) or [] if p.text and not p.thought)
            if event.is_final_response() and text:
                answer = text
        answers.append(answer)
    return events, answers


if __name__ == "__main__":
    _, answers = asyncio.run(run_conversation(model=os.environ.get("LITELLM_MODEL", DEFAULT_MODEL)))
    for query, answer in zip(CONVERSATION, answers):
        print(f">>> {query}\n<<< {answer}")
