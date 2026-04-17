"""Strands Weather Forecaster agent.

Based on the official Strands Agents sample:
https://strandsagents.com/docs/examples/python/weather_forecaster/

Uses the http_request tool to fetch weather data from the
National Weather Service API (no API key required).
"""

from strands import Agent
from strands.models.bedrock import BedrockModel
from strands_tools import http_request

WEATHER_SYSTEM_PROMPT = """You are a weather assistant with HTTP capabilities. You can:
1. Make HTTP requests to the National Weather Service API
2. Process and display weather forecast data
3. Provide weather information for locations in the United States

When retrieving weather information:
1. First get the coordinates or grid information using https://api.weather.gov/points/{latitude},{longitude}
2. Then use the returned forecast URL to get the actual forecast

When displaying responses:
- Format weather data in a human-readable way
- Highlight important information like temperature, precipitation, and alerts
- Handle errors appropriately
- Convert technical terms to user-friendly language

Always explain the weather conditions clearly and provide context for the forecast."""

# Use non-streaming mode for cassette recording compatibility
model = BedrockModel(streaming=False)

weather_agent = Agent(
    model=model,
    system_prompt=WEATHER_SYSTEM_PROMPT,
    tools=[http_request],
)

if __name__ == "__main__":
    response = weather_agent("What's the weather in Seattle?")
    print(response)
