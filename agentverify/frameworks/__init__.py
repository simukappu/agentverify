"""Built-in framework adapters for agentverify.

Provides convenience functions to build ExecutionResult directly from popular agent framework outputs without writing a custom converter.

Supported frameworks:
- Strands Agents: ``from agentverify.frameworks.strands import from_strands``
- LangChain: ``from agentverify.frameworks.langchain import from_langchain``
- LangGraph: ``from agentverify.frameworks.langgraph import from_langgraph``
- OpenAI Agents SDK: ``from agentverify.frameworks.openai_agents import from_openai_agents``
- Pydantic AI: ``from agentverify.frameworks.pydantic_ai import from_pydantic_ai``
- Google ADK: ``from agentverify.frameworks.google_adk import from_google_adk``
- CrewAI: ``from agentverify.frameworks.crewai import from_crewai``

Other frameworks that export OpenTelemetry spans: ``from agentverify.traces import from_otel``.
"""
