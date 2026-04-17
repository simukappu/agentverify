"""Built-in framework adapters for agentverify.

Provides convenience functions to build ExecutionResult directly from
popular agent framework outputs without writing a custom converter.

Supported frameworks:
- Strands Agents: ``from agentverify.frameworks.strands import from_strands``
- LangChain: ``from agentverify.frameworks.langchain import from_langchain``
"""
