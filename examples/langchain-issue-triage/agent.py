"""LangChain GitHub Issue triage agent.

Uses a GitHub MCP server (or mock MCP server) to analyze issues in a
target repository and suggest labels, priorities, and assignees.
The target repository is configured via CLI argument, environment variable,
or default (simukappu/agentverify).
"""

import argparse
import asyncio
import os
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

try:
    # LangGraph v1.0+: create_react_agent moved to langchain.agents.create_agent
    from langchain.agents import create_agent as _create_agent

    def create_react_agent(llm, tools, prompt=None):
        return _create_agent(llm, tools, system_prompt=prompt)
except ImportError:
    from langgraph.prebuilt import create_react_agent


DEFAULT_REPO = "simukappu/agentverify"
DEFAULT_MODEL = "gpt-4o-mini"


def resolve_target_repo(cli_repo: str | None = None) -> str:
    """Resolve the target repository.

    Priority:
    1. CLI argument --repo
    2. Environment variable ISSUE_TRIAGE_REPO
    3. Default: simukappu/agentverify

    Returns:
        A string in "owner/repo" format
    """
    if cli_repo:
        return cli_repo

    env_repo = os.environ.get("ISSUE_TRIAGE_REPO")
    if env_repo:
        return env_repo

    return DEFAULT_REPO


async def run_issue_triage(
    target_repo: str,
    use_mock: bool = False,
) -> dict:
    """Run the Issue triage agent using a GitHub MCP server.

    Args:
        target_repo: Target repository in "owner/repo" format
        use_mock: If True, use the mock GitHub MCP server.

    Returns:
        Agent execution result (dict containing messages)
    """
    if use_mock:
        mock_server_path = str(
            (Path(__file__).parent / ".." / "mcp-server" / "github_server.py").resolve()
        )
        mcp_config = {
            "github": {
                "command": "python",
                "args": [mock_server_path],
                "env": {"MOCK_REPO": target_repo},
                "transport": "stdio",
            }
        }
    else:
        mcp_config = {
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.environ.get("GITHUB_TOKEN", ""),
                },
                "transport": "stdio",
            }
        }

    model = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    llm = ChatOpenAI(model=model, temperature=0)

    system_message = (
        f"You are a GitHub Issue triage assistant. "
        f"Analyze the issues in the repository {target_repo} "
        f"and suggest labels, priorities, and assignees. "
        f"Only perform read operations and add labels; "
        f"do not close or delete any issues."
    )

    # langchain-mcp-adapters 0.1.0+ does not support context manager usage.
    # Use the flow: create instance -> get_tools() -> run agent.
    mcp_client = MultiServerMCPClient(mcp_config)
    tools = await mcp_client.get_tools()
    agent = create_react_agent(llm, tools, prompt=system_message)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": f"Please triage the issues in {target_repo}."}]}
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GitHub Issue triage agent"
    )
    parser.add_argument("--repo", help="Target repository (owner/repo)")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the mock MCP server",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"OpenAI model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    repo = resolve_target_repo(args.repo)
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model
    result = asyncio.run(run_issue_triage(repo, use_mock=args.mock))

    # Print the final message
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)
