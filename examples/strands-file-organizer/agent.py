"""Strands Agents file organizer agent.

Uses a Filesystem MCP server to analyze and suggest organization for the
file structure of a target directory.
The target directory is configured via CLI argument, environment variable,
or default (repository root).
"""

import argparse
import os
from pathlib import Path

from strands import Agent
from strands.tools.mcp import MCPClient
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters


def resolve_target_dir(cli_target_dir: str | None = None) -> Path:
    """Resolve the target directory.

    Priority:
    1. CLI argument --target-dir
    2. Environment variable FILE_ORGANIZER_TARGET_DIR
    3. Default: repository root (../../ from examples/strands-file-organizer/)
    """
    if cli_target_dir:
        return Path(cli_target_dir).resolve()

    env_dir = os.environ.get("FILE_ORGANIZER_TARGET_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    # Default: repository root relative to this file
    return (Path(__file__).parent / ".." / "..").resolve()


def create_file_organizer_agent(target_dir: Path) -> Agent:
    """Build a file organizer agent using the Filesystem MCP server.

    Args:
        target_dir: Directory path to analyze.
            Set as allowed_directories for the Filesystem MCP server.
    """
    # Strands MCPClient accepts a transport factory (callable).
    # Create a function that passes StdioServerParameters to stdio_client.
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(target_dir),  # passed as allowed_directories
        ],
    )
    mcp_client = MCPClient(lambda: stdio_client(server_params))

    with mcp_client:
        tools = mcp_client.list_tools_sync()
        agent = Agent(
            system_prompt=(
                "You are a file organization assistant. "
                "Analyze the file structure of the specified directory "
                "and suggest how to organize and categorize the files. "
                "Use read-only operations; do not write, move, or delete any files. "
                "IMPORTANT: Use list_directory (not directory_tree) to explore the structure incrementally. "
                "Skip hidden directories, .venv, .git, node_modules, __pycache__, "
                "*.egg-info, and dist directories."
            ),
            tools=tools,
        )
    return agent, mcp_client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", help="Target directory to analyze")
    args = parser.parse_args()

    target = resolve_target_dir(args.target_dir)
    agent, mcp_client = create_file_organizer_agent(target)
    with mcp_client:
        result = agent(f"Analyze the file structure of {target} and suggest how to organize it.")
        print(result)
