# Mock GitHub MCP Server

A [FastMCP](https://github.com/modelcontextprotocol/python-sdk)-based mock GitHub MCP server for the agentverify examples. Returns hardcoded issue data so you can run the [LangChain Issue Triage example](../langchain-issue-triage/) without a real GitHub token.

## Available Tools

| Tool | Description |
|---|---|
| `list_issues(repo, state)` | Returns a list of mock issues (filtered by state) |
| `get_issue(repo, issue_number)` | Returns details for a specific issue (or an error response for unknown issues) |
| `list_labels(repo)` | Returns the list of available labels |

The mock data includes three issue types: a feature request, a bug report, and a question — modeled after realistic agentverify issues.

## Starting the Server

The mock server is typically started automatically by the LangChain agent when `--mock` is passed. If you want to run it standalone:

```bash
python github_server.py
```

The server communicates over stdio (MCP's standard transport).

## Connection Configuration

When used from the LangChain issue triage agent, the server is launched as a subprocess via MCP's `StdioServerParameters`:

```python
from mcp import StdioServerParameters

server_params = StdioServerParameters(
    command="python",
    args=["../mcp-server/github_server.py"],
    env={"MOCK_REPO": "simukappu/agentverify"},
)
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MOCK_REPO` | Target repository name returned in responses | `simukappu/agentverify` |
