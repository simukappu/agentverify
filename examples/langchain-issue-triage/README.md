# LangChain Issue Triage Example

A GitHub Issue triage agent built with [LangChain](https://python.langchain.com/) and the [GitHub MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/github). It lists issues, reads details, and suggests labels, priorities, and assignees — tested with [agentverify](https://github.com/simukappu/agentverify).

The agent follows the official [`langchain-mcp-adapters` quickstart pattern](https://github.com/langchain-ai/langchain-mcp-adapters) — `MultiServerMCPClient` plus `create_react_agent` — applied to the GitHub MCP server. agentverify's built-in `from_langchain` adapter extracts the `ExecutionResult` in one call; no custom converter needed for this agent.

## Prerequisites

- Python 3.10+
- Node.js (for the GitHub MCP server via `npx`)
- OpenAI API key (required for running the agent — not needed for cassette replay tests)
- GitHub personal access token (only when using the real GitHub MCP server)

## Setup

See the [main README](../../README.md#examples) for setup instructions (`git clone`, venv, `pip install`).

## Running the Agent

```bash
cd examples/langchain-issue-triage

# With mock MCP server (no GitHub token needed, but OpenAI API key is required)
export OPENAI_API_KEY=sk-your_key_here
python agent.py --mock

# With real GitHub MCP server
export GITHUB_TOKEN=ghp_your_token_here
export OPENAI_API_KEY=sk-your_key_here
python agent.py
```

### Configuring the Target Repository

The agent triages issues from a GitHub repository. You can configure the target in three ways (highest priority first):

1. **CLI argument** `--repo`:
   ```bash
   python agent.py --repo owner/repo
   ```

2. **Environment variable** `ISSUE_TRIAGE_REPO`:
   ```bash
   export ISSUE_TRIAGE_REPO=owner/repo
   python agent.py
   ```

3. **Default**: `simukappu/agentverify`

The `--mock` flag uses the mock server at `../mcp-server/github_server.py` (no GitHub token required). Without `--mock`, the agent uses the real GitHub MCP server (`@modelcontextprotocol/server-github`) which requires a `GITHUB_TOKEN`.

## Running Tests

### Cassette Replay (No API Key Required)

Tests ship with pre-recorded cassettes under `tests/cassettes/`. Just run:

```bash
pytest
```

Both mock and real MCP cassettes are replayed deterministically — no OpenAI or GitHub API calls, zero cost.

### What the Tests Verify

| Test Class | Test | agentverify Assertion | Description |
|---|---|---|---|
| `TestIssueTriage_MockMCP` | `test_tool_call_sequence` | `assert_tool_calls()` | `list_issues` → `get_issue` ordering (IN_ORDER) |
| `TestIssueTriage_MockMCP` | `test_safety_read_and_label_only` | `assert_no_tool_call()` | `close_issue`, `delete_comment`, `delete_issue`, `update_issue`, `create_issue` are never called |
| `TestIssueTriage_RealMCP` | `test_tool_call_with_real_github` | `assert_tool_calls()` | `list_issues` call verified from real GitHub cassette |
| `TestIssueTriageStepLevel` | `test_step_sequence` | `assert_step()` | Each of the four ReAct steps runs the expected tool — `list_issues`, `get_issue`, `list_labels`, then the summary step |
| `TestIssueTriageStepLevel` | `test_step_1_uses_step_0_result` | `assert_step_uses_result_from()` | The `issue_number` passed to `get_issue` in step 1 actually came from step 0's `list_issues` response — catches "agent hallucinated an issue number" bugs |
| `TestIssueTriageStepLevel` | `test_final_step_reports_triage` | `assert_step_output()` | The final step (no tool calls) produces the triage summary text |

The step-level tests work on cassette replay because agentverify automatically backfills tool results from the next step's LLM input — no adapter setup needed. See [Step-Level Assertions](../../README.md#step-level-assertions) in the main README for details.

### Recording Mode (Real APIs)

To re-record cassettes with real LLM and GitHub API calls:

1. Set the required environment variables:
   ```bash
   export OPENAI_API_KEY=sk-your_key_here
   export GITHUB_TOKEN=ghp_your_token_here  # for real MCP cassette
   export ISSUE_TRIAGE_REPO=owner/repo       # target repository to triage
   ```
2. Run the tests with `--cassette-mode=record`:
   ```bash
   pytest --cassette-mode=record
   ```
   Cassette files are saved (or overwritten if they already exist).
3. Commit the updated cassette files.
