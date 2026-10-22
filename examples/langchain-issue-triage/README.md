# LangChain Issue Triage Example

A GitHub Issue triage agent built with [LangChain](https://python.langchain.com/) and the [GitHub MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/github). It lists issues, reads details, and suggests labels, priorities, and assignees. Tested with [agentverify](https://github.com/simukappu/agentverify).

The agent follows the official [`langchain-mcp-adapters` quickstart pattern](https://github.com/langchain-ai/langchain-mcp-adapters) (`MultiServerMCPClient` plus `create_react_agent`) applied to the GitHub MCP server. The tests build the `ExecutionResult` from the cassette with `rec.to_execution_result()`, since the agent's MCP tool calls are recorded along with the LLM calls; no custom converter needed for this agent.

## Prerequisites

- Python 3.10+
- Node.js (for the GitHub MCP server via `npx`)
- OpenAI API key (required for running the agent; not needed for cassette replay tests)
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

The mock-server tests run the agent against the mock GitHub MCP server, replaying the model's responses and the tool results from `issue_triage_mock.yaml`. The real-server cassette was recorded against the real GitHub MCP server, which needs a token to start, so its test asserts on the recording without running the agent. No OpenAI or GitHub API calls, zero cost.

### What the Tests Verify

| Test Class | Test | agentverify Assertion | Description |
|---|---|---|---|
| `TestIssueTriage_MockMCP` | `test_tool_call_sequence` | `assert_tool_calls()` | `list_issues` → `get_issue` ordering (IN_ORDER) |
| `TestIssueTriage_MockMCP` | `test_safety_read_and_label_only` | `assert_no_tool_call()` | `close_issue`, `delete_comment`, `delete_issue`, `update_issue`, `create_issue` are never called |
| `TestIssueTriage_RealMCP` | `test_tool_call_with_real_github` | `assert_tool_calls()` | `list_issues` call verified from real GitHub cassette |
| `TestIssueTriageStepLevel` | `test_step_sequence` | `assert_step()` | Step 0 runs `list_issues`; step 1 runs `list_labels` and a `get_issue` for each listed issue, in parallel; step 2 is the summary |
| `TestIssueTriageStepLevel` | `test_step_1_uses_step_0_result` | `assert_step_uses_result_from()` | The `issue_number` passed to `get_issue` in step 1 actually came from step 0's `list_issues` response. Catches "agent hallucinated an issue number" bugs |
| `TestIssueTriageStepLevel` | `test_final_step_reports_triage` | `assert_step_output()` | The final step (no tool calls) produces the triage summary text |
| `TestIssueTriageToolResults` | `test_no_tool_errored` | `assert_no_tool_errors()` | No GitHub read in the run came back as an error |
| `TestIssueTriageToolResults` | `test_list_issues_returned_usable_result` | `assert_tool_invocation_succeeded()` | Step 0's `list_issues` produced a usable (non-error) result |
| `TestIssueTriageToolResults` | `test_list_issues_result_content` | `assert_tool_result_matches()` | The issue list came back with the labels the triage logic needs |
| `TestIssueTriageToolResults` | `test_get_issue_returned_requested_number` | `assert_tool_result_matches()` | Step 1's `get_issue` calls returned the issues they asked for, #2 among them |

The step-level tests work on cassette replay because each step's tool results are read from the request that continues its conversation. No adapter setup needed. See [Step-Level Assertions](../../README.md#step-level-assertions) in the main README for details.

### Recording Mode (Real LLM)

To re-record the mock-server cassette with real LLM calls (the mock server answers the tools):

1. Set your OpenAI key:
   ```bash
   export OPENAI_API_KEY=sk-your_key_here
   ```
2. Re-record from a single test, so the file ends up in a deterministic state:
   ```bash
   pytest -k "TestIssueTriage_MockMCP and test_tool_call_sequence" --cassette-mode=record
   ```
   The real-server cassette is not re-recorded by the tests, since its test does not run the agent.
3. Commit the updated cassette files.
