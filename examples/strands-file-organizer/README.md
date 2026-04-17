# Strands File Organizer Example

A file organizer agent built with [Strands Agents SDK](https://strandsagents.com/) and the [Filesystem MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem). It scans a target directory, reads files, and suggests how to reorganize them — tested with [agentverify](https://github.com/simukappu/agentverify).

## Prerequisites

- Python 3.10+
- Node.js (for the Filesystem MCP server via `npx`)
- AWS credentials configured (only for recording mode — not needed for cassette replay)

## Setup

See the [main README](../../README.md#examples) for setup instructions (`git clone`, venv, `pip install`).

## Running the Agent

```bash
cd examples/strands-file-organizer
python agent.py
```

### Configuring the Target Directory

The agent analyzes a local directory. You can configure the target in three ways (highest priority first):

1. **CLI argument** `--target-dir`:
   ```bash
   python agent.py --target-dir /path/to/your/project
   ```

2. **Environment variable** `FILE_ORGANIZER_TARGET_DIR`:
   ```bash
   export FILE_ORGANIZER_TARGET_DIR=/path/to/your/project
   python agent.py
   ```

3. **Default**: the agentverify repository root (`../../` relative to this directory).

The resolved path is passed to the Filesystem MCP server as its `allowed_directories`, so the agent can only access files within that directory.

## Running Tests

### Cassette Replay (No API Key Required)

Tests ship with a pre-recorded cassette (`tests/cassettes/file_organizer.yaml`). Just run:

```bash
pytest
```

The cassette is replayed deterministically — no Bedrock API calls, zero cost.

### What the Tests Verify

| Test | agentverify Assertion | Description |
|---|---|---|
| `test_tool_call_sequence` | `assert_tool_calls()` | `list_directory` → `read_file` ordering (IN_ORDER) |
| `test_token_budget` | `assert_cost()` | Total tokens ≤ 5000 |
| `test_safety_read_only` | `assert_no_tool_call()` | `write_file`, `move_file`, `create_directory`, `delete_file`, `rename_file` are never called |

### Recording Mode (Real Bedrock API)

To re-record the cassette with real LLM calls:

1. Configure AWS credentials for Amazon Bedrock.
2. Run the tests with `--cassette-mode=record`:
   ```bash
   pytest --cassette-mode=record
   ```
   The cassette file is saved (or overwritten if it already exists).
3. Commit the updated cassette file.

## Conversion Helper

> **Recommended**: Use the built-in adapter instead of the manual converter:
>
> ```python
> from agentverify.frameworks.strands import from_strands
>
> result = agent("Analyze the file structure")
> execution_result = from_strands(result)
> ```
>
> The built-in adapter performs the same conversion as `converter.py` below. The manual converter is kept as a reference for customization.

`converter.py` provides `strands_result_to_execution_result()` which converts a Strands `AgentResult` into an agentverify `ExecutionResult`. See the inline comments for the full mapping:

| Strands AgentResult | ExecutionResult |
|---|---|
| `state.messages[*].content[*].toolUse.name` | `tool_calls[*].name` |
| `state.messages[*].content[*].toolUse.input` | `tool_calls[*].arguments` |
| `metrics.inputTokens` / `outputTokens` | `token_usage` |
| `message.content[*].text` | `final_output` |
