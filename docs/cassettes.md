# Cassettes in Detail

The details behind [Record & Replay with Cassettes](../README.md#record--replay-with-cassettes) in the README.

## Recorded and Unsupported Calls

Cassettes record chat calls, streamed or not, sync or async: OpenAI `chat.completions.create` and `responses.create` (and their `stream()` helpers), Anthropic `messages.create` and `messages.stream()`, Gemini `models.generate_content` and `generate_content_stream`, Bedrock `Converse` and `ConverseStream`, and LiteLLM `completion` / `acompletion`. Inside a cassette or `MockLLM`, other LLM calls are handled as follows.

These raise `CassetteUnsupportedCallError`:
- Other streaming calls (OpenAI `responses.connect()` WebSocket sessions, Anthropic `messages.stream(output_format=...)`, Bedrock `InvokeModelWithResponseStream`, ...), in record and replay mode.
- In replay mode, every other request through an OpenAI, Anthropic, or Gemini SDK client (`parse()`, embeddings, beta resources, ...), Bedrock operations that generate (`InvokeModel`, `InvokeAgent`, `RetrieveAndGenerate`, ...), and LiteLLM `text_completion` and `responses`.

These reach the real API:
- In record mode, those other requests. `parse()`, the Bedrock operations, and the LiteLLM calls warn that the cassette will not contain them; embeddings, beta resources, and WebSocket APIs do not. Replay then fails at that call.
- With no cassette file (`AUTO` passthrough), every call. Nothing is saved.

OpenAI models on Amazon Bedrock can be reached three ways; record each with the provider that owns the client:

| Route | Cassette provider |
|---|---|
| boto3 `bedrock-runtime` `converse` | `"bedrock"` |
| OpenAI SDK with `base_url="https://bedrock-runtime.<region>.amazonaws.com/openai/v1"` (`chat.completions.create`, `responses.create`) | `"openai"` |
| boto3 `invoke_model` | not recorded; raises in replay mode |

## Streaming

A streamed call is stored as the same response as a non-streamed one, so a cassette recorded either way replays either way. Frameworks that stream by default (Strands' `BedrockModel`, `Runner.run_streamed` in the OpenAI Agents SDK, LangChain's `stream()`) need no change. Replay yields the text in one chunk and each tool call in one chunk, not in the recorded chunk sizes. A stream the agent stops reading early is recorded with what it read.

## OpenAI Responses API

`responses.create` is recorded in the same `provider="openai"` cassette as Chat Completions, including the `with_raw_response` path that langchain-openai uses. The OpenAI Agents SDK uses it by default. Tool results sent back as `function_call_output` items become the step's `tool_results`, so `assert_step_uses_result_from` works the same way. Only function tools are recorded: a response with a client-executed computer, shell, apply-patch, or custom tool call raises `CassetteUnsupportedCallError`.

With `previous_response_id` or `conversation`, the API keeps earlier turns on the server and each request carries only the new input. The cassette then misses those turns, and agentverify warns. Pass `store=False` and send the full input instead.

## Async and Concurrent Calls

Async clients of OpenAI, Anthropic, Gemini, LiteLLM, and Bedrock (`aiobotocore` / `aioboto3`) are recorded and replayed. agentverify does not depend on `pytest-asyncio` or `anyio`; use whichever you already have, or `asyncio.run()` in a sync test.

Calls may overlap (`asyncio.gather`, threads). Replay matches each request to its recording on the model, tool names, system prompt, user text, and history length, so calls that finish in another order than when recorded still get their own responses; requests identical in all of these replay in recorded order. Concurrent agents keep their own `step_probe` steps and tool results.

## Tool Replay

- Tools are recorded through LangChain / LangGraph, Strands Agents, OpenAI Agents SDK, Pydantic AI, Google ADK, CrewAI, and MCP's `ClientSession.call_tool`, including agents created before the cassette. For a hand-written agent, decorate the tool function with `@cassette_tool`.
- Calls match on the tool name and arguments, in any order. A call with no recorded match raises `CassetteToolMismatchError`, and fails the test when the cassette closes, even if the framework passed the error to the model.
- A tool error replays as it was recorded: the error result the framework made of it, or the same exception type and message (`agentverify.cassette.tools.ReplayedToolError` when the type cannot be rebuilt from its message).
- Some calls run for real on replay too. Results that do not survive YAML (a LangGraph `Command`, binary content), CrewAI tools with `result_as_answer`, and LangGraph interrupts are marked `run_on_replay` in the cassette. Tools that steer the run (Google ADK `transfer_to_agent` / `exit_loop`, Strands `handoff_to_agent`) and the tool behind Strands' `structured_output_model` are not recorded at all.
- A replayed tool that called the LLM while recording (an agent used as a tool) skips those LLM calls on replay, so the outer agent's calls stay in step.
- Replay returns the recorded result only. What a tool does to the agent's own state is not replayed: Google ADK `tool_context.state`, Pydantic AI `deps`, an OpenAI Agents SDK context object, or anything else the tool changes in the process. LangChain tool callbacks (`on_tool_start` / `on_tool_end`) do not fire for a replayed call either. Pass `replay_tools=False` when a test depends on these.
- MCP servers still start on replay, since the agent lists their tools.
- Cassettes recorded before 0.7.0 have no `tool_executions`, and their tools run for real.

## Request Matching

Each replayed request is compared with the recorded one:
- **Model name** (an empty model on either side skips the check)
- **Tool names**, and each tool's **parameter schema** (a request with no tools on either side skips the check)
- **System prompt**: system and developer messages, and the providers' `system` / `instructions` parameters
- **User message text**. Tool results in their own messages or blocks are not compared, since a tool running for real may return something new.

A field the cassette did not record (no system prompt, or a cassette from an older version) is not compared. The request is sanitized like the cassette before comparing. A difference raises `CassetteRequestMismatchError` with a diff, and fails the test when the cassette closes even if the agent's framework caught it.

## Drift

Request matching leaves tool results alone. Drift compares the rest: tool results (key by key, when they are JSON, so a new `null` counts), the number and roles of the messages, and the provider's other parameters.

- `off` (default), `warn`, or `fail`, from the `agentverify_cassette_drift` setting, `--cassette-drift`, or `cassette(drift=...)` (the later ones win). It also applies with `match_requests=False`.
- `warn` adds the cassette and the places that differ (never the values) to the stale list at the end of the run; `fail` raises `CassetteDriftError`.
- `drift_ignore` takes globs over the places as `iter_request_texts` names them (`request[*].message[*].tool_result[*].now`), with or without the `(role)` and `(tool)` parts. `match_ignore` applies too.
- It is off by default because a tool that runs for real may return something new on every run.

## Expiry and the Cassette Report

- Replaying a cassette recorded longer ago than `max_age` raises `CassetteExpiredError` before the agent runs (`on_expired="warn"` or `--cassette-expired=warn` only warns), and the cassette joins the stale list.
- The age comes from the cassette's `metadata.recorded_at`; a value without a time zone is UTC, and a cassette without one counts as expired.
- The limit comes from `agentverify_cassette_max_age`, `--cassette-max-age`, or `cassette(max_age=...)` (the later ones win). The setting and the option apply to cassettes opened with the `cassette` fixture; an `LLMCassetteRecorder` you create yourself checks only the `max_age` you pass it.
- `pytest --cassette-report` runs no tests and lists the cassettes in the `cassettes/` directory next to each collected test file (not a custom `cassette_dir`): when and with which model each was recorded, and whether it is past the suite or run limit (not a `max_age=` passed to one cassette). Files there that are not cassettes are left out. It needs a run without pytest-xdist workers (`-p no:xdist`).

## Checking What the Model Was Sent

`iter_request_texts` reads a cassette path, an `LLMCassetteRecorder` or `MockLLM` after the run, or one `NormalizedRequest`. `where` reads like `request[0].system`, `request[0].message[2](assistant).tool_call[0](get_weather).arguments.city`, `request[0].message[3](tool).tool_result[0](get_weather).temp`, `request[0].tool[get_weather].schema.properties.unit.enum[1]`, or `request[0].parameters.temperature`.

- Messages carry their role, so `(assistant)` picks out the model's own answers; tool results carry the tool's name.
- JSON tool arguments and results are read key by key, and numbers and booleans come as text.
- Provider ids (`toolUseId`, `tool_call_id`), reasoning signatures, and embedded images or files are left out of the provider's envelope; inside tool arguments, results, and schemas every key is kept.
- A saved cassette holds the sanitized request; while recording, the recorder keeps the request as sent.

`assert_no_credential_urls` searches every one of those texts. A URL fails when a query or fragment name is in `agentverify.credentials.CREDENTIAL_QUERY_NAMES` (`token`, `access_token`, `security_token`, `api_key`, `key`, `secret`, `password`, `sig`, `signature`, the AWS and Google signature names, and more) or a JWT sits in its path, query, or fragment. HTML entities such as `&amp;` are decoded first. `extra_patterns` adds regular expressions (`r"[?&]code="` for OAuth codes) and `ignore` drops default names (`key` gives false alarms on APIs that use it for ordinary values). The error lists where each URL was, its host, and why, never the values.
