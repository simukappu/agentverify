"""Strands Agents AgentResult to agentverify ExecutionResult conversion helper.

Provides a conversion function that builds an agentverify ExecutionResult
from a Strands Agents SDK AgentResult object.

The Strands AgentResult has the following structure:
- result.message: Final response message (list of content blocks)
- result.metrics: Metrics such as token consumption
- result.state.messages: Conversation history (including tool calls)

Conversion mapping:
  state.messages[*].content[*].toolUse.name   -> tool_calls[*].name
  state.messages[*].content[*].toolUse.input   -> tool_calls[*].arguments
  metrics.inputTokens                          -> token_usage.input_tokens
  metrics.outputTokens                         -> token_usage.output_tokens
  message.content[*].text                      -> final_output
"""

from agentverify import ExecutionResult, TokenUsage, ToolCall


def strands_result_to_execution_result(result) -> ExecutionResult:
    """Build an agentverify ExecutionResult from a Strands Agents AgentResult.

    Args:
        result: A Strands Agents SDK AgentResult object.
            The following attributes are referenced:
            - result.state.messages: List of conversation history messages
            - result.metrics: Token consumption metrics (optional)
            - result.message: Final response message (optional)

    Returns:
        An agentverify ExecutionResult containing tool calls, token usage,
        and final output text.

    Conversion logic:
    1. Extract toolUse entries from the content blocks of each message
       in state.messages
       - Each message's content is a list of blocks (dicts)
       - Retrieve tool name and arguments from blocks that have a toolUse key
    2. Convert each toolUse's name and input into a ToolCall
       - If input is missing, use an empty dict as the default value
    3. Convert inputTokens / outputTokens from metrics into TokenUsage
       - If metrics is not present, token_usage=None
    4. Set the text block from the final message (result.message) as final_output
       - If no text block exists, final_output=None
    """
    # --- Extract tool calls ---
    # Traverse the conversation history (state.messages) and collect toolUse
    # entries from the content blocks of each message. In Strands, when the LLM
    # calls a tool, the message content includes {"toolUse": {"name": ..., "input": ...}}.
    tool_calls: list[ToolCall] = []

    for message in result.state.messages:
        # content is a list of blocks. Skip if not a dict.
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue

        for block in content:
            # Process only blocks that are dicts and have a toolUse key
            if isinstance(block, dict) and "toolUse" in block:
                tool_use = block["toolUse"]
                # Convert to ToolCall: name is used as-is, input is retrieved as a dict.
                # If input is not set, use an empty dict as the default value.
                tool_calls.append(
                    ToolCall(
                        name=tool_use["name"],
                        arguments=tool_use.get("input", {}),
                    )
                )

    # --- Extract token usage ---
    # Retrieve inputTokens / outputTokens from the Strands metrics object.
    # If metrics is not present (e.g., during mock execution), token_usage=None.
    token_usage = None
    if hasattr(result, "metrics") and result.metrics:
        metrics = result.metrics
        token_usage = TokenUsage(
            input_tokens=metrics.get("inputTokens", 0),
            output_tokens=metrics.get("outputTokens", 0),
        )

    # --- Extract final output text ---
    # result.message is the final response message containing a list of content blocks.
    # Retrieve the final output from text blocks ({"text": "..."} format).
    # If there are multiple text blocks, use the last one.
    final_output = None
    if hasattr(result, "message") and result.message:
        message_content = result.message.get("content") if isinstance(result.message, dict) else None
        if isinstance(message_content, list):
            for block in message_content:
                if isinstance(block, dict) and "text" in block:
                    final_output = block["text"]

    return ExecutionResult(
        tool_calls=tool_calls,
        token_usage=token_usage,
        final_output=final_output,
    )
