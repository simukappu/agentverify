"""LangChain AgentExecutor output to agentverify ExecutionResult conversion helper.

Provides a conversion function that builds an agentverify ExecutionResult
from the dictionary output of a LangChain AgentExecutor.

The LangChain AgentExecutor returns the following structure:
- result["output"]: Final response text
- result["intermediate_steps"]: List of [(AgentAction, observation), ...]

Token consumption is not included in the AgentExecutor output, so it is
aggregated separately from AIMessage.usage_metadata in the conversation history.

Conversion mapping:
  intermediate_steps[*][0].tool        -> tool_calls[*].name
  intermediate_steps[*][0].tool_input  -> tool_calls[*].arguments
  AIMessage.usage_metadata.input_tokens  -> token_usage.input_tokens (sum of all AIMessages)
  AIMessage.usage_metadata.output_tokens -> token_usage.output_tokens (sum of all AIMessages)
  output                               -> final_output
"""

from langchain_core.messages import AIMessage

from agentverify import ExecutionResult, TokenUsage, ToolCall


def langchain_result_to_execution_result(
    result: dict,
    messages: list | None = None,
) -> ExecutionResult:
    """Build an agentverify ExecutionResult from a LangChain AgentExecutor output.

    Args:
        result: Return value of AgentExecutor.invoke() (dict).
            The following keys are referenced:
            - result["output"]: Final response text
            - result["intermediate_steps"]: Tool call history
        messages: LangChain conversation history message list (optional).
            Aggregates token consumption from AIMessage usage_metadata.
            Since the AgentExecutor output does not include token information,
            pass messages obtained separately from memory or callbacks.

    Returns:
        An agentverify ExecutionResult containing tool calls, token usage,
        and final output text.

    Conversion logic:
    1. Extract tool and tool_input from each AgentAction in intermediate_steps
       - intermediate_steps is a list of tuples [(AgentAction, observation), ...]
       - AgentAction.tool is the tool name, AgentAction.tool_input is the arguments
    2. Convert each action to ToolCall(name=tool, arguments=tool_input)
       - If tool_input is not a dict (e.g., a string), use an empty dict as default
    3. Aggregate token consumption from AIMessage.usage_metadata in messages
       - Sum input_tokens / output_tokens across all AIMessages
       - If messages is not specified or contains no AIMessages, token_usage=None
    4. Set result["output"] as final_output
    """
    # --- Extract tool calls ---
    # intermediate_steps is the tool call history recorded by AgentExecutor.
    # Each element is a tuple of (AgentAction, observation), where
    # AgentAction.tool holds the tool name and AgentAction.tool_input holds the arguments.
    tool_calls: list[ToolCall] = []

    for action, _observation in result.get("intermediate_steps", []):
        # tool_input is usually a dict, but can also be a string or other type.
        # If it is not a dict, use an empty dict as the default value
        # to maintain consistency with the ToolCall arguments type (dict).
        arguments = action.tool_input if isinstance(action.tool_input, dict) else {}

        tool_calls.append(
            ToolCall(
                name=action.tool,
                arguments=arguments,
            )
        )

    # --- Extract token usage ---
    # The AgentExecutor output dict does not include token information,
    # so we aggregate usage_metadata from AIMessages in the conversation history.
    # usage_metadata is the token consumption returned by the LLM provider,
    # in the format {"input_tokens": N, "output_tokens": M}.
    # If there are multiple LLM calls (tool call loops), the values are summed.
    token_usage = None
    if messages:
        total_input = 0
        total_output = 0
        for msg in messages:
            # Only AIMessages have token consumption (HumanMessages etc. do not)
            if isinstance(msg, AIMessage) and msg.usage_metadata:
                total_input += msg.usage_metadata.get("input_tokens", 0)
                total_output += msg.usage_metadata.get("output_tokens", 0)
        # Generate TokenUsage only if any token count is greater than 0.
        # If all are 0, consider metrics unavailable and set to None.
        if total_input > 0 or total_output > 0:
            token_usage = TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
            )

    # --- Extract final output text ---
    # AgentExecutor stores the final response in result["output"].
    # If the output key does not exist, final_output=None.
    return ExecutionResult(
        tool_calls=tool_calls,
        token_usage=token_usage,
        final_output=result.get("output"),
    )
