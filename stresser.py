import json
import time
import random
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


@dataclass
class ConversationStats:
    """Statistics collected during a conversation stress test."""
    conversation_id: str
    total_messages_sent: int = 0
    total_user_messages: int = 0
    total_ai_responses: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_tokens: int = 0
    total_latency_seconds: float = 0.0
    average_latency_seconds: float = 0.0
    min_latency_seconds: float = float('inf')
    max_latency_seconds: float = 0.0
    total_sleep_time_seconds: float = 0.0
    message_latencies: list[float] = field(default_factory=list)
    error: Optional[str] = None


def clean_response_content(content: str) -> str:
    """
    Clean response content by removing thinking tags and other artifacts.

    Some models include <think>...</think> tags or other internal reasoning
    that should be stripped before using the response in conversation history.

    Args:
        content: Raw response content

    Returns:
        Cleaned content
    """
    if not content:
        return ""

    # Remove <think>...</think> tags and their content
    cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)

    # Remove any other common XML-style tags that might appear
    cleaned = re.sub(r'</?(?:reasoning|internal|scratch)>', '', cleaned, flags=re.IGNORECASE)

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def calculate_sleep_time(message_length: int, base_time: float = 0.5, time_per_char: float = 0.01) -> float:
    """
    Calculate sleep time based on message length.
    Simulates thinking/typing time proportional to message length.

    Args:
        message_length: Number of characters in the next message
        base_time: Base sleep time in seconds
        time_per_char: Additional time per character

    Returns:
        Sleep time in seconds with some randomness
    """
    calculated_time = base_time + (message_length * time_per_char)
    # Add some randomness (±20%)
    randomness = random.uniform(0.8, 1.2)
    return calculated_time * randomness


def run_conversation_stress_test(
    conversation_file_path: str,
    model_name: str,
    temperature: float,
    max_tokens: Optional[int] = None,
    max_messages: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None
) -> ConversationStats:
    """
    Run a stress test conversation against an LLM API.

    This function loads a conversation from a JSON file and replays it against the LLM API,
    sending user messages one at a time and collecting statistics. After each response,
    it sleeps for a time proportional to the next message length before sending the next
    message along with the simulated conversation history.

    Args:
        conversation_file_path: Path to the JSON conversation file
        model_name: Name of the model to use (e.g., 'gpt-3.5-turbo')
        temperature: Temperature parameter for the model
        max_tokens: Maximum tokens for the response (None = no limit)
        max_messages: Maximum number of user messages to send (None = all)
        api_key: OpenAI API key (optional, uses env var if not provided)
        api_base: API base URL (optional, uses default if not provided)

    Returns:
        ConversationStats object with collected statistics
    """
    # Load conversation from JSON file
    conversation_path = Path(conversation_file_path)
    if not conversation_path.exists():
        raise FileNotFoundError(f"Conversation file not found: {conversation_file_path}")

    with open(conversation_path, 'r') as f:
        conversation_data = json.load(f)

    conversation_id = conversation_data.get('conversation_id', conversation_path.stem)
    messages = conversation_data.get('messages', [])

    # Extract only user messages from the conversation
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    simulated_responses = [msg for msg in messages if msg['role'] == 'assistant']

    # Initialize statistics
    stats = ConversationStats(conversation_id=conversation_id)

    # Determine how many messages to send
    messages_to_send = len(user_messages)
    if max_messages is not None:
        messages_to_send = min(messages_to_send, max_messages)

    if messages_to_send == 0:
        return stats

    # Initialize LangChain ChatOpenAI
    llm_kwargs = {
        'model': model_name,
        'temperature': temperature,
    }
    # Only set max_tokens if provided (None = no limit)
    if max_tokens is not None:
        llm_kwargs['max_tokens'] = max_tokens
    if api_key:
        llm_kwargs['api_key'] = api_key
    if api_base:
        llm_kwargs['base_url'] = api_base

    try:
        llm = ChatOpenAI(**llm_kwargs)

        # Build conversation history progressively
        conversation_history = []

        for idx in range(messages_to_send):
            user_msg_content = user_messages[idx]['content']

            # Add current user message to history
            conversation_history.append(HumanMessage(content=user_msg_content))

            # Send to LLM and measure latency
            start_time = time.time()

            try:
                response = llm.invoke(conversation_history)
                end_time = time.time()

                latency = end_time - start_time
                stats.message_latencies.append(latency)
                stats.total_latency_seconds += latency
                stats.min_latency_seconds = min(stats.min_latency_seconds, latency)
                stats.max_latency_seconds = max(stats.max_latency_seconds, latency)

                # Extract token usage from response metadata
                if hasattr(response, 'response_metadata'):
                    token_usage = response.response_metadata.get('token_usage', {})
                    stats.total_tokens_input += token_usage.get('prompt_tokens', 0)
                    stats.total_tokens_output += token_usage.get('completion_tokens', 0)
                    stats.total_tokens += token_usage.get('total_tokens', 0)

                stats.total_user_messages += 1
                stats.total_ai_responses += 1
                stats.total_messages_sent += 1

                # Replace the real AI response with the simulated one from JSON
                # This maintains the scripted conversation flow and prevents issues
                # with models that generate verbose or malformed responses
                if idx < len(simulated_responses):
                    simulated_content = simulated_responses[idx]['content']
                    # Ensure the simulated content is valid and not empty
                    if simulated_content and isinstance(simulated_content, str):
                        conversation_history.append(AIMessage(content=simulated_content))
                    else:
                        # Fall back to actual response if simulated is invalid
                        cleaned_content = clean_response_content(response.content)
                        conversation_history.append(AIMessage(content=cleaned_content))
                else:
                    # If we don't have a simulated response, use the actual one (cleaned)
                    # Make sure the content is valid
                    content = response.content if hasattr(response, 'content') and response.content else ""
                    if content:
                        cleaned_content = clean_response_content(content)
                        if cleaned_content:
                            conversation_history.append(AIMessage(content=cleaned_content))

                # Sleep before next message (if there is a next message)
                if idx + 1 < messages_to_send:
                    next_msg_length = len(user_messages[idx + 1]['content'])
                    sleep_time = calculate_sleep_time(next_msg_length)
                    stats.total_sleep_time_seconds += sleep_time
                    time.sleep(sleep_time)

            except Exception as e:
                stats.error = f"Error during message {idx + 1}: {str(e)}"
                break

        # Calculate average latency
        if stats.total_ai_responses > 0:
            stats.average_latency_seconds = stats.total_latency_seconds / stats.total_ai_responses

        # Reset min latency if no messages were sent
        if stats.min_latency_seconds == float('inf'):
            stats.min_latency_seconds = 0.0

    except Exception as e:
        stats.error = f"Error initializing LLM: {str(e)}"

    return stats
