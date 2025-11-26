import json
import time
import random
import re
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime

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


@dataclass
class UserSessionStats:
    """Aggregate statistics from a simulated user session."""
    session_start: datetime
    session_end: datetime
    duration_seconds: float
    total_conversations: int
    successful_conversations: int
    failed_conversations: int
    total_messages_sent: int
    total_tokens: int
    total_tokens_input: int
    total_tokens_output: int
    average_latency_seconds: float
    min_latency_seconds: float
    max_latency_seconds: float
    conversations_per_minute: float
    models_used: List[str] = field(default_factory=list)
    conversation_stats: List[ConversationStats] = field(default_factory=list)


def get_available_models(api_key: str, api_base: str) -> List[str]:
    """
    Fetch available models from the API endpoint.

    Args:
        api_key: API key for authentication
        api_base: Base URL for the API

    Returns:
        List of available model IDs
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base)
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        # Fallback to some common models if API call fails
        print(f"Warning: Could not fetch models from API: {e}")
        return ['gpt-3.5-turbo', 'gpt-4']


def simulate_user_session(
    conversations_dir: str = 'conversations',
    duration_seconds: int = 60,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature_range: tuple[float, float] = (0.5, 1.0)
) -> UserSessionStats:
    """
    Simulate a user running stress tests repeatedly for a specified duration.

    Randomly selects conversations and parameters for each test iteration.
    Continues running until the duration expires.

    Args:
        conversations_dir: Directory containing conversation JSON files
        duration_seconds: How long to run the simulation (default: 60 seconds)
        model_name: Model to use (if None, randomly selects from available models)
        api_key: API key (required)
        api_base: API base URL (required)
        temperature_range: Range for random temperature selection (min, max)

    Returns:
        UserSessionStats object with aggregated results
    """
    if not api_key or not api_base:
        raise ValueError("api_key and api_base are required")

    # Get list of conversation files
    conv_path = Path(conversations_dir)
    if not conv_path.exists():
        raise FileNotFoundError(f"Conversations directory not found: {conversations_dir}")

    conversation_files = list(conv_path.glob('*.json'))
    if not conversation_files:
        raise FileNotFoundError(f"No conversation files found in {conversations_dir}")

    # Get available models if model_name not specified
    available_models = None
    if model_name is None:
        available_models = get_available_models(api_key, api_base)
        if not available_models:
            raise ValueError("Could not determine available models")

    # Initialize session tracking
    session_start = datetime.now()
    end_time = time.time() + duration_seconds

    all_stats: List[ConversationStats] = []
    models_used = set()

    print(f"Starting user session simulation for {duration_seconds} seconds...")
    print(f"Available conversations: {len(conversation_files)}")
    if model_name:
        print(f"Using model: {model_name}")
    else:
        print(f"Randomly selecting from {len(available_models)} available models")
    print("-" * 60)

    iteration = 0
    while time.time() < end_time:
        iteration += 1

        # Randomly select conversation
        conversation_file = random.choice(conversation_files)

        # Randomly select model if not specified
        selected_model = model_name if model_name else random.choice(available_models)
        models_used.add(selected_model)

        # Randomly select temperature within range
        temperature = random.uniform(*temperature_range)

        # Randomly decide whether to limit messages (20% chance of limiting)
        max_messages = None
        if random.random() < 0.2:
            max_messages = random.randint(1, 3)

        print(f"[{iteration}] Running: {conversation_file.name} | Model: {selected_model} | Temp: {temperature:.2f}", end=" ")

        try:
            # Run the stress test
            stats = run_conversation_stress_test(
                conversation_file_path=str(conversation_file),
                model_name=selected_model,
                temperature=temperature,
                max_tokens=None,  # Always unlimited
                max_messages=max_messages,
                api_key=api_key,
                api_base=api_base
            )

            all_stats.append(stats)

            if stats.error:
                print(f"✗ Error: {stats.error[:50]}...")
            else:
                print(f"✓ {stats.total_messages_sent} msgs, {stats.average_latency_seconds:.2f}s avg")

        except Exception as e:
            print(f"✗ Exception: {str(e)[:50]}...")
            # Create a failed stats entry
            failed_stats = ConversationStats(conversation_id=conversation_file.stem)
            failed_stats.error = str(e)
            all_stats.append(failed_stats)

    session_end = datetime.now()

    # Aggregate statistics
    successful = [s for s in all_stats if not s.error]
    failed = [s for s in all_stats if s.error]

    total_messages = sum(s.total_messages_sent for s in successful)
    total_tokens = sum(s.total_tokens for s in successful)
    total_tokens_input = sum(s.total_tokens_input for s in successful)
    total_tokens_output = sum(s.total_tokens_output for s in successful)

    # Calculate average latency across all successful conversations
    all_latencies = []
    for s in successful:
        all_latencies.extend(s.message_latencies)

    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    min_latency = min(all_latencies) if all_latencies else 0.0
    max_latency = max(all_latencies) if all_latencies else 0.0

    duration = duration_seconds
    conversations_per_minute = (len(all_stats) / duration) * 60

    return UserSessionStats(
        session_start=session_start,
        session_end=session_end,
        duration_seconds=duration,
        total_conversations=len(all_stats),
        successful_conversations=len(successful),
        failed_conversations=len(failed),
        total_messages_sent=total_messages,
        total_tokens=total_tokens,
        total_tokens_input=total_tokens_input,
        total_tokens_output=total_tokens_output,
        average_latency_seconds=avg_latency,
        min_latency_seconds=min_latency,
        max_latency_seconds=max_latency,
        conversations_per_minute=conversations_per_minute,
        models_used=sorted(list(models_used)),
        conversation_stats=all_stats
    )


def print_session_report(stats: UserSessionStats) -> None:
    """
    Print a formatted report of the user session statistics.

    Args:
        stats: UserSessionStats object to report
    """
    print("\n" + "=" * 60)
    print("USER SESSION SUMMARY REPORT")
    print("=" * 60)
    print(f"Session Duration: {stats.duration_seconds}s ({stats.duration_seconds/60:.1f} minutes)")
    print(f"Start Time: {stats.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {stats.session_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("CONVERSATION STATISTICS")
    print("-" * 60)
    print(f"Total Conversations: {stats.total_conversations}")
    print(f"  Successful: {stats.successful_conversations} ({stats.successful_conversations/stats.total_conversations*100:.1f}%)")
    print(f"  Failed: {stats.failed_conversations} ({stats.failed_conversations/stats.total_conversations*100:.1f}%)")
    print(f"  Rate: {stats.conversations_per_minute:.2f} conversations/minute")
    print()

    print("MESSAGE & TOKEN STATISTICS")
    print("-" * 60)
    print(f"Total Messages Sent: {stats.total_messages_sent}")
    print(f"Total Tokens: {stats.total_tokens:,}")
    print(f"  Input Tokens: {stats.total_tokens_input:,}")
    print(f"  Output Tokens: {stats.total_tokens_output:,}")
    print()

    print("LATENCY STATISTICS")
    print("-" * 60)
    print(f"Average Latency: {stats.average_latency_seconds:.3f}s")
    print(f"Min Latency: {stats.min_latency_seconds:.3f}s")
    print(f"Max Latency: {stats.max_latency_seconds:.3f}s")
    print()

    print("MODELS USED")
    print("-" * 60)
    for model in stats.models_used:
        model_conversations = [s for s in stats.conversation_stats if hasattr(s, 'conversation_id')]
        # Count how many times this model was used (approximate, as we don't track model per conversation)
        print(f"  - {model}")
    print()

    if stats.failed_conversations > 0:
        print("ERRORS")
        print("-" * 60)
        error_counts = {}
        for s in stats.conversation_stats:
            if s.error:
                # Get first 80 chars of error
                error_key = s.error[:80]
                error_counts[error_key] = error_counts.get(error_key, 0) + 1

        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  [{count}x] {error}...")

    print("=" * 60)
