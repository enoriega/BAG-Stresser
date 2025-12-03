import json
import time
import asyncio
import random
import re
import traceback
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Thread pool for blocking I/O operations
_io_executor = ThreadPoolExecutor(max_workers=4)

# Cache for LLM clients (avoid repeated initialization)
_llm_client_cache: Dict[tuple, ChatOpenAI] = {}


def clear_llm_cache():
    """Clear the LLM client cache. Useful for testing."""
    global _llm_client_cache
    _llm_client_cache.clear()


def extract_traceback_info(exc: Exception) -> tuple[Optional[str], Optional[int], str]:
    """
    Extract script name, line number, and full traceback from an exception.

    Args:
        exc: The exception to extract information from

    Returns:
        Tuple of (script_name, line_number, full_traceback)
    """
    tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # Extract the last traceback entry (where the error actually occurred)
    tb_lines = traceback.extract_tb(exc.__traceback__)
    if tb_lines:
        last_frame = tb_lines[-1]
        script_name = last_frame.filename
        line_number = last_frame.lineno
    else:
        script_name = None
        line_number = None

    return script_name, line_number, tb_str


@dataclass
class ErrorDetails:
    """Detailed information about an error that occurred during a conversation."""
    model_name: str
    conversation_file: str
    step: int  # Which message/step in the conversation failed
    exception_type: str
    error_message: str
    server_error: Optional[str] = None  # Server-specific error if available
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    script_name: Optional[str] = None  # Script file where error occurred
    line_number: Optional[int] = None  # Line number where error occurred
    traceback: Optional[str] = None  # Full traceback for debugging


@dataclass
class ConversationStats:
    """Statistics collected during a conversation stress test."""
    conversation_id: str
    model_name: str = ""
    conversation_file: str = ""
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
    error: Optional[str] = None  # Brief error summary for backwards compatibility
    error_details: Optional[ErrorDetails] = None  # Detailed error information


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


async def load_conversation_async(file_path: str) -> dict:
    """
    Load conversation from JSON file asynchronously using thread pool.

    Args:
        file_path: Path to the conversation JSON file

    Returns:
        Dictionary containing conversation data
    """
    loop = asyncio.get_event_loop()

    def _load_file():
        with open(file_path, 'r') as f:
            return json.load(f)

    return await loop.run_in_executor(_io_executor, _load_file)


def get_or_create_llm_client(
    model_name: str,
    temperature: float,
    max_tokens: Optional[int],
    api_key: Optional[str],
    api_base: Optional[str]
) -> ChatOpenAI:
    """
    Get cached LLM client or create new one if not exists.

    This avoids the overhead of creating a new ChatOpenAI instance
    for every conversation, significantly improving performance.

    Args:
        model_name: Model name
        temperature: Temperature setting
        max_tokens: Max tokens (None for unlimited)
        api_key: API key
        api_base: API base URL

    Returns:
        Cached or new ChatOpenAI instance
    """
    # Create cache key from parameters
    cache_key = (model_name, temperature, max_tokens, api_key, api_base)

    if cache_key not in _llm_client_cache:
        llm_kwargs = {
            'model': model_name,
            'temperature': temperature,
        }
        if max_tokens is not None:
            llm_kwargs['max_tokens'] = max_tokens
        if api_key:
            llm_kwargs['api_key'] = api_key
        if api_base:
            llm_kwargs['base_url'] = api_base

        _llm_client_cache[cache_key] = ChatOpenAI(**llm_kwargs)

    return _llm_client_cache[cache_key]


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


async def run_conversation_stress_test(
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
    # Load conversation from JSON file asynchronously
    conversation_path = Path(conversation_file_path)
    if not conversation_path.exists():
        raise FileNotFoundError(f"Conversation file not found: {conversation_file_path}")

    # Use async file loading to avoid blocking
    conversation_data = await load_conversation_async(conversation_file_path)

    conversation_id = conversation_data.get('conversation_id', conversation_path.stem)
    messages = conversation_data.get('messages', [])

    # Extract only user messages from the conversation
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    simulated_responses = [msg for msg in messages if msg['role'] == 'assistant']

    # Initialize statistics
    stats = ConversationStats(
        conversation_id=conversation_id,
        model_name=model_name,
        conversation_file=conversation_path.name
    )

    # Determine how many messages to send
    messages_to_send = len(user_messages)
    if max_messages is not None:
        messages_to_send = min(messages_to_send, max_messages)

    if messages_to_send == 0:
        return stats

    # Get or create cached LLM client (avoids repeated initialization overhead)
    try:
        llm = get_or_create_llm_client(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=api_base
        )

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
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                # Extract server error details if available
                server_error = None
                error_message = str(e)

                # Try to extract more detailed error information from the exception
                if hasattr(e, 'response'):
                    # For HTTP errors with response objects
                    try:
                        if hasattr(e.response, 'json'):
                            error_data = e.response.json()
                            server_error = json.dumps(error_data)
                        elif hasattr(e.response, 'text'):
                            server_error = e.response.text
                    except:
                        pass

                # Extract traceback information
                script_name, line_number, tb_str = extract_traceback_info(e)

                # Create detailed error information
                stats.error_details = ErrorDetails(
                    model_name=model_name,
                    conversation_file=conversation_path.name,
                    step=idx + 1,
                    exception_type=type(e).__name__,
                    error_message=error_message,
                    server_error=server_error,
                    script_name=script_name,
                    line_number=line_number,
                    traceback=tb_str
                )
                stats.error = f"Error at step {idx + 1}: {type(e).__name__}: {error_message}"
                break

        # Calculate average latency
        if stats.total_ai_responses > 0:
            stats.average_latency_seconds = stats.total_latency_seconds / stats.total_ai_responses

        # Reset min latency if no messages were sent
        if stats.min_latency_seconds == float('inf'):
            stats.min_latency_seconds = 0.0

    except Exception as e:
        # Capture detailed error information for initialization failures
        server_error = None
        error_message = str(e)

        if hasattr(e, 'response'):
            try:
                if hasattr(e.response, 'json'):
                    error_data = e.response.json()
                    server_error = json.dumps(error_data)
                elif hasattr(e.response, 'text'):
                    server_error = e.response.text
            except:
                pass

        # Extract traceback information
        script_name, line_number, tb_str = extract_traceback_info(e)

        stats.error_details = ErrorDetails(
            model_name=model_name,
            conversation_file=conversation_path.name,
            step=0,  # 0 indicates initialization failure
            exception_type=type(e).__name__,
            error_message=error_message,
            server_error=server_error,
            script_name=script_name,
            line_number=line_number,
            traceback=tb_str
        )
        stats.error = f"Error initializing LLM: {type(e).__name__}: {error_message}"

    return stats


def load_model_filter(filter_file: str = Path(__file__) / "model_filter.txt") -> set[str]:
    """
    Load model names to filter out from a file.

    The file should contain one model name per line. Lines starting with # are ignored.
    Empty lines are also ignored.

    Args:
        filter_file: Path to the filter file (default: model_filter.txt)

    Returns:
        Set of model names to exclude
    """
    if not Path(filter_file).exists():
        return set()

    filtered_models = set()
    try:
        with open(filter_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    filtered_models.add(line)
    except Exception as e:
        print(f"Warning: Could not read model filter file {filter_file}: {e}")

    return filtered_models


def filter_models(models: List[str], filter_file: str = Path(__file__) / "model_filter.txt") -> List[str]:
    """
    Filter out models based on the filter file.

    Args:
        models: List of model names
        filter_file: Path to the filter file

    Returns:
        Filtered list of model names
    """
    filtered = load_model_filter(filter_file)
    if not filtered:
        return models

    original_count = len(models)
    filtered_models = [m for m in models if m not in filtered]

    if len(filtered_models) < original_count:
        print(f"Filtered out {original_count - len(filtered_models)} model(s) based on {filter_file}")

    return filtered_models


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


async def get_available_models_async(api_key: str, api_base: str, filter_file: str = Path(__file__) / "model_filter.txt") -> List[str]:
    """
    Fetch available models from the API endpoint asynchronously and apply filtering.

    Args:
        api_key: API key for authentication
        api_base: Base URL for the API
        filter_file: Path to file containing model names to exclude

    Returns:
        List of available model IDs (after filtering)
    """
    loop = asyncio.get_event_loop()

    def _fetch_models():
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=api_base)
            models = client.models.list()
            all_models = [model.id for model in models.data]
            # Apply filter
            return filter_models(all_models, filter_file)
        except Exception as e:
            # Fallback to some common models if API call fails
            print(f"Warning: Could not fetch models from API: {e}")
            return []

    return await loop.run_in_executor(_io_executor, _fetch_models)


def get_available_models(api_key: str, api_base: str, filter_file: str = Path(__file__) / "model_filter.txt") -> List[str]:
    """
    Fetch available models from the API endpoint (synchronous wrapper for compatibility) and apply filtering.

    Args:
        api_key: API key for authentication
        api_base: Base URL for the API
        filter_file: Path to file containing model names to exclude

    Returns:
        List of available model IDs (after filtering)
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=api_base)
        models = client.models.list()
        all_models = [model.id for model in models.data]
        # Apply filter
        return filter_models(all_models, filter_file)
    except Exception as e:
        # Fallback to some common models if API call fails
        print(f"Warning: Could not fetch models from API: {e}")
        return []


async def simulate_user_session(
    conversations_dir: str = 'conversations',
    duration_seconds: int = 60,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature_range: tuple[float, float] = (0.5, 1.0),
    concurrency: int = 1,
    model_filter_file: str = Path(__file__) / "model_filter.txt"
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
        concurrency: Number of conversations to run concurrently (default: 1)
        model_filter_file: Path to file containing model names to exclude (default: model_filter.txt)

    Returns:
        UserSessionStats object with aggregated results
    """
    if not api_key or not api_base:
        raise ValueError("api_key and api_base are required")

    # Get list of conversation files (run in thread pool to avoid blocking)
    loop = asyncio.get_event_loop()

    def _scan_conversations():
        conv_path = Path(conversations_dir)
        if not conv_path.exists():
            raise FileNotFoundError(f"Conversations directory not found: {conversations_dir}")
        files = list(conv_path.glob('*.json'))
        if not files:
            raise FileNotFoundError(f"No conversation files found in {conversations_dir}")
        return files

    conversation_files = await loop.run_in_executor(_io_executor, _scan_conversations)

    # Get available models asynchronously if model_name not specified
    available_models = None
    if model_name is None:
        available_models = await get_available_models_async(api_key, api_base, model_filter_file)
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
    print(f"Concurrency level: {concurrency} concurrent conversation(s)")
    print("-" * 60)

    if concurrency == 1:
        # Sequential execution (original behavior)
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
                stats = await run_conversation_stress_test(
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
                print(f"✗ Exception: {type(e).__name__}: {str(e)[:50]}...")
                # Create a failed stats entry with detailed error information
                server_error = None
                error_message = str(e)

                if hasattr(e, 'response'):
                    try:
                        if hasattr(e.response, 'json'):
                            error_data = e.response.json()
                            server_error = json.dumps(error_data)
                        elif hasattr(e.response, 'text'):
                            server_error = e.response.text
                    except:
                        pass

                # Extract traceback information
                script_name, line_number, tb_str = extract_traceback_info(e)

                failed_stats = ConversationStats(
                    conversation_id=conversation_file.stem,
                    model_name=selected_model,
                    conversation_file=conversation_file.name
                )
                failed_stats.error_details = ErrorDetails(
                    model_name=selected_model,
                    conversation_file=conversation_file.name,
                    step=-1,  # -1 indicates error occurred before conversation started
                    exception_type=type(e).__name__,
                    error_message=error_message,
                    server_error=server_error,
                    script_name=script_name,
                    line_number=line_number,
                    traceback=tb_str
                )
                failed_stats.error = f"{type(e).__name__}: {error_message}"
                all_stats.append(failed_stats)

    else:
        # Concurrent execution
        async def run_single_conversation(conv_id: int):
            """Run a single conversation as part of concurrent pool."""
            conversation_file = random.choice(conversation_files)
            selected_model = model_name if model_name else random.choice(available_models)
            temperature = random.uniform(*temperature_range)
            max_messages = None if random.random() >= 0.2 else random.randint(1, 3)

            models_used.add(selected_model)

            print(f"[{conv_id}] Running: {conversation_file.name} | Model: {selected_model} | Temp: {temperature:.2f}", end=" ")

            try:
                stats = await run_conversation_stress_test(
                    conversation_file_path=str(conversation_file),
                    model_name=selected_model,
                    temperature=temperature,
                    max_tokens=None,
                    max_messages=max_messages,
                    api_key=api_key,
                    api_base=api_base
                )

                if stats.error:
                    print(f"✗ Error: {stats.error[:50]}...")
                else:
                    print(f"✓ {stats.total_messages_sent} msgs, {stats.average_latency_seconds:.2f}s avg")

                return stats

            except Exception as e:
                print(f"✗ Exception: {type(e).__name__}: {str(e)[:50]}...")
                # Create a failed stats entry with detailed error information
                server_error = None
                error_message = str(e)

                if hasattr(e, 'response'):
                    try:
                        if hasattr(e.response, 'json'):
                            error_data = e.response.json()
                            server_error = json.dumps(error_data)
                        elif hasattr(e.response, 'text'):
                            server_error = e.response.text
                    except:
                        pass

                # Extract traceback information
                script_name, line_number, tb_str = extract_traceback_info(e)

                failed_stats = ConversationStats(
                    conversation_id=conversation_file.stem,
                    model_name=selected_model,
                    conversation_file=conversation_file.name
                )
                failed_stats.error_details = ErrorDetails(
                    model_name=selected_model,
                    conversation_file=conversation_file.name,
                    step=-1,  # -1 indicates error occurred before conversation started
                    exception_type=type(e).__name__,
                    error_message=error_message,
                    server_error=server_error,
                    script_name=script_name,
                    line_number=line_number,
                    traceback=tb_str
                )
                failed_stats.error = f"{type(e).__name__}: {error_message}"
                return failed_stats

        # Manage concurrent task pool
        pending_tasks = set()
        conversation_counter = 0

        # Initial burst to fill concurrency slots
        for _ in range(concurrency):
            if time.time() >= end_time:
                break
            conversation_counter += 1
            task = asyncio.create_task(run_single_conversation(conversation_counter))
            pending_tasks.add(task)

        # Main loop: as tasks complete, spawn new ones
        while pending_tasks and time.time() < end_time:
            # Wait for at least one task to complete
            done, pending_tasks = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=max(0.1, end_time - time.time())
            )

            # Collect results from completed tasks
            for task in done:
                try:
                    stats = await task
                    all_stats.append(stats)
                except Exception as e:
                    print(f"✗ Task exception: {str(e)[:50]}...")

            # Spawn new tasks to maintain concurrency level
            while len(pending_tasks) < concurrency and time.time() < end_time:
                conversation_counter += 1
                task = asyncio.create_task(run_single_conversation(conversation_counter))
                pending_tasks.add(task)

        # Wait for any remaining tasks to complete
        if pending_tasks:
            done, _ = await asyncio.wait(pending_tasks, timeout=10.0)
            for task in done:
                try:
                    stats = await task
                    all_stats.append(stats)
                except Exception as e:
                    print(f"✗ Task exception: {str(e)[:50]}...")

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


def print_error_timeline(stats: UserSessionStats, interval_seconds: int = 5) -> None:
    """
    Print a timeline visualization of when errors occurred during the session.

    Args:
        stats: UserSessionStats object containing error details
        interval_seconds: Size of time intervals in seconds (default: 5)
    """
    if stats.failed_conversations == 0:
        return

    # Collect all errors with timestamps
    errors_with_time = []
    for conv_stat in stats.conversation_stats:
        if conv_stat.error_details:
            try:
                error_time = datetime.fromisoformat(conv_stat.error_details.timestamp)
                errors_with_time.append({
                    'time': error_time,
                    'type': conv_stat.error_details.exception_type,
                    'details': conv_stat.error_details
                })
            except:
                pass

    if not errors_with_time:
        return

    # Sort errors by time
    errors_with_time.sort(key=lambda x: x['time'])

    # Calculate session start and end times
    session_start = stats.session_start
    session_end = stats.session_end
    total_duration = (session_end - session_start).total_seconds()

    # Create time buckets
    num_intervals = int(total_duration / interval_seconds) + 1
    buckets = [[] for _ in range(num_intervals)]

    # Assign errors to buckets
    for error in errors_with_time:
        elapsed = (error['time'] - session_start).total_seconds()
        bucket_idx = int(elapsed / interval_seconds)
        if 0 <= bucket_idx < num_intervals:
            buckets[bucket_idx].append(error)

    # Find max errors in any bucket for scaling
    max_errors = max(len(bucket) for bucket in buckets) if buckets else 1
    bar_width = 20  # Width of the bar chart

    print("\nERROR TIMELINE")
    print("-" * 60)
    print(f"Session Duration: {total_duration:.1f} seconds ({session_start.strftime('%Y-%m-%d %H:%M:%S')} to {session_end.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"\nErrors over time ({interval_seconds}-second intervals):")

    # Print timeline
    for i, bucket in enumerate(buckets):
        start_sec = i * interval_seconds
        end_sec = min((i + 1) * interval_seconds, total_duration)
        num_errors = len(bucket)

        # Format time range
        time_range = f"{int(start_sec//60):02d}:{int(start_sec%60):02d}-{int(end_sec//60):02d}:{int(end_sec%60):02d}"

        # Create bar
        if num_errors > 0:
            bar_len = int((num_errors / max_errors) * bar_width)
            bar = "█" * bar_len
        else:
            bar = " " * 10

        # Count error types in this bucket
        error_type_counts = {}
        for error in bucket:
            error_type = error['type']
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

        # Format error type summary
        if error_type_counts:
            type_summary = ", ".join(f"{k}: {v}" for k, v in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True))
            error_label = "error" if num_errors == 1 else "errors"
            print(f"{time_range} [{bar:<{bar_width}}] {num_errors} {error_label:<7} ({type_summary})")
        else:
            print(f"{time_range} [{bar:<{bar_width}}] 0 errors")

    # Calculate statistics
    error_windows = sum(1 for bucket in buckets if len(bucket) > 0)
    error_free_windows = num_intervals - error_windows
    avg_errors_per_interval = len(errors_with_time) / num_intervals if num_intervals > 0 else 0
    peak_errors = max_errors
    peak_interval_idx = next(i for i, bucket in enumerate(buckets) if len(bucket) == peak_errors)
    peak_start = peak_interval_idx * interval_seconds
    peak_end = min((peak_interval_idx + 1) * interval_seconds, total_duration)

    print("\nError Rate Statistics:")
    print(f"  Peak Error Rate: {peak_errors} errors in {interval_seconds}-second window ({int(peak_start//60):02d}:{int(peak_start%60):02d}-{int(peak_end//60):02d}:{int(peak_end%60):02d})")
    print(f"  Average Errors per {interval_seconds}s: {avg_errors_per_interval:.1f} errors")
    print(f"  Total Error Windows: {error_windows} out of {num_intervals} intervals")
    print(f"  Error-Free Windows: {error_free_windows} out of {num_intervals} intervals")

    # Overall error type counts
    all_error_types = {}
    for error in errors_with_time:
        error_type = error['type']
        all_error_types[error_type] = all_error_types.get(error_type, 0) + 1

    print("\nMost Common Errors:")
    for error_type, count in sorted(all_error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  [{count}x] {error_type}")


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
        # Print error timeline first
        print_error_timeline(stats)

        print("\nERRORS")
        print("-" * 60)

        # Group errors by type and count them
        error_counts = {}
        detailed_errors = []

        for s in stats.conversation_stats:
            if s.error:
                if s.error_details:
                    # Collect detailed error information
                    detailed_errors.append(s.error_details)
                    # Group by exception type
                    error_key = s.error_details.exception_type
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1
                else:
                    # Fallback to old error string
                    error_key = s.error[:80]
                    error_counts[error_key] = error_counts.get(error_key, 0) + 1

        # Print error summary by type
        print("\nError Summary:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  [{count}x] {error_type}")

        # Print detailed error information
        print("\nDetailed Error Information:")
        print("-" * 60)
        for idx, error in enumerate(detailed_errors, 1):  # Show all detailed errors
            print(f"\n[Error {idx}]")
            print(f"  Model: {error.model_name}")
            print(f"  Conversation File: {error.conversation_file}")
            print(f"  Failed at Step: {error.step}" +
                  (" (initialization)" if error.step == 0 else
                   " (before start)" if error.step == -1 else ""))
            print(f"  Exception Type: {error.exception_type}")
            print(f"  Error Message: {error.error_message}")
            if error.script_name:
                print(f"  Script: {error.script_name}:{error.line_number}")
            if error.server_error:
                # Truncate server error if too long
                server_err_display = error.server_error
                print(f"  Server Error: {server_err_display}")
            print(f"  Timestamp: {error.timestamp}")
            if error.traceback:
                print(f"  Traceback:\n{error.traceback}")

    print("\n" + "=" * 60)
