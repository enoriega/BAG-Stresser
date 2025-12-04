"""
Find the maximum number of concurrent connections a server can handle.

This script uses binary search to find the critical point where the server
starts failing to handle connections. It creates worker processes that each
send the first message from random conversations and monitors for errors.

The binary search narrows down the exact capacity by testing different
connection counts until it finds the maximum number that succeeds.
"""
import os
import sys
import json
import argparse
import random
import multiprocessing
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()


@dataclass
class ConnectionAttempt:
    """Details about a single connection attempt."""
    worker_id: int
    conversation_file: str
    model_name: str
    success: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    latency_seconds: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CapacityTestResult:
    """Results from testing a specific number of connections."""
    num_connections: int
    successful_connections: int
    failed_connections: int
    total_time_seconds: float
    attempts: List[ConnectionAttempt] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.successful_connections + self.failed_connections
        return (self.successful_connections / total * 100) if total > 0 else 0.0

    @property
    def is_success(self) -> bool:
        """Determine if this test is considered successful (no errors)."""
        return self.failed_connections == 0


@dataclass
class BinarySearchResult:
    """Final results from the binary search."""
    max_successful_connections: int
    min_failed_connections: Optional[int]
    total_tests: int
    test_results: List[CapacityTestResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0


def get_available_models(api_key: str, api_base: str, filter_file: str = "model_filter.txt") -> List[str]:
    """
    Fetch available models from the API endpoint and apply filtering.

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

        # Apply filter if file exists
        if Path(filter_file).exists():
            filtered_models = set()
            with open(filter_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        filtered_models.add(line)

            original_count = len(all_models)
            all_models = [m for m in all_models if m not in filtered_models]

            if len(all_models) < original_count:
                print(f"Filtered out {original_count - len(all_models)} model(s) based on {filter_file}")

        return all_models
    except Exception as e:
        print(f"Warning: Could not fetch models from API: {e}")
        return []


def load_all_conversations(conversations_dir: str) -> Dict[str, dict]:
    """
    Preload all conversations from directory into memory.

    Args:
        conversations_dir: Directory containing conversation JSON files

    Returns:
        Dictionary mapping filename to conversation data
    """
    conv_path = Path(conversations_dir)
    conversation_files = list(conv_path.glob('*.json'))

    conversations = {}
    print(f"Preloading {len(conversation_files)} conversations into memory...")

    for conv_file in conversation_files:
        with open(conv_file, 'r') as f:
            conversations[conv_file.name] = json.load(f)

    print(f"✓ Loaded {len(conversations)} conversations")
    return conversations


def load_conversation(file_path: str) -> dict:
    """
    Load conversation from JSON file.

    Args:
        file_path: Path to the conversation JSON file

    Returns:
        Dictionary containing conversation data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def send_first_message(
    conversation_data: dict,
    conversation_filename: str,
    model_name: str,
    temperature: float,
    api_key: str,
    api_base: str,
    worker_id: int
) -> ConnectionAttempt:
    """
    Send only the first message from a conversation.

    Args:
        conversation_data: Preloaded conversation data
        conversation_filename: Name of the conversation file
        model_name: Model to use
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL
        worker_id: Worker process ID

    Returns:
        ConnectionAttempt with results
    """
    try:
        # Extract messages from preloaded conversation
        messages = conversation_data.get('messages', [])

        # Extract first user message
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        if not user_messages:
            return ConnectionAttempt(
                worker_id=worker_id,
                conversation_file=conversation_filename,
                model_name=model_name,
                success=False,
                error="No user messages found in conversation"
            )

        first_message = user_messages[0]['content']

        # Create LLM client
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=api_base
        )

        # Send message and measure latency
        start_time = time.time()
        response = llm.invoke([HumanMessage(content=first_message)])
        end_time = time.time()

        latency = end_time - start_time

        return ConnectionAttempt(
            worker_id=worker_id,
            conversation_file=conversation_filename,
            model_name=model_name,
            success=True,
            latency_seconds=latency
        )

    except Exception as e:
        return ConnectionAttempt(
            worker_id=worker_id,
            conversation_file=conversation_filename,
            model_name=model_name,
            success=False,
            error=str(e),
            error_type=type(e).__name__
        )


def worker_process(
    worker_id: int,
    conversations: Dict[str, dict],
    available_models: List[str],
    temperature: float,
    api_key: str,
    api_base: str
) -> ConnectionAttempt:
    """
    Worker process entry point - sends first message from a random conversation.

    Args:
        worker_id: Unique worker ID
        conversations: Preloaded conversations dictionary (filename -> data)
        available_models: List of available models to randomly select from
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL

    Returns:
        ConnectionAttempt with results
    """
    # Select random conversation from preloaded data
    if not conversations:
        return ConnectionAttempt(
            worker_id=worker_id,
            conversation_file="N/A",
            model_name="N/A",
            success=False,
            error="No conversations provided"
        )

    if not available_models:
        return ConnectionAttempt(
            worker_id=worker_id,
            conversation_file="N/A",
            model_name="N/A",
            success=False,
            error="No models available"
        )

    # Randomly select model for this worker
    model_name = random.choice(available_models)

    conversation_filename = random.choice(list(conversations.keys()))
    conversation_data = conversations[conversation_filename]

    return send_first_message(
        conversation_data=conversation_data,
        conversation_filename=conversation_filename,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        api_base=api_base,
        worker_id=worker_id
    )


def test_capacity(
    num_connections: int,
    conversations: Dict[str, dict],
    available_models: List[str],
    temperature: float,
    api_key: str,
    api_base: str
) -> CapacityTestResult:
    """
    Test server capacity with a specific number of concurrent connections.

    Args:
        num_connections: Number of concurrent connections to attempt
        conversations: Preloaded conversations dictionary
        available_models: List of available models to randomly select from
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL

    Returns:
        CapacityTestResult with test outcomes
    """
    print(f"\n{'='*70}")
    print(f"Testing {num_connections} concurrent connections...")
    print(f"{'='*70}")

    start_time = time.time()

    # Create worker pool and launch all workers simultaneously
    with multiprocessing.Pool(processes=num_connections) as pool:
        worker_args = [
            (
                worker_id,
                conversations,
                available_models,
                temperature,
                api_key,
                api_base
            )
            for worker_id in range(1, num_connections + 1)
        ]

        # Run all workers in parallel
        attempts = pool.starmap(worker_process, worker_args)

    end_time = time.time()
    total_time = end_time - start_time

    # Count successes and failures
    successful = sum(1 for a in attempts if a.success)
    failed = sum(1 for a in attempts if not a.success)

    result = CapacityTestResult(
        num_connections=num_connections,
        successful_connections=successful,
        failed_connections=failed,
        total_time_seconds=total_time,
        attempts=attempts
    )

    print(f"Result: {successful}/{num_connections} succeeded ({result.success_rate:.1f}%)")
    print(f"Time: {total_time:.2f}s")

    if failed > 0:
        print(f"Failures: {failed}")
        # Show first few errors
        errors = [a for a in attempts if not a.success]
        for error in errors[:3]:
            print(f"  [{error.error_type}] {error.error}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more errors")

    return result


def binary_search_capacity(
    max_connections: int,
    conversations: Dict[str, dict],
    available_models: List[str],
    temperature: float,
    api_key: str,
    api_base: str,
    recovery_delay: float = 15.0
) -> BinarySearchResult:
    """
    Use binary search to find maximum number of successful connections.

    Args:
        max_connections: Upper bound for search
        conversations: Preloaded conversations dictionary
        available_models: List of available models to randomly select from
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL
        recovery_delay: Seconds to wait between iterations for server recovery (default: 15.0)

    Returns:
        BinarySearchResult with final results
    """
    print(f"\n{'='*70}")
    print(f"BINARY SEARCH FOR MAXIMUM CAPACITY")
    print(f"{'='*70}")
    print(f"Search range: 1 to {max_connections} connections")
    print(f"Models: {len(available_models)} available (randomly selected per worker)")
    print(f"Temperature: {temperature}")
    print(f"Recovery delay: {recovery_delay}s between iterations")
    print(f"{'='*70}")

    search_start = time.time()

    left = 1
    right = max_connections
    max_successful = 0
    min_failed = None
    test_results = []

    while left <= right:
        mid = (left + right) // 2

        # Test at this capacity
        result = test_capacity(
            num_connections=mid,
            conversations=conversations,
            available_models=available_models,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base
        )
        test_results.append(result)

        if result.is_success:
            # Success - try higher
            max_successful = mid
            left = mid + 1
            print(f"✓ Success at {mid} - searching higher...")
        else:
            # Failure - try lower
            if min_failed is None or mid < min_failed:
                min_failed = mid
            right = mid - 1
            print(f"✗ Failed at {mid} - searching lower...")

        # Add recovery delay between iterations (but not after the last one)
        if left <= right and recovery_delay > 0:
            print(f"Waiting {recovery_delay}s for server recovery...")
            time.sleep(recovery_delay)

    search_end = time.time()
    total_duration = search_end - search_start

    return BinarySearchResult(
        max_successful_connections=max_successful,
        min_failed_connections=min_failed,
        total_tests=len(test_results),
        test_results=test_results,
        total_duration_seconds=total_duration
    )


def print_detailed_statistics(result: BinarySearchResult):
    """
    Print detailed statistics from the binary search.

    Args:
        result: BinarySearchResult to report
    """
    print(f"\n{'='*70}")
    print(f"CAPACITY TEST RESULTS")
    print(f"{'='*70}")
    print(f"Maximum Successful Connections: {result.max_successful_connections}")
    if result.min_failed_connections:
        print(f"Minimum Failed Connections: {result.min_failed_connections}")
        print(f"Critical Point: Between {result.max_successful_connections} and {result.min_failed_connections}")
    print(f"Total Tests Performed: {result.total_tests}")
    print(f"Total Duration: {result.total_duration_seconds:.2f}s ({result.total_duration_seconds/60:.1f} minutes)")
    print()

    # Calculate aggregate statistics
    total_attempts = sum(len(t.attempts) for t in result.test_results)
    total_successful = sum(t.successful_connections for t in result.test_results)
    total_failed = sum(t.failed_connections for t in result.test_results)

    print(f"AGGREGATE STATISTICS")
    print(f"{'-'*70}")
    print(f"Total Connection Attempts: {total_attempts}")
    print(f"  Successful: {total_successful} ({total_successful/total_attempts*100:.1f}%)")
    print(f"  Failed: {total_failed} ({total_failed/total_attempts*100:.1f}%)")
    print()

    # Latency statistics (from successful connections)
    all_latencies = []
    for test_result in result.test_results:
        for attempt in test_result.attempts:
            if attempt.success and attempt.latency_seconds is not None:
                all_latencies.append(attempt.latency_seconds)

    if all_latencies:
        avg_latency = sum(all_latencies) / len(all_latencies)
        min_latency = min(all_latencies)
        max_latency = max(all_latencies)

        print(f"LATENCY STATISTICS (Successful Connections)")
        print(f"{'-'*70}")
        print(f"Average: {avg_latency:.3f}s")
        print(f"Min: {min_latency:.3f}s")
        print(f"Max: {max_latency:.3f}s")
        print()

    # Test progression
    print(f"TEST PROGRESSION")
    print(f"{'-'*70}")
    print(f"{'Test':<6} {'Connections':<12} {'Success':<10} {'Failed':<8} {'Rate':<12} {'Time':<8}")
    print(f"{'-'*70}")
    for i, test_result in enumerate(result.test_results, 1):
        status = "✓" if test_result.is_success else "✗"
        print(f"{i:<6} {test_result.num_connections:<12} {test_result.successful_connections:<10} "
              f"{test_result.failed_connections:<8} {test_result.success_rate:>5.1f}%{' ':<5} "
              f"{test_result.total_time_seconds:>6.2f}s")
    print()

    # Error analysis (if any)
    all_errors = []
    for test_result in result.test_results:
        for attempt in test_result.attempts:
            if not attempt.success and attempt.error:
                all_errors.append(attempt)

    if all_errors:
        print(f"ERROR ANALYSIS")
        print(f"{'-'*70}")

        # Group by error type
        error_types = {}
        for attempt in all_errors:
            error_type = attempt.error_type or "Unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1

        print("Error Types:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  [{count}x] {error_type}")

        print("\nSample Errors (first 5):")
        for i, attempt in enumerate(all_errors[:5], 1):
            print(f"  [{i}] Worker {attempt.worker_id}: {attempt.error_type} - {attempt.error[:80]}")

        if len(all_errors) > 5:
            print(f"  ... and {len(all_errors) - 5} more errors")
        print()

    print(f"{'='*70}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Find maximum concurrent connections using binary search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test up to 100 connections
  python capacity_tester.py --max 100

  # Test up to 500 connections with specific model
  python capacity_tester.py --max 500 --model gpt-3.5-turbo

  # Test with custom temperature and 30s recovery delay
  python capacity_tester.py --max 200 --temperature 0.7 --recovery-delay 30

  # Test with no recovery delay between iterations
  python capacity_tester.py --max 100 --recovery-delay 0
        """
    )

    parser.add_argument(
        '--max', '-m',
        type=int,
        required=True,
        help='Maximum number of connections to test (upper bound for binary search)'
    )
    parser.add_argument(
        '--conversations-dir', '-c',
        type=str,
        default='conversations',
        help='Directory containing conversation JSON files (default: conversations)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Specific model to use for all workers (default: workers randomly select from all available models, or MODEL_NAME env var)'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.7,
        help='Temperature parameter (default: 0.7)'
    )
    parser.add_argument(
        '--recovery-delay', '-r',
        type=float,
        default=15.0,
        help='Seconds to wait between iterations for server recovery (default: 15.0, use 0 to disable)'
    )

    args = parser.parse_args()

    # Get API configuration
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE')
    model_name = args.model or os.getenv('MODEL_NAME')

    if not api_key or not api_base:
        print("Error: OPENAI_API_KEY and OPENAI_API_BASE must be set in .env file")
        sys.exit(1)

    # Fetch available models for workers to randomly select from
    print("Fetching available models...")
    available_models = get_available_models(api_key, api_base, filter_file="model_filter.txt")

    if not available_models:
        print("Error: Could not fetch available models from API")
        sys.exit(1)

    # If specific model provided, use only that model
    if model_name:
        if model_name in available_models:
            available_models = [model_name]
            print(f"Using specified model: {model_name}")
        else:
            print(f"Warning: Specified model '{model_name}' not found in available models")
            print(f"Using all {len(available_models)} available models for random selection")
    else:
        print(f"No model specified - workers will randomly select from {len(available_models)} available models")

    # Validate arguments
    if args.max < 1:
        print("Error: Maximum connections must be at least 1")
        sys.exit(1)

    if args.temperature < 0 or args.temperature > 2.0:
        print("Error: Temperature must be between 0 and 2.0")
        sys.exit(1)

    if args.recovery_delay < 0:
        print("Error: Recovery delay must be non-negative")
        sys.exit(1)

    # Verify conversations directory exists
    conv_path = Path(args.conversations_dir)
    if not conv_path.exists():
        print(f"Error: Conversations directory not found: {args.conversations_dir}")
        sys.exit(1)

    conversation_files = list(conv_path.glob('*.json'))
    if not conversation_files:
        print(f"Error: No conversation files found in {args.conversations_dir}")
        sys.exit(1)

    print(f"Found {len(conversation_files)} conversation files")

    # Preload all conversations into memory
    conversations = load_all_conversations(args.conversations_dir)
    print()

    # Run binary search
    result = binary_search_capacity(
        max_connections=args.max,
        conversations=conversations,
        available_models=available_models,
        temperature=args.temperature,
        api_key=api_key,
        api_base=api_base,
        recovery_delay=args.recovery_delay
    )

    # Print detailed statistics
    print_detailed_statistics(result)

    # Summary
    print(f"\nSUMMARY: Server can handle up to {result.max_successful_connections} concurrent connections")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
