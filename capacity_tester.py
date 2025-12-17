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
import argparse
import random
import asyncio
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from capacity_common import (
    ConnectionAttempt,
    CapacityTestResult,
    BinarySearchResult,
    get_available_models,
    load_all_conversations,
    send_first_message,
    save_json_output,
    print_detailed_statistics
)

# Load environment variables
load_dotenv()


async def worker_task(
    worker_id: int,
    conversations: Dict[str, dict],
    available_models: List[str],
    temperature: float,
    api_key: str,
    api_base: str,
    timeout: float = 60.0
) -> ConnectionAttempt:
    """
    Async worker task - sends first message from a random conversation.

    Args:
        worker_id: Unique worker ID
        conversations: Preloaded conversations dictionary (filename -> data)
        available_models: List of available models to randomly select from
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL
        timeout: Request timeout in seconds (default: 60.0)

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

    return await send_first_message(
        conversation_data=conversation_data,
        conversation_filename=conversation_filename,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        api_base=api_base,
        worker_id=worker_id,
        timeout=timeout
    )





def run_worker_process(
    worker_id: int,
    num_connections: int,
    conversations: Dict[str, dict],
    available_models: List[str],
    temperature: float,
    api_key: str,
    api_base: str,
    result_queue: mp.Queue,
    timeout: float = 60.0
) -> None:
    """
    Worker process that runs async tasks for a subset of connections.

    Args:
        worker_id: ID of this worker process
        num_connections: Number of connections this worker should handle
        conversations: Preloaded conversations dictionary
        available_models: List of available models to randomly select from
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL
        result_queue: Queue to send results back to main process
        timeout: Request timeout in seconds (default: 60.0)
    """
    async def run_async_tasks():
        # Create all worker tasks for this process
        tasks = [
            worker_task(
                worker_id=worker_id * 1000 + task_id,  # Unique ID across all processes
                conversations=conversations,
                available_models=available_models,
                temperature=temperature,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout
            )
            for task_id in range(num_connections)
        ]

        # Run all tasks concurrently
        attempts = await asyncio.gather(*tasks, return_exceptions=False)
        return attempts

    # Run the async tasks in this process
    attempts = asyncio.run(run_async_tasks())

    # Send results back to main process
    result_queue.put(attempts)


async def test_capacity(
    num_connections: int,
    conversations: Dict[str, dict],
    available_models: List[str],
    temperature: float,
    api_key: str,
    api_base: str,
    num_workers: int = 1,
    timeout: float = 60.0
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
        num_workers: Number of worker processes to use (default: 1)
        timeout: Request timeout in seconds (default: 60.0)

    Returns:
        CapacityTestResult with test outcomes
    """
    print(f"\n{'='*70}")
    print(f"Testing {num_connections} concurrent connections...")
    if num_workers > 1:
        print(f"Using {num_workers} worker processes")
    print(f"{'='*70}")

    start_time = time.time()

    if num_workers == 1:
        # Single-process mode: use async only
        tasks = [
            worker_task(
                worker_id=worker_id,
                conversations=conversations,
                available_models=available_models,
                temperature=temperature,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout
            )
            for worker_id in range(1, num_connections + 1)
        ]

        # Run all tasks concurrently and wait for all to complete
        attempts = await asyncio.gather(*tasks, return_exceptions=False)
    else:
        # Multi-process mode: distribute connections across worker processes
        connections_per_worker = num_connections // num_workers
        remaining_connections = num_connections % num_workers

        # Create queue for collecting results
        result_queue = mp.Queue()
        processes = []

        # Start worker processes
        for worker_id in range(num_workers):
            # Distribute remaining connections among first workers
            worker_connections = connections_per_worker + (1 if worker_id < remaining_connections else 0)

            if worker_connections > 0:
                process = mp.Process(
                    target=run_worker_process,
                    args=(
                        worker_id,
                        worker_connections,
                        conversations,
                        available_models,
                        temperature,
                        api_key,
                        api_base,
                        result_queue,
                        timeout
                    )
                )
                process.start()
                processes.append(process)

        # Wait for all processes to complete
        for process in processes:
            process.join()

        # Collect results from all workers
        attempts = []
        while not result_queue.empty():
            worker_attempts = result_queue.get()
            attempts.extend(worker_attempts)

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


async def single_test_capacity(
    num_connections: int,
    conversations: Dict[str, dict],
    available_models: List[str],
    temperature: float,
    api_key: str,
    api_base: str,
    num_workers: int = 1,
    timeout: float = 60.0
) -> BinarySearchResult:
    """
    Run a single capacity test without binary search (async).

    Args:
        num_connections: Number of concurrent connections to test
        conversations: Preloaded conversations dictionary
        available_models: List of available models to randomly select from
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL
        num_workers: Number of worker processes to use (default: 1)
        timeout: Request timeout in seconds (default: 60.0)

    Returns:
        BinarySearchResult with single test result
    """
    print(f"\n{'='*70}")
    print(f"SINGLE CAPACITY TEST")
    print(f"{'='*70}")
    print(f"Testing: {num_connections} connections")
    print(f"Models: {len(available_models)} available (randomly selected per worker)")
    print(f"Temperature: {temperature}")
    if num_workers > 1:
        print(f"Worker processes: {num_workers}")
    print(f"{'='*70}")

    test_start = time.time()

    # Run single test
    result = await test_capacity(
        num_connections=num_connections,
        conversations=conversations,
        available_models=available_models,
        temperature=temperature,
        api_key=api_key,
        api_base=api_base,
        num_workers=num_workers,
        timeout=timeout
    )

    test_end = time.time()
    total_duration = test_end - test_start

    # Wrap in BinarySearchResult format
    return BinarySearchResult(
        max_successful_connections=result.successful_connections,
        min_failed_connections=result.failed_connections if result.failed_connections > 0 else None,
        total_tests=1,
        test_results=[result],
        total_duration_seconds=total_duration
    )


async def binary_search_capacity(
    max_connections: int,
    conversations: Dict[str, dict],
    available_models: List[str],
    temperature: float,
    api_key: str,
    api_base: str,
    recovery_delay: float = 15.0,
    num_workers: int = 1,
    timeout: float = 60.0
) -> BinarySearchResult:
    """
    Use binary search to find maximum number of successful connections (async).

    Args:
        max_connections: Upper bound for search
        conversations: Preloaded conversations dictionary
        available_models: List of available models to randomly select from
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL
        recovery_delay: Seconds to wait between iterations for server recovery (default: 15.0)
        num_workers: Number of worker processes to use (default: 1)
        timeout: Request timeout in seconds (default: 60.0)

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
    if num_workers > 1:
        print(f"Worker processes: {num_workers}")
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
        result = await test_capacity(
            num_connections=mid,
            conversations=conversations,
            available_models=available_models,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            num_workers=num_workers,
            timeout=timeout
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
            await asyncio.sleep(recovery_delay)

    search_end = time.time()
    total_duration = search_end - search_start

    return BinarySearchResult(
        max_successful_connections=max_successful,
        min_failed_connections=min_failed,
        total_tests=len(test_results),
        test_results=test_results,
        total_duration_seconds=total_duration
    )


async def main():
    """Main async function."""
    parser = argparse.ArgumentParser(
        description='Find maximum concurrent connections using binary search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Binary search to find max capacity up to 100 connections
  python capacity_tester.py --max 100

  # Test up to 500 connections with specific model
  python capacity_tester.py --max 500 --model gpt-3.5-turbo

  # Test with custom temperature and 30s recovery delay
  python capacity_tester.py --max 200 --temperature 0.7 --recovery-delay 30

  # Test with no recovery delay between iterations
  python capacity_tester.py --max 100 --recovery-delay 0

  # Run a single test with exactly 100 connections (no binary search)
  python capacity_tester.py --max 100 --no-binary-search

  # Use 4 worker processes for parallel execution
  python capacity_tester.py --max 200 --workers 4

  # Single test with 8 workers and JSON output
  python capacity_tester.py --max 500 --workers 8 --no-binary-search --json-output results.json

  # Binary search with 4 workers and JSON output
  python capacity_tester.py --max 200 --workers 4 --model gpt-4 --json-output capacity_test_results.json

  # Test with custom timeout of 30 seconds
  python capacity_tester.py --max 100 --timeout 30
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
    parser.add_argument(
        '--json-output', '-j',
        type=str,
        default=None,
        help='Path to save JSON output file with detailed statistics (optional)'
    )
    parser.add_argument(
        '--no-binary-search',
        action='store_true',
        help='Skip binary search and run a single test with the specified --max connections'
    )
    parser.add_argument(
        '--model-filter',
        type=str,
        default='model_filter.txt',
        help='Path to file containing model names to exclude (default: model_filter.txt)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Number of worker processes to use for parallel execution (default: 1)'
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Request timeout in seconds (default: 60.0)'
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
    available_models = get_available_models(api_key, api_base, filter_file=args.model_filter)

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

    if args.workers < 1:
        print("Error: Number of workers must be at least 1")
        sys.exit(1)

    if args.timeout <= 0:
        print("Error: Timeout must be greater than 0")
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

    # Run test (either single test or binary search)
    if args.no_binary_search:
        result = await single_test_capacity(
            num_connections=args.max,
            conversations=conversations,
            available_models=available_models,
            temperature=args.temperature,
            api_key=api_key,
            api_base=api_base,
            num_workers=args.workers,
            timeout=args.timeout
        )
    else:
        result = await binary_search_capacity(
            max_connections=args.max,
            conversations=conversations,
            available_models=available_models,
            temperature=args.temperature,
            api_key=api_key,
            api_base=api_base,
            recovery_delay=args.recovery_delay,
            num_workers=args.workers,
            timeout=args.timeout
        )

    # Print detailed statistics
    print_detailed_statistics(result)

    # Save JSON output if requested
    if args.json_output:
        save_json_output(result, args.json_output)

    # Summary
    if args.no_binary_search:
        print(f"\nSUMMARY: Test completed with {result.max_successful_connections}/{args.max} successful connections")
    else:
        print(f"\nSUMMARY: Server can handle up to {result.max_successful_connections} concurrent connections")


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for cross-platform compatibility
    mp.set_start_method('spawn', force=True)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
