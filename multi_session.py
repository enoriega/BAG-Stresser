"""
Run multiple concurrent user sessions for stress testing.

This script launches multiple user sessions in parallel using asyncio and multiprocessing,
allowing for higher throughput stress testing.

With multiprocessing enabled:
- N = concurrent sessions per process (--sessions)
- M = number of worker processes (--workers)
- Total concurrent sessions = N × M
"""
import os
import sys
import asyncio
import argparse
import multiprocessing
from datetime import datetime
from dotenv import load_dotenv
from stresser import simulate_user_session, print_session_report, UserSessionStats

# Load environment variables
load_dotenv()


async def run_multi_session(
    num_sessions: int,
    duration: int,
    conversations_dir: str,
    model_name: str = None,
    api_key: str = None,
    api_base: str = None,
    temperature_range: tuple[float, float] = (0.5, 1.0),
    worker_id: int = None,
    model_filter_file: str = "model_filter.txt"
) -> list[UserSessionStats]:
    """
    Run multiple user sessions concurrently.

    Args:
        num_sessions: Number of concurrent sessions to run
        duration: Duration in seconds for each session
        conversations_dir: Directory containing conversation files
        model_name: Model to use (None = random selection)
        api_key: API key
        api_base: API base URL
        temperature_range: Temperature range for random selection
        worker_id: Optional worker ID for logging (used in multiprocessing mode)
        model_filter_file: Path to file containing model names to exclude

    Returns:
        List of UserSessionStats from each session
    """
    # Prefix for all log messages
    prefix = f"[Worker {worker_id}] " if worker_id is not None else ""

    print("=" * 70)
    print(f"{prefix}MULTI-SESSION STRESS TEST")
    print("=" * 70)
    print(f"{prefix}Sessions: {num_sessions} concurrent sessions")
    print(f"{prefix}Duration: {duration} seconds per session")
    print(f"{prefix}Model: {model_name if model_name else 'Random selection'}")
    print(f"{prefix}Temperature range: {temperature_range[0]} - {temperature_range[1]}")
    print("=" * 70)
    print()

    # Create tasks for all sessions
    tasks = []
    for session_id in range(1, num_sessions + 1):
        print(f"{prefix}[Session {session_id}] Preparing to launch...")
        task = asyncio.create_task(
            run_single_session(
                session_id=session_id,
                duration=duration,
                conversations_dir=conversations_dir,
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                temperature_range=temperature_range,
                worker_id=worker_id,
                model_filter_file=model_filter_file
            )
        )
        tasks.append(task)

    print()
    print(f"{prefix}Launching all {num_sessions} sessions concurrently...")
    print("=" * 70)
    print()

    # Run all sessions concurrently
    start_time = datetime.now()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = datetime.now()

    # Filter out exceptions and get successful results
    successful_results = []
    failed_count = 0
    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"\n{prefix}[Session {i}] FAILED with error: {result}")
            failed_count += 1
        else:
            successful_results.append(result)

    print()
    print("=" * 70)
    print(f"{prefix}ALL SESSIONS COMPLETED")
    print("=" * 70)
    print(f"{prefix}Total elapsed time: {(end_time - start_time).total_seconds():.1f} seconds")
    print(f"{prefix}Successful sessions: {len(successful_results)}/{num_sessions}")
    if failed_count > 0:
        print(f"{prefix}Failed sessions: {failed_count}")
    print()

    return successful_results


async def run_single_session(
    session_id: int,
    duration: int,
    conversations_dir: str,
    model_name: str = None,
    api_key: str = None,
    api_base: str = None,
    temperature_range: tuple[float, float] = (0.5, 1.0),
    worker_id: int = None,
    model_filter_file: str = "model_filter.txt"
) -> UserSessionStats:
    """
    Run a single user session with session ID prefix for logging.

    Args:
        session_id: Unique identifier for this session
        duration: Duration in seconds
        conversations_dir: Directory containing conversation files
        model_name: Model to use (None = random selection)
        api_key: API key
        api_base: API base URL
        temperature_range: Temperature range for random selection
        worker_id: Optional worker ID for logging (used in multiprocessing mode)
        model_filter_file: Path to file containing model names to exclude

    Returns:
        UserSessionStats from the session
    """
    # Build logging prefix
    if worker_id is not None:
        prefix = f"[Worker {worker_id}][Session {session_id}]"
    else:
        prefix = f"[Session {session_id}]"

    print(f"\n{prefix} Starting...")

    try:
        stats = await simulate_user_session(
            conversations_dir=conversations_dir,
            duration_seconds=duration,
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature_range=temperature_range,
            model_filter_file=model_filter_file
        )

        print(f"\n{prefix} ✓ Completed successfully")
        print(f"{prefix}   - Conversations: {stats.total_conversations}")
        print(f"{prefix}   - Messages: {stats.total_messages_sent}")
        print(f"{prefix}   - Tokens: {stats.total_tokens:,}")

        return stats

    except Exception as e:
        print(f"\n{prefix} ✗ Failed: {str(e)}")
        raise


def aggregate_session_stats(all_sessions: list[UserSessionStats]) -> dict:
    """
    Aggregate statistics from multiple sessions.

    Args:
        all_sessions: List of UserSessionStats from each session

    Returns:
        Dictionary with aggregated statistics
    """
    if not all_sessions:
        return {}

    total_conversations = sum(s.total_conversations for s in all_sessions)
    total_successful = sum(s.successful_conversations for s in all_sessions)
    total_failed = sum(s.failed_conversations for s in all_sessions)
    total_messages = sum(s.total_messages_sent for s in all_sessions)
    total_tokens = sum(s.total_tokens for s in all_sessions)
    total_tokens_input = sum(s.total_tokens_input for s in all_sessions)
    total_tokens_output = sum(s.total_tokens_output for s in all_sessions)

    # Collect all latencies across all sessions
    all_latencies = []
    for session in all_sessions:
        for conv_stat in session.conversation_stats:
            if not conv_stat.error:
                all_latencies.extend(conv_stat.message_latencies)

    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    min_latency = min(all_latencies) if all_latencies else 0.0
    max_latency = max(all_latencies) if all_latencies else 0.0

    # Collect all models used
    all_models = set()
    for session in all_sessions:
        all_models.update(session.models_used)

    # Collect all error details across all sessions
    all_error_details = []
    for session in all_sessions:
        for conv_stat in session.conversation_stats:
            if conv_stat.error and conv_stat.error_details:
                all_error_details.append(conv_stat.error_details)

    # Calculate total duration (max of all sessions)
    total_duration = max(s.duration_seconds for s in all_sessions)

    return {
        'num_sessions': len(all_sessions),
        'total_conversations': total_conversations,
        'successful_conversations': total_successful,
        'failed_conversations': total_failed,
        'total_messages': total_messages,
        'total_tokens': total_tokens,
        'total_tokens_input': total_tokens_input,
        'total_tokens_output': total_tokens_output,
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'models_used': sorted(list(all_models)),
        'total_duration': total_duration,
        'conversations_per_minute': (total_conversations / total_duration) * 60 if total_duration > 0 else 0,
        'tokens_per_second': total_tokens / total_duration if total_duration > 0 else 0,
        'error_details': all_error_details
    }


def print_aggregate_report(stats: dict):
    """
    Print aggregated statistics from multiple sessions.

    Args:
        stats: Dictionary with aggregated statistics
    """
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS ACROSS ALL SESSIONS")
    print("=" * 70)
    print(f"Number of Sessions: {stats['num_sessions']}")
    print(f"Total Duration: {stats['total_duration']:.1f}s")
    print()

    print("CONVERSATION STATISTICS")
    print("-" * 70)
    print(f"Total Conversations: {stats['total_conversations']}")
    print(f"  Successful: {stats['successful_conversations']} "
          f"({stats['successful_conversations']/stats['total_conversations']*100:.1f}%)")
    print(f"  Failed: {stats['failed_conversations']} "
          f"({stats['failed_conversations']/stats['total_conversations']*100:.1f}%)")
    print(f"  Rate: {stats['conversations_per_minute']:.2f} conversations/minute")
    print()

    print("MESSAGE & TOKEN STATISTICS")
    print("-" * 70)
    print(f"Total Messages Sent: {stats['total_messages']:,}")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print(f"  Input Tokens: {stats['total_tokens_input']:,}")
    print(f"  Output Tokens: {stats['total_tokens_output']:,}")
    print(f"  Throughput: {stats['tokens_per_second']:.1f} tokens/second")
    print()

    print("LATENCY STATISTICS")
    print("-" * 70)
    print(f"Average Latency: {stats['avg_latency']:.3f}s")
    print(f"Min Latency: {stats['min_latency']:.3f}s")
    print(f"Max Latency: {stats['max_latency']:.3f}s")
    print()

    print("MODELS USED")
    print("-" * 70)
    for model in stats['models_used']:
        print(f"  - {model}")
    print()

    # Display detailed error information if there are any failures
    if stats['failed_conversations'] > 0 and stats.get('error_details'):
        print("ERRORS")
        print("-" * 70)

        # Group errors by type and count them
        error_counts = {}
        for error_detail in stats['error_details']:
            error_type = error_detail.exception_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        # Print error summary by type
        print("\nError Summary:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  [{count}x] {error_type}")

        # Print detailed error information (up to 10 errors)
        print("\nDetailed Error Information:")
        print("-" * 70)
        for idx, error in enumerate(stats['error_details'][:10], 1):
            print(f"\n[Error {idx}]")
            print(f"  Model: {error.model_name}")
            print(f"  Conversation File: {error.conversation_file}")
            print(f"  Failed at Step: {error.step}" +
                  (" (initialization)" if error.step == 0 else
                   " (before start)" if error.step == -1 else ""))
            print(f"  Exception Type: {error.exception_type}")
            print(f"  Error Message: {error.error_message}")
            if error.server_error:
                # Truncate server error if too long
                server_err_display = error.server_error[:200] + "..." if len(error.server_error) > 200 else error.server_error
                print(f"  Server Error: {server_err_display}")
            print(f"  Timestamp: {error.timestamp}")

        if len(stats['error_details']) > 10:
            print(f"\n... and {len(stats['error_details']) - 10} more errors")
        print()

    print("=" * 70)


def run_worker_process(
    worker_id: int,
    num_sessions: int,
    duration: int,
    conversations_dir: str,
    model_name: str,
    api_key: str,
    api_base: str,
    temperature_range: tuple[float, float],
    model_filter_file: str
) -> list[UserSessionStats]:
    """
    Worker process function that runs multiple sessions.

    This function is executed in a separate process. It creates a new event loop
    and runs the async run_multi_session function.

    Args:
        worker_id: Unique identifier for this worker process
        num_sessions: Number of concurrent sessions this worker should run
        duration: Duration in seconds for each session
        conversations_dir: Directory containing conversation files
        model_name: Model to use (None = random selection)
        api_key: API key
        api_base: API base URL
        temperature_range: Temperature range for random selection
        model_filter_file: Path to file containing model names to exclude

    Returns:
        List of UserSessionStats from all sessions in this worker
    """
    print(f"\n{'=' * 70}")
    print(f"[Worker {worker_id}] Process started (PID: {os.getpid()})")
    print(f"{'=' * 70}")

    # Create new event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Run the async multi-session function with worker_id for logging
        results = loop.run_until_complete(
            run_multi_session(
                num_sessions=num_sessions,
                duration=duration,
                conversations_dir=conversations_dir,
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                temperature_range=temperature_range,
                worker_id=worker_id,
                model_filter_file=model_filter_file
            )
        )
        print(f"\n{'=' * 70}")
        print(f"[Worker {worker_id}] Completed with {len(results)} successful sessions")
        print(f"{'=' * 70}")
        return results
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"[Worker {worker_id}] Failed with error: {e}")
        print(f"{'=' * 70}")
        return []
    finally:
        loop.close()


def run_with_multiprocessing(
    num_workers: int,
    num_sessions_per_worker: int,
    duration: int,
    conversations_dir: str,
    model_name: str,
    api_key: str,
    api_base: str,
    temperature_range: tuple[float, float],
    model_filter_file: str = "model_filter.txt"
) -> list[UserSessionStats]:
    """
    Run multiple worker processes, each running multiple sessions.

    Args:
        num_workers: Number of worker processes (M)
        num_sessions_per_worker: Number of concurrent sessions per worker (N)
        duration: Duration in seconds for each session
        conversations_dir: Directory containing conversation files
        model_name: Model to use (None = random selection)
        api_key: API key
        api_base: API base URL
        temperature_range: Temperature range for random selection
        model_filter_file: Path to file containing model names to exclude

    Returns:
        Combined list of UserSessionStats from all workers
    """
    total_sessions = num_workers * num_sessions_per_worker

    print("=" * 70)
    print("MULTI-PROCESS STRESS TEST")
    print("=" * 70)
    print(f"Worker Processes: {num_workers}")
    print(f"Sessions per Worker: {num_sessions_per_worker}")
    print(f"Total Concurrent Sessions: {total_sessions} (N×M)")
    print(f"Duration: {duration} seconds per session")
    print(f"Model: {model_name if model_name else 'Random selection'}")
    print(f"Temperature range: {temperature_range[0]} - {temperature_range[1]}")
    print("=" * 70)
    print()

    # Create process pool and launch workers
    print(f"Launching {num_workers} worker processes...")
    start_time = datetime.now()

    with multiprocessing.Pool(processes=num_workers) as pool:
        # Create argument tuples for each worker
        worker_args = [
            (
                worker_id,
                num_sessions_per_worker,
                duration,
                conversations_dir,
                model_name,
                api_key,
                api_base,
                temperature_range,
                model_filter_file
            )
            for worker_id in range(1, num_workers + 1)
        ]

        # Launch all workers and collect results
        worker_results = pool.starmap(run_worker_process, worker_args)

    end_time = datetime.now()

    # Flatten results from all workers
    all_results = []
    for results in worker_results:
        all_results.extend(results)

    print()
    print("=" * 70)
    print("ALL WORKERS COMPLETED")
    print("=" * 70)
    print(f"Total elapsed time: {(end_time - start_time).total_seconds():.1f} seconds")
    print(f"Total successful sessions: {len(all_results)}/{total_sessions}")
    print()

    return all_results


async def main():
    """Main async function."""
    parser = argparse.ArgumentParser(
        description='Run multiple concurrent user sessions for stress testing with optional multiprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 1 session (default) for 30 seconds
  python multi_session.py --duration 30

  # Run 3 concurrent sessions for 60 seconds
  python multi_session.py --sessions 3 --duration 60

  # Run 5 sessions with a specific model
  python multi_session.py --sessions 5 --duration 45 --model gpt-4

  # Run 10 sessions with custom temperature range
  python multi_session.py --sessions 10 --duration 30 --temp-min 0.3 --temp-max 0.9

  # Use multiprocessing: 4 worker processes, each running 3 sessions = 12 total
  python multi_session.py --sessions 3 --workers 4 --duration 60

  # Maximum stress: 5 workers × 5 sessions = 25 concurrent sessions
  python multi_session.py --sessions 5 --workers 5 --duration 30
        """
    )

    parser.add_argument(
        '--sessions', '-s',
        type=int,
        default=1,
        help='Number of concurrent sessions per worker (default: 1)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Number of worker processes (default: 1). Total sessions = sessions × workers'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Duration in seconds for each session (default: 60)'
    )
    parser.add_argument(
        '--conversations-dir', '-c',
        type=str,
        default='conversations',
        help='Directory containing conversation JSON files (default: conversations)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Model name to use (default: random selection from API)'
    )
    parser.add_argument(
        '--temp-min',
        type=float,
        default=0.5,
        help='Minimum temperature for random selection (default: 0.5)'
    )
    parser.add_argument(
        '--temp-max',
        type=float,
        default=1.0,
        help='Maximum temperature for random selection (default: 1.0)'
    )
    parser.add_argument(
        '--show-individual',
        action='store_true',
        help='Show detailed report for each individual session'
    )
    parser.add_argument(
        '--model-filter',
        type=str,
        default='model_filter.txt',
        help='Path to file containing model names to exclude (default: model_filter.txt)'
    )

    args = parser.parse_args()

    # Get API configuration from environment
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE')
    model_name = args.model or os.getenv('MODEL_NAME')

    if not api_key or not api_base:
        print("Error: OPENAI_API_KEY and OPENAI_API_BASE must be set in .env file")
        sys.exit(1)

    # Validate arguments
    if args.sessions < 1:
        print("Error: Number of sessions must be at least 1")
        sys.exit(1)

    if args.workers < 1:
        print("Error: Number of workers must be at least 1")
        sys.exit(1)

    if args.duration < 1:
        print("Error: Duration must be at least 1 second")
        sys.exit(1)

    if args.temp_min < 0 or args.temp_max > 2.0 or args.temp_min >= args.temp_max:
        print("Error: Invalid temperature range (min must be < max, both in range 0-2.0)")
        sys.exit(1)

    # Choose execution mode based on number of workers
    if args.workers > 1:
        # Use multiprocessing
        results = run_with_multiprocessing(
            num_workers=args.workers,
            num_sessions_per_worker=args.sessions,
            duration=args.duration,
            conversations_dir=args.conversations_dir,
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature_range=(args.temp_min, args.temp_max),
            model_filter_file=args.model_filter
        )
    else:
        # Use asyncio only (single process)
        results = await run_multi_session(
            num_sessions=args.sessions,
            duration=args.duration,
            conversations_dir=args.conversations_dir,
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature_range=(args.temp_min, args.temp_max),
            model_filter_file=args.model_filter
        )

    # Print individual session reports if requested
    if args.show_individual and results:
        print("\n" + "=" * 70)
        print("INDIVIDUAL SESSION REPORTS")
        print("=" * 70)
        for i, session_stats in enumerate(results, 1):
            print(f"\n### Session {i} ###")
            print_session_report(session_stats)

    # Print aggregate statistics
    if results:
        aggregate_stats = aggregate_session_stats(results)
        print_aggregate_report(aggregate_stats)
    else:
        print("\nNo successful sessions to report.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
