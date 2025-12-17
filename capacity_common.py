"""
Shared functionality for capacity testing scripts.

This module contains common data structures, functions, and utilities
used by both the local capacity_tester.py and SLURM-based testing scripts.
"""
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


@dataclass
class ConnectionAttempt:
    """Details about a single connection attempt."""
    worker_id: int
    conversation_file: str
    model_name: str
    success: bool
    error: Optional[str] = None
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None
    latency_seconds: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
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


async def send_first_message(
    conversation_data: dict,
    conversation_filename: str,
    model_name: str,
    temperature: float,
    api_key: str,
    api_base: str,
    worker_id: int,
    timeout: float = 60.0
) -> ConnectionAttempt:
    """
    Send only the first message from a conversation (async non-blocking).

    Args:
        conversation_data: Preloaded conversation data
        conversation_filename: Name of the conversation file
        model_name: Model to use
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL
        worker_id: Worker process ID
        timeout: Request timeout in seconds (default: 60.0)

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

        # Create LLM client with timeout
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=api_base,
            timeout=timeout
        )

        # Send message and measure latency (non-blocking async call)
        start_time = time.time()
        response = await llm.ainvoke([HumanMessage(content=first_message)])
        end_time = time.time()

        latency = end_time - start_time

        # Extract token usage from response metadata
        input_tokens = None
        output_tokens = None
        total_tokens = None

        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('token_usage', {})
            if usage:
                input_tokens = usage.get('prompt_tokens')
                output_tokens = usage.get('completion_tokens')
                total_tokens = usage.get('total_tokens')

        return ConnectionAttempt(
            worker_id=worker_id,
            conversation_file=conversation_filename,
            model_name=model_name,
            success=True,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )

    except Exception as e:
        return ConnectionAttempt(
            worker_id=worker_id,
            conversation_file=conversation_filename,
            model_name=model_name,
            success=False,
            error=str(e),
            error_type=type(e).__name__,
            error_traceback=traceback.format_exc()
        )


def serialize_to_json(result: BinarySearchResult) -> dict:
    """
    Convert BinarySearchResult to JSON-serializable dictionary.

    Args:
        result: BinarySearchResult to serialize

    Returns:
        Dictionary containing all test data in JSON-serializable format
    """
    return {
        "summary": {
            "max_successful_connections": result.max_successful_connections,
            "min_failed_connections": result.min_failed_connections,
            "total_tests": result.total_tests,
            "total_duration_seconds": result.total_duration_seconds,
            "total_duration_minutes": result.total_duration_seconds / 60
        },
        "test_results": [
            {
                "num_connections": test.num_connections,
                "successful_connections": test.successful_connections,
                "failed_connections": test.failed_connections,
                "total_time_seconds": test.total_time_seconds,
                "success_rate": test.success_rate,
                "is_success": test.is_success,
                "attempts": [
                    {
                        "worker_id": attempt.worker_id,
                        "conversation_file": attempt.conversation_file,
                        "model_name": attempt.model_name,
                        "success": attempt.success,
                        "error": attempt.error,
                        "error_type": attempt.error_type,
                        "error_traceback": attempt.error_traceback,
                        "latency_seconds": attempt.latency_seconds,
                        "timestamp": attempt.timestamp
                    }
                    for attempt in test.attempts
                ]
            }
            for test in result.test_results
        ],
        "aggregate_statistics": {
            "total_attempts": sum(len(t.attempts) for t in result.test_results),
            "total_successful": sum(t.successful_connections for t in result.test_results),
            "total_failed": sum(t.failed_connections for t in result.test_results),
            "overall_success_rate": (
                sum(t.successful_connections for t in result.test_results) /
                sum(len(t.attempts) for t in result.test_results) * 100
                if sum(len(t.attempts) for t in result.test_results) > 0 else 0.0
            )
        },
        "latency_statistics": _calculate_latency_stats(result),
        "error_analysis": _calculate_error_stats(result),
        "generated_at": datetime.now().isoformat()
    }


def _calculate_latency_stats(result: BinarySearchResult) -> Optional[dict]:
    """Calculate latency statistics from successful connections."""
    all_latencies = []
    for test_result in result.test_results:
        for attempt in test_result.attempts:
            if attempt.success and attempt.latency_seconds is not None:
                all_latencies.append(attempt.latency_seconds)

    if not all_latencies:
        return None

    all_latencies.sort()
    n = len(all_latencies)

    return {
        "count": n,
        "average_seconds": sum(all_latencies) / n,
        "min_seconds": min(all_latencies),
        "max_seconds": max(all_latencies),
        "median_seconds": all_latencies[n // 2] if n > 0 else None,
        "p95_seconds": all_latencies[int(n * 0.95)] if n > 0 else None,
        "p99_seconds": all_latencies[int(n * 0.99)] if n > 0 else None
    }


def _calculate_error_stats(result: BinarySearchResult) -> dict:
    """Calculate error statistics from failed connections."""
    all_errors = []
    error_types = {}

    for test_result in result.test_results:
        for attempt in test_result.attempts:
            if not attempt.success:
                all_errors.append({
                    "worker_id": attempt.worker_id,
                    "conversation_file": attempt.conversation_file,
                    "model_name": attempt.model_name,
                    "error": attempt.error,
                    "error_type": attempt.error_type,
                    "error_traceback": attempt.error_traceback,
                    "timestamp": attempt.timestamp
                })
                error_type = attempt.error_type or "Unknown"
                error_types[error_type] = error_types.get(error_type, 0) + 1

    return {
        "total_errors": len(all_errors),
        "error_types": error_types,
        "all_errors": all_errors
    }


def save_json_output(result: BinarySearchResult, output_file: str):
    """
    Save test results to JSON file.

    Args:
        result: BinarySearchResult to save
        output_file: Path to output JSON file
    """
    try:
        json_data = serialize_to_json(result)

        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"\nJSON output saved to: {output_file}")

        # Print file size
        file_size = Path(output_file).stat().st_size
        if file_size < 1024:
            print(f"File size: {file_size} bytes")
        elif file_size < 1024 * 1024:
            print(f"File size: {file_size / 1024:.2f} KB")
        else:
            print(f"File size: {file_size / (1024 * 1024):.2f} MB")

    except Exception as e:
        print(f"\nError: Failed to save JSON output to {output_file}")
        print(f"  {type(e).__name__}: {e}")


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
