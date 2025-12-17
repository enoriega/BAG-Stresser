"""
Incremental capacity testing script.

This script performs incremental stress testing by starting at a base number of
connections and incrementing by a specified step size. It captures error rates
and latency statistics at each step, stopping when either the maximum number of
connections is reached or the error rate exceeds a threshold.

Results are saved to a CSV file for analysis.
"""
import os
import sys
import argparse
import csv
import json
import asyncio
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

from capacity_common import (
    ConnectionAttempt,
    CapacityTestResult,
    get_available_models,
    load_all_conversations,
    send_first_message
)

# Load environment variables
load_dotenv()


@dataclass
class IncrementalStepResult:
    """Results from a single step in the incremental test."""
    step_number: int
    num_connections: int
    successful_connections: int
    failed_connections: int
    error_rate: float
    total_time_seconds: float
    latency_min: float
    latency_mean: float
    latency_max: float
    attempts: List[ConnectionAttempt] = field(default_factory=list)


@dataclass
class IncrementalTestResult:
    """Final results from the incremental test."""
    start_connections: int
    increment_size: int
    max_connections: int
    max_error_rate: float
    cooldown_seconds: float
    steps: List[IncrementalStepResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    stopped_reason: str = ""  # "max_connections_reached" or "error_rate_exceeded"


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
    import random

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


async def monitor_progress(num_connections: int, check_interval: float = 10.0):
    """
    Monitor and print progress periodically.

    Args:
        num_connections: Total number of connections expected
        check_interval: Seconds between progress updates
    """
    start = time.time()
    while True:
        await asyncio.sleep(check_interval)
        elapsed = time.time() - start
        print(f"  [Progress] {elapsed:.0f}s elapsed, still waiting for {num_connections} connections to complete...")


async def test_step(
    step_number: int,
    num_connections: int,
    conversations: Dict[str, dict],
    available_models: List[str],
    temperature: float,
    api_key: str,
    api_base: str,
    num_workers: int = 1,
    timeout: float = 60.0,
    step_timeout: float = 300.0,
    verbose: bool = False
) -> IncrementalStepResult:
    """
    Test a single step with a specific number of concurrent connections.

    Args:
        step_number: Step number in the sequence
        num_connections: Number of concurrent connections to attempt
        conversations: Preloaded conversations dictionary
        available_models: List of available models to randomly select from
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL
        num_workers: Number of worker processes to use (default: 1)
        timeout: Request timeout in seconds (default: 60.0)
        step_timeout: Maximum time to wait for entire step in seconds (default: 300.0)
        verbose: Enable verbose debugging output (default: False)

    Returns:
        IncrementalStepResult with test outcomes
    """
    print(f"\n{'='*70}")
    print(f"Step {step_number}: Testing {num_connections} concurrent connections...")
    if num_workers > 1:
        print(f"Using {num_workers} worker processes")
    print(f"Step timeout: {step_timeout}s")
    print(f"Request timeout: {timeout}s")
    if verbose:
        print(f"Verbose mode: ON")
    print(f"{'='*70}")

    start_time = time.time()
    attempts = []
    timed_out = False

    try:
        if num_workers == 1:
            # Single-process mode: use async only with step timeout
            if verbose:
                print(f"[Verbose] Creating {num_connections} async tasks...")

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

            if verbose:
                print(f"[Verbose] Starting async execution with {step_timeout}s timeout...")

            # Create progress monitor task
            monitor_task = asyncio.create_task(monitor_progress(num_connections, check_interval=10.0))

            try:
                # Run all tasks concurrently with step-level timeout
                attempts = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=False),
                    timeout=step_timeout
                )
                # Cancel monitor task when done
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            except asyncio.TimeoutError:
                # Cancel monitor task on timeout
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
                raise
        else:
            # Multi-process mode: distribute connections across worker processes
            if verbose:
                print(f"[Verbose] Distributing {num_connections} connections across {num_workers} worker processes...")

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
                    if verbose:
                        print(f"[Verbose] Starting worker {worker_id} with {worker_connections} connections...")

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

            if verbose:
                print(f"[Verbose] All {len(processes)} worker processes started. Waiting up to {step_timeout}s...")

            # ABSOLUTE TIMEOUT: No matter what, we stop at step_timeout
            step_deadline = time.time() + step_timeout
            last_progress_time = time.time()
            progress_interval = 10.0

            # Collection loop with absolute deadline
            while time.time() < step_deadline:
                # Check for timeout first - this is our hard limit
                elapsed = time.time() - start_time
                if elapsed >= step_timeout:
                    if verbose:
                        print(f"[Verbose] Absolute step timeout of {step_timeout}s reached. Forcing cleanup...")
                    timed_out = True
                    break

                # Print progress updates
                if time.time() - last_progress_time >= progress_interval:
                    completed = sum(1 for p in processes if not p.is_alive())
                    running = sum(1 for p in processes if p.is_alive())
                    print(f"  [Progress] {elapsed:.0f}s elapsed, {completed}/{len(processes)} workers completed, {running} still running, {len(attempts)} results collected")
                    last_progress_time = time.time()

                # Try to collect results from queue
                try:
                    worker_attempts = result_queue.get(timeout=0.5)
                    attempts.extend(worker_attempts)
                    if verbose:
                        print(f"[Verbose] Collected batch of {len(worker_attempts)} results ({len(attempts)} total)")
                except:
                    # No results available, continue checking deadline
                    pass

                # Check if all processes completed
                if all(not p.is_alive() for p in processes):
                    if verbose:
                        print(f"[Verbose] All worker processes completed")
                    # Give a moment to collect any remaining results
                    final_deadline = time.time() + 2.0
                    while time.time() < final_deadline:
                        try:
                            worker_attempts = result_queue.get(timeout=0.2)
                            attempts.extend(worker_attempts)
                            if verbose:
                                print(f"[Verbose] Collected final batch of {len(worker_attempts)} results")
                        except:
                            break
                    break

            # FORCE CLEANUP: Kill any remaining processes
            alive_processes = [p for p in processes if p.is_alive()]
            if alive_processes:
                print(f"WARNING: Step timeout reached. Terminating {len(alive_processes)} hung worker(s)...")
                if verbose:
                    for p in alive_processes:
                        print(f"[Verbose] Terminating process {p.pid}...")

                for p in alive_processes:
                    try:
                        p.terminate()
                    except:
                        pass

                # Wait briefly for termination
                time.sleep(1.0)

                # Force kill any still alive
                for p in alive_processes:
                    if p.is_alive():
                        if verbose:
                            print(f"[Verbose] Force killing process {p.pid}")
                        try:
                            p.kill()
                        except:
                            pass

                timed_out = True

            if verbose:
                print(f"[Verbose] Final result: {len(attempts)} results collected out of {num_connections} expected")

    except asyncio.TimeoutError:
        timed_out = True
        print(f"WARNING: Step timeout reached after {step_timeout}s")
        print(f"Only {len(attempts)} out of {num_connections} connections completed")

    end_time = time.time()
    total_time = end_time - start_time

    # If step timed out, create failure attempts for missing connections
    if timed_out and len(attempts) < num_connections:
        missing_count = num_connections - len(attempts)
        print(f"Creating {missing_count} timeout failure records for incomplete connections")

        for i in range(missing_count):
            attempts.append(ConnectionAttempt(
                worker_id=-1,  # Special ID for timeout failures
                conversation_file="N/A",
                model_name="N/A",
                success=False,
                error=f"Step timeout: Connection did not complete within {step_timeout}s",
                error_type="StepTimeoutError"
            ))

    # Count successes and failures
    successful = sum(1 for a in attempts if a.success)
    failed = sum(1 for a in attempts if not a.success)

    # Calculate error rate
    error_rate = (failed / num_connections * 100) if num_connections > 0 else 0.0

    # Calculate latency statistics from successful connections
    latencies = [a.latency_seconds for a in attempts if a.success and a.latency_seconds is not None]

    if latencies:
        latency_min = min(latencies)
        latency_mean = sum(latencies) / len(latencies)
        latency_max = max(latencies)
    else:
        latency_min = latency_mean = latency_max = 0.0

    result = IncrementalStepResult(
        step_number=step_number,
        num_connections=num_connections,
        successful_connections=successful,
        failed_connections=failed,
        error_rate=error_rate,
        total_time_seconds=total_time,
        latency_min=latency_min,
        latency_mean=latency_mean,
        latency_max=latency_max,
        attempts=attempts
    )

    print(f"Result: {successful}/{num_connections} succeeded ({100-error_rate:.1f}% success rate)")
    print(f"Error Rate: {error_rate:.2f}%")
    print(f"Latency: min={latency_min:.3f}s, mean={latency_mean:.3f}s, max={latency_max:.3f}s")
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


async def run_incremental_test(
    start_connections: int,
    increment_size: int,
    max_connections: int,
    max_error_rate: float,
    cooldown_seconds: float,
    conversations: Dict[str, dict],
    available_models: List[str],
    temperature: float,
    api_key: str,
    api_base: str,
    num_workers: int = 1,
    timeout: float = 60.0,
    step_timeout: float = 300.0,
    verbose: bool = False
) -> IncrementalTestResult:
    """
    Run incremental capacity test.

    Args:
        start_connections: Initial number of connections
        increment_size: Number of connections to add each step
        max_connections: Maximum number of connections to test
        max_error_rate: Maximum acceptable error rate (percentage)
        cooldown_seconds: Seconds to wait between steps
        conversations: Preloaded conversations dictionary
        available_models: List of available models to randomly select from
        temperature: Temperature parameter
        api_key: API key
        api_base: API base URL
        num_workers: Number of worker processes to use (default: 1)
        timeout: Request timeout in seconds (default: 60.0)
        step_timeout: Maximum time for entire step in seconds (default: 300.0)
        verbose: Enable verbose debugging output (default: False)

    Returns:
        IncrementalTestResult with all step results
    """
    print(f"\n{'='*70}")
    print(f"INCREMENTAL CAPACITY TEST")
    print(f"{'='*70}")
    print(f"Start: {start_connections} connections")
    print(f"Increment: {increment_size} connections per step")
    print(f"Maximum: {max_connections} connections")
    print(f"Max Error Rate: {max_error_rate}%")
    print(f"Cooldown: {cooldown_seconds}s between steps")
    print(f"Step Timeout: {step_timeout}s")
    print(f"Models: {len(available_models)} available (randomly selected per worker)")
    print(f"Temperature: {temperature}")
    if num_workers > 1:
        print(f"Worker processes: {num_workers}")
    print(f"{'='*70}")

    test_start = time.time()

    steps = []
    current_connections = start_connections
    step_number = 1
    stopped_reason = ""

    while current_connections <= max_connections:
        # Run test step
        step_result = await test_step(
            step_number=step_number,
            num_connections=current_connections,
            conversations=conversations,
            available_models=available_models,
            temperature=temperature,
            api_key=api_key,
            api_base=api_base,
            num_workers=num_workers,
            timeout=timeout,
            step_timeout=step_timeout,
            verbose=verbose
        )

        steps.append(step_result)

        # Check if error rate exceeded threshold
        if step_result.error_rate > max_error_rate:
            stopped_reason = "error_rate_exceeded"
            print(f"\nStopping: Error rate ({step_result.error_rate:.2f}%) exceeded threshold ({max_error_rate}%)")
            break

        # Check if we've reached max connections
        if current_connections >= max_connections:
            stopped_reason = "max_connections_reached"
            print(f"\nCompleted: Reached maximum connections ({max_connections})")
            break

        # Increment for next step
        current_connections += increment_size
        step_number += 1

        # Cooldown before next step (but not after the last one)
        if current_connections <= max_connections and cooldown_seconds > 0:
            print(f"\nCooling down for {cooldown_seconds}s...")
            await asyncio.sleep(cooldown_seconds)

    test_end = time.time()
    total_duration = test_end - test_start

    return IncrementalTestResult(
        start_connections=start_connections,
        increment_size=increment_size,
        max_connections=max_connections,
        max_error_rate=max_error_rate,
        cooldown_seconds=cooldown_seconds,
        steps=steps,
        total_duration_seconds=total_duration,
        stopped_reason=stopped_reason
    )


def save_results_to_csv(result: IncrementalTestResult, output_file: str):
    """
    Save incremental test results to CSV file.

    Args:
        result: IncrementalTestResult to save
        output_file: Path to output CSV file
    """
    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'Step',
                'Connections',
                'Successful',
                'Failed',
                'Error Rate (%)',
                'Latency Min (s)',
                'Latency Mean (s)',
                'Latency Max (s)',
                'Total Time (s)',
                'Avg Input Tokens',
                'Avg Output Tokens',
                'Avg Total Tokens',
                'Total Input Tokens',
                'Total Output Tokens',
                'Total Tokens'
            ])

            # Write data rows
            for step in result.steps:
                # Calculate token stats for this step
                step_attempts_with_tokens = [a for a in step.attempts if a.success and a.total_tokens is not None]

                avg_input = sum(a.input_tokens for a in step_attempts_with_tokens if a.input_tokens) / len([a for a in step_attempts_with_tokens if a.input_tokens]) if any(a.input_tokens for a in step_attempts_with_tokens) else 0
                avg_output = sum(a.output_tokens for a in step_attempts_with_tokens if a.output_tokens) / len([a for a in step_attempts_with_tokens if a.output_tokens]) if any(a.output_tokens for a in step_attempts_with_tokens) else 0
                avg_total = sum(a.total_tokens for a in step_attempts_with_tokens if a.total_tokens) / len([a for a in step_attempts_with_tokens if a.total_tokens]) if any(a.total_tokens for a in step_attempts_with_tokens) else 0

                total_input = sum(a.input_tokens for a in step_attempts_with_tokens if a.input_tokens)
                total_output = sum(a.output_tokens for a in step_attempts_with_tokens if a.output_tokens)
                total_tokens = sum(a.total_tokens for a in step_attempts_with_tokens if a.total_tokens)

                writer.writerow([
                    step.step_number,
                    step.num_connections,
                    step.successful_connections,
                    step.failed_connections,
                    f"{step.error_rate:.2f}",
                    f"{step.latency_min:.3f}",
                    f"{step.latency_mean:.3f}",
                    f"{step.latency_max:.3f}",
                    f"{step.total_time_seconds:.2f}",
                    f"{avg_input:.1f}",
                    f"{avg_output:.1f}",
                    f"{avg_total:.1f}",
                    total_input,
                    total_output,
                    total_tokens
                ])

        print(f"\nResults saved to: {output_file}")

        # Print file size
        file_size = Path(output_file).stat().st_size
        if file_size < 1024:
            print(f"File size: {file_size} bytes")
        elif file_size < 1024 * 1024:
            print(f"File size: {file_size / 1024:.2f} KB")
        else:
            print(f"File size: {file_size / (1024 * 1024):.2f} MB")

    except Exception as e:
        print(f"\nError: Failed to save CSV output to {output_file}")
        print(f"  {type(e).__name__}: {e}")


def save_detailed_json(result: IncrementalTestResult, output_file: str, conversations: Dict[str, dict]):
    """
    Save detailed incremental test results including all errors to JSON file.

    Args:
        result: IncrementalTestResult to save
        output_file: Path to output JSON file
        conversations: Dictionary of conversations (for error context)
    """
    try:
        # Build comprehensive JSON structure
        json_data = {
            "test_metadata": {
                "generated_at": datetime.now().isoformat(),
                "test_type": "incremental_capacity_test",
                "start_connections": result.start_connections,
                "increment_size": result.increment_size,
                "max_connections": result.max_connections,
                "max_error_rate": result.max_error_rate,
                "cooldown_seconds": result.cooldown_seconds,
                "total_duration_seconds": result.total_duration_seconds,
                "total_duration_minutes": result.total_duration_seconds / 60,
                "stopped_reason": result.stopped_reason,
                "total_steps": len(result.steps)
            },
            "summary": {
                "total_attempts": sum(step.num_connections for step in result.steps),
                "total_successful": sum(step.successful_connections for step in result.steps),
                "total_failed": sum(step.failed_connections for step in result.steps),
                "overall_error_rate": (
                    sum(step.failed_connections for step in result.steps) /
                    sum(step.num_connections for step in result.steps) * 100
                    if sum(step.num_connections for step in result.steps) > 0 else 0.0
                )
            },
            "steps": [],
            "all_errors": []
        }

        # Build best performance info
        best_step = None
        for step in result.steps:
            if step.error_rate <= result.max_error_rate:
                if best_step is None or step.num_connections > best_step.num_connections:
                    best_step = step

        if best_step:
            json_data["best_performance"] = {
                "step_number": best_step.step_number,
                "num_connections": best_step.num_connections,
                "success_rate": 100 - best_step.error_rate,
                "error_rate": best_step.error_rate,
                "latency_min_seconds": best_step.latency_min,
                "latency_mean_seconds": best_step.latency_mean,
                "latency_max_seconds": best_step.latency_max
            }

        # Process each step
        for step in result.steps:
            step_data = {
                "step_number": step.step_number,
                "num_connections": step.num_connections,
                "successful_connections": step.successful_connections,
                "failed_connections": step.failed_connections,
                "error_rate": step.error_rate,
                "total_time_seconds": step.total_time_seconds,
                "latency_statistics": {
                    "min_seconds": step.latency_min,
                    "mean_seconds": step.latency_mean,
                    "max_seconds": step.latency_max
                },
                "errors_in_step": []
            }

            # Collect all errors with full details
            for attempt in step.attempts:
                if not attempt.success:
                    # Get conversation content if available
                    conversation_content = None
                    if attempt.conversation_file in conversations:
                        conversation_content = conversations[attempt.conversation_file]

                    error_detail = {
                        "worker_id": attempt.worker_id,
                        "timestamp": attempt.timestamp,
                        "conversation_file": attempt.conversation_file,
                        "model_name": attempt.model_name,
                        "error_type": attempt.error_type,
                        "error_message": attempt.error,
                        "error_traceback": attempt.error_traceback,
                        "conversation_content": conversation_content,
                        "step_number": step.step_number,
                        "step_connections": step.num_connections
                    }

                    step_data["errors_in_step"].append(error_detail)
                    json_data["all_errors"].append(error_detail)

            # Add successful attempt statistics
            successful_attempts = [a for a in step.attempts if a.success]
            if successful_attempts:
                step_data["successful_attempts_sample"] = [
                    {
                        "worker_id": attempt.worker_id,
                        "timestamp": attempt.timestamp,
                        "conversation_file": attempt.conversation_file,
                        "model_name": attempt.model_name,
                        "latency_seconds": attempt.latency_seconds,
                        "input_tokens": attempt.input_tokens,
                        "output_tokens": attempt.output_tokens,
                        "total_tokens": attempt.total_tokens
                    }
                    for attempt in successful_attempts[:5]  # First 5 successful attempts
                ]

            # Calculate token statistics for this step
            step_tokens = [a for a in step.attempts if a.success and a.total_tokens is not None]
            if step_tokens:
                step_data["token_statistics"] = {
                    "count": len(step_tokens),
                    "total_input_tokens": sum(a.input_tokens for a in step_tokens if a.input_tokens),
                    "total_output_tokens": sum(a.output_tokens for a in step_tokens if a.output_tokens),
                    "total_tokens": sum(a.total_tokens for a in step_tokens if a.total_tokens),
                    "avg_input_tokens": sum(a.input_tokens for a in step_tokens if a.input_tokens) / len([a for a in step_tokens if a.input_tokens]) if any(a.input_tokens for a in step_tokens) else 0,
                    "avg_output_tokens": sum(a.output_tokens for a in step_tokens if a.output_tokens) / len([a for a in step_tokens if a.output_tokens]) if any(a.output_tokens for a in step_tokens) else 0,
                    "avg_total_tokens": sum(a.total_tokens for a in step_tokens if a.total_tokens) / len([a for a in step_tokens if a.total_tokens]) if any(a.total_tokens for a in step_tokens) else 0
                }

            json_data["steps"].append(step_data)

        # Add error analysis
        error_types = {}
        error_by_model = {}
        error_by_conversation = {}

        for error in json_data["all_errors"]:
            # Count by error type
            error_type = error["error_type"] or "Unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1

            # Count by model
            model = error["model_name"]
            error_by_model[model] = error_by_model.get(model, 0) + 1

            # Count by conversation
            conv = error["conversation_file"]
            error_by_conversation[conv] = error_by_conversation.get(conv, 0) + 1

        json_data["error_analysis"] = {
            "total_errors": len(json_data["all_errors"]),
            "error_types": error_types,
            "errors_by_model": error_by_model,
            "errors_by_conversation": error_by_conversation,
            "unique_error_types": len(error_types),
            "models_with_errors": len(error_by_model),
            "conversations_with_errors": len(error_by_conversation)
        }

        # Calculate latency statistics across all successful attempts
        all_latencies = []
        for step in result.steps:
            for attempt in step.attempts:
                if attempt.success and attempt.latency_seconds is not None:
                    all_latencies.append(attempt.latency_seconds)

        if all_latencies:
            all_latencies_sorted = sorted(all_latencies)
            n = len(all_latencies_sorted)
            json_data["latency_statistics"] = {
                "count": n,
                "min_seconds": min(all_latencies),
                "max_seconds": max(all_latencies),
                "mean_seconds": sum(all_latencies) / n,
                "median_seconds": all_latencies_sorted[n // 2] if n > 0 else None,
                "p95_seconds": all_latencies_sorted[int(n * 0.95)] if n > 0 else None,
                "p99_seconds": all_latencies_sorted[int(n * 0.99)] if n > 0 else None
            }

        # Calculate overall token statistics
        all_attempts_with_tokens = []
        for step in result.steps:
            for attempt in step.attempts:
                if attempt.success and attempt.total_tokens is not None:
                    all_attempts_with_tokens.append(attempt)

        if all_attempts_with_tokens:
            json_data["token_statistics"] = {
                "count": len(all_attempts_with_tokens),
                "total_input_tokens": sum(a.input_tokens for a in all_attempts_with_tokens if a.input_tokens),
                "total_output_tokens": sum(a.output_tokens for a in all_attempts_with_tokens if a.output_tokens),
                "total_tokens": sum(a.total_tokens for a in all_attempts_with_tokens if a.total_tokens),
                "avg_input_tokens": sum(a.input_tokens for a in all_attempts_with_tokens if a.input_tokens) / len([a for a in all_attempts_with_tokens if a.input_tokens]) if any(a.input_tokens for a in all_attempts_with_tokens) else 0,
                "avg_output_tokens": sum(a.output_tokens for a in all_attempts_with_tokens if a.output_tokens) / len([a for a in all_attempts_with_tokens if a.output_tokens]) if any(a.output_tokens for a in all_attempts_with_tokens) else 0,
                "avg_total_tokens": sum(a.total_tokens for a in all_attempts_with_tokens if a.total_tokens) / len([a for a in all_attempts_with_tokens if a.total_tokens]) if any(a.total_tokens for a in all_attempts_with_tokens) else 0,
                "min_input_tokens": min(a.input_tokens for a in all_attempts_with_tokens if a.input_tokens) if any(a.input_tokens for a in all_attempts_with_tokens) else None,
                "max_input_tokens": max(a.input_tokens for a in all_attempts_with_tokens if a.input_tokens) if any(a.input_tokens for a in all_attempts_with_tokens) else None,
                "min_output_tokens": min(a.output_tokens for a in all_attempts_with_tokens if a.output_tokens) if any(a.output_tokens for a in all_attempts_with_tokens) else None,
                "max_output_tokens": max(a.output_tokens for a in all_attempts_with_tokens if a.output_tokens) if any(a.output_tokens for a in all_attempts_with_tokens) else None,
                "min_total_tokens": min(a.total_tokens for a in all_attempts_with_tokens if a.total_tokens) if any(a.total_tokens for a in all_attempts_with_tokens) else None,
                "max_total_tokens": max(a.total_tokens for a in all_attempts_with_tokens if a.total_tokens) if any(a.total_tokens for a in all_attempts_with_tokens) else None
            }

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"\nDetailed JSON results saved to: {output_file}")

        # Print file size
        file_size = Path(output_file).stat().st_size
        if file_size < 1024:
            print(f"File size: {file_size} bytes")
        elif file_size < 1024 * 1024:
            print(f"File size: {file_size / 1024:.2f} KB")
        else:
            print(f"File size: {file_size / (1024 * 1024):.2f} MB")

        # Print error summary
        if json_data["all_errors"]:
            print(f"\nError Summary:")
            print(f"  Total errors: {len(json_data['all_errors'])}")
            print(f"  Unique error types: {len(error_types)}")
            print(f"  Top error types:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    - {error_type}: {count}")

    except Exception as e:
        print(f"\nError: Failed to save JSON output to {output_file}")
        print(f"  {type(e).__name__}: {e}")


def print_summary(result: IncrementalTestResult):
    """
    Print summary of incremental test results.

    Args:
        result: IncrementalTestResult to summarize
    """
    print(f"\n{'='*70}")
    print(f"INCREMENTAL TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Start: {result.start_connections} connections")
    print(f"  Increment: {result.increment_size} connections")
    print(f"  Maximum: {result.max_connections} connections")
    print(f"  Max Error Rate: {result.max_error_rate}%")
    print(f"  Cooldown: {result.cooldown_seconds}s")
    print()

    print(f"Test Results:")
    print(f"  Total Steps: {len(result.steps)}")
    print(f"  Total Duration: {result.total_duration_seconds:.2f}s ({result.total_duration_seconds/60:.1f} minutes)")
    print(f"  Stopped Reason: {result.stopped_reason.replace('_', ' ').title()}")
    print()

    # Find best performing step (highest connections with acceptable error rate)
    best_step = None
    for step in result.steps:
        if step.error_rate <= result.max_error_rate:
            if best_step is None or step.num_connections > best_step.num_connections:
                best_step = step

    if best_step:
        print(f"Best Performance (within error threshold):")
        print(f"  Connections: {best_step.num_connections}")
        print(f"  Success Rate: {100 - best_step.error_rate:.2f}%")
        print(f"  Error Rate: {best_step.error_rate:.2f}%")
        print(f"  Latency: min={best_step.latency_min:.3f}s, mean={best_step.latency_mean:.3f}s, max={best_step.latency_max:.3f}s")
        print()

    # Step-by-step results table
    print(f"Step-by-Step Results:")
    print(f"{'-'*70}")
    print(f"{'Step':<6} {'Conn':<6} {'Success':<8} {'Failed':<7} {'Error%':<8} {'LatMin':<8} {'LatMean':<9} {'LatMax':<8}")
    print(f"{'-'*70}")

    for step in result.steps:
        marker = "!" if step.error_rate > result.max_error_rate else " "
        print(f"{marker}{step.step_number:<5} {step.num_connections:<6} {step.successful_connections:<8} "
              f"{step.failed_connections:<7} {step.error_rate:>6.2f}% {step.latency_min:>7.3f}s "
              f"{step.latency_mean:>8.3f}s {step.latency_max:>7.3f}s")

    print(f"{'-'*70}")
    print(f"Note: Steps marked with '!' exceeded the error rate threshold")
    print(f"{'='*70}")


async def main():
    """Main async function."""
    parser = argparse.ArgumentParser(
        description='Incremental capacity testing with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test starting at 100, incrementing by 100, up to 1000 connections
  python incremental_test.py --start 100 --increment 100 --max 1000

  # Test with 5% max error rate and 30s cooldown
  python incremental_test.py --start 50 --increment 50 --max 500 --max-error-rate 5 --cooldown 30

  # Test with 4 workers and save to custom output files
  python incremental_test.py --start 100 --increment 100 --max 1000 --workers 4 --output results.csv --json-output results.json

  # Test with custom model and no cooldown
  python incremental_test.py --start 100 --increment 100 --max 500 --model gpt-4 --cooldown 0
        """
    )

    parser.add_argument(
        '--start', '-s',
        type=int,
        default=100,
        help='Starting number of connections (default: 100)'
    )
    parser.add_argument(
        '--increment', '-i',
        type=int,
        default=100,
        help='Number of connections to increment each step (default: 100)'
    )
    parser.add_argument(
        '--max', '-m',
        type=int,
        default=1000,
        help='Maximum number of connections to test (default: 1000)'
    )
    parser.add_argument(
        '--max-error-rate', '-e',
        type=float,
        default=5.0,
        help='Maximum acceptable error rate percentage (default: 5.0)'
    )
    parser.add_argument(
        '--cooldown', '-c',
        type=float,
        default=15.0,
        help='Seconds to wait between steps (default: 15.0, use 0 to disable)'
    )
    parser.add_argument(
        '--step-timeout',
        type=float,
        default=300.0,
        help='Maximum time in seconds for an entire step to complete before marking incomplete connections as failures (default: 300.0)'
    )
    parser.add_argument(
        '--conversations-dir',
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
        '--output', '-o',
        type=str,
        default='incremental_test_results.csv',
        help='Path to save CSV output file (default: incremental_test_results.csv)'
    )
    parser.add_argument(
        '--json-output', '-j',
        type=str,
        default='incremental_test_results.json',
        help='Path to save detailed JSON output file with all errors (default: incremental_test_results.json)'
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
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debugging output to diagnose hanging or slow connections'
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
    if args.start < 1:
        print("Error: Starting connections must be at least 1")
        sys.exit(1)

    if args.increment < 1:
        print("Error: Increment must be at least 1")
        sys.exit(1)

    if args.max < args.start:
        print("Error: Maximum connections must be >= starting connections")
        sys.exit(1)

    if args.max_error_rate < 0 or args.max_error_rate > 100:
        print("Error: Max error rate must be between 0 and 100")
        sys.exit(1)

    if args.cooldown < 0:
        print("Error: Cooldown must be non-negative")
        sys.exit(1)

    if args.temperature < 0 or args.temperature > 2.0:
        print("Error: Temperature must be between 0 and 2.0")
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

    # Run incremental test
    result = await run_incremental_test(
        start_connections=args.start,
        increment_size=args.increment,
        max_connections=args.max,
        max_error_rate=args.max_error_rate,
        cooldown_seconds=args.cooldown,
        conversations=conversations,
        available_models=available_models,
        temperature=args.temperature,
        api_key=api_key,
        api_base=api_base,
        num_workers=args.workers,
        timeout=args.timeout,
        step_timeout=args.step_timeout,
        verbose=args.verbose
    )

    # Print summary
    print_summary(result)

    # Save to CSV
    save_results_to_csv(result, args.output)

    # Save detailed JSON with all errors
    save_detailed_json(result, args.json_output, conversations)


def cleanup_processes():
    """Clean up any remaining child processes."""
    current_process = mp.current_process()
    for child in mp.active_children():
        if child != current_process:
            try:
                child.terminate()
                child.join(timeout=2)
                if child.is_alive():
                    child.kill()
            except:
                pass


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for cross-platform compatibility
    mp.set_start_method('spawn', force=True)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
        cleanup_processes()
        print("Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        cleanup_processes()
        sys.exit(1)
