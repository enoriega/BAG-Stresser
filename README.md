# BAG-Stresser

An LLM OpenAI API stress testing application built with Python and LangChain.

## Overview

BAG-Stresser is designed to test and measure the performance of LLM APIs by simulating realistic conversation patterns. It replays pre-scripted conversations, measuring latency, token usage, and other key metrics.

## Features

- **Conversation Replay**: Load and replay conversations from JSON files
- **User Session Simulation**: Simulate realistic user behavior with repeated stress tests over time
- **Async/Await Architecture**: Built with asyncio for efficient concurrent execution
- **Progressive Conversation Building**: Simulates natural conversation flow by building context progressively
- **Intelligent Timing**: Uses async sleep between messages based on message length to simulate realistic user behavior
- **Random Parameter Selection**: Automatically varies conversations, models, and temperature for realistic testing
- **Comprehensive Statistics**: Tracks tokens (input/output), latency (min/max/average), success rates, and more
- **Configurable**: Control model parameters, message limits, and API endpoints
- **No Token Limits**: LLMs can generate responses of unlimited length
- **Well-Tested**: Comprehensive unit test suite with 25 tests

## Installation

This project uses [uv](https://github.com/astral-sh/uv) as the Python package manager.

```bash
# Install dependencies
uv sync

# Install with dev dependencies (for testing)
uv sync --extra dev
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API credentials:
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=500
```

## Usage

### Basic Example

```python
import asyncio
from stresser import run_conversation_stress_test

async def main():
    stats = await run_conversation_stress_test(
        conversation_file_path='conversations/conversation_01.json',
        model_name='gpt-3.5-turbo',
        temperature=0.7,
        max_tokens=None,  # No limit - let LLM generate as much as it wants
        max_messages=None,  # Send all messages
        api_key='your_api_key',
        api_base='https://api.openai.com/v1'
    )

    print(f"Total messages: {stats.total_messages_sent}")
    print(f"Average latency: {stats.average_latency_seconds:.3f}s")
    print(f"Total tokens: {stats.total_tokens}")

asyncio.run(main())
```

### Running the Example Script

```bash
uv run python example_stresser.py
```

### User Session Simulation

Simulate a realistic user session with repeated stress tests over a specified duration:

```python
import asyncio
from stresser import simulate_user_session, print_session_report

async def main():
    # Run simulation for 60 seconds (default)
    # Randomly selects conversations, models, and parameters
    stats = await simulate_user_session(
        conversations_dir='conversations',
        duration_seconds=60,  # How long to run
        model_name=None,  # None = random from API, or specify a model
        api_key='your_api_key',
        api_base='https://api.openai.com/v1',
        temperature_range=(0.5, 1.0)  # Random temperature range
    )

    # Print comprehensive report
    print_session_report(stats)

asyncio.run(main())
```

**Features:**
- **Time-based execution**: Runs continuously for specified duration
- **Random selection**: Automatically varies conversations, models, and temperature
- **Aggregate statistics**: Collects metrics across all conversations
- **Success rate tracking**: Monitors failures and errors
- **Formatted reporting**: Professional summary with all key metrics

**Example output:**
```
USER SESSION SUMMARY REPORT
============================================================
Session Duration: 60s (1.0 minutes)
Total Conversations: 15
  Successful: 14 (93.3%)
  Failed: 1 (6.7%)
  Rate: 15.00 conversations/minute

MESSAGE & TOKEN STATISTICS
Total Messages Sent: 65
Total Tokens: 8,450
  Input Tokens: 3,200
  Output Tokens: 5,250

LATENCY STATISTICS
Average Latency: 1.823s
Min Latency: 0.654s
Max Latency: 4.123s
```

### Running the Session Example

```bash
uv run python example_session.py
```

## Conversation Format

Conversations are stored as JSON files with the following structure:

```json
{
  "conversation_id": "conv_01",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you!"
    },
    {
      "role": "user",
      "content": "Can you help me with Python?"
    },
    {
      "role": "assistant",
      "content": "Of course! I'd be happy to help."
    }
  ]
}
```

The application will:
1. Send the first user message
2. Receive a response from the API
3. Replace the API response with the scripted assistant response
4. Async sleep for a time proportional to the next message length
5. Send the conversation history + next user message
6. Repeat until all messages are sent or `max_messages` is reached

## Core Functions

### `run_conversation_stress_test()`

Run a single conversation stress test (async function).

```python
async def run_conversation_stress_test(
    conversation_file_path: str,
    model_name: str,
    temperature: float,
    max_tokens: Optional[int] = None,
    max_messages: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None
) -> ConversationStats
```

**Parameters:**
- `conversation_file_path`: Path to the JSON conversation file
- `model_name`: Name of the model (e.g., 'gpt-3.5-turbo', 'gpt-4')
- `temperature`: Temperature parameter for the model (0.0-2.0)
- `max_tokens`: Maximum tokens for each response (None = no limit, let LLM generate as much as it wants)
- `max_messages`: Maximum number of user messages to send (None = all)
- `api_key`: OpenAI API key (optional, uses env var if not provided)
- `api_base`: API base URL (optional, uses default if not provided)

**Returns:** `ConversationStats` object containing:
- `total_messages_sent`: Total number of messages sent
- `total_tokens_input`: Total input tokens used
- `total_tokens_output`: Total output tokens used
- `total_tokens`: Total tokens used
- `total_latency_seconds`: Total time spent waiting for responses
- `average_latency_seconds`: Average response time
- `min_latency_seconds`: Fastest response time
- `max_latency_seconds`: Slowest response time
- `total_sleep_time_seconds`: Total time spent sleeping between messages
- `message_latencies`: List of individual message latencies
- `error`: Error message if something went wrong (None otherwise)

### `simulate_user_session()`

Simulate a user running multiple stress tests over a time period (async function).

```python
async def simulate_user_session(
    conversations_dir: str = 'conversations',
    duration_seconds: int = 60,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature_range: tuple[float, float] = (0.5, 1.0)
) -> UserSessionStats
```

**Parameters:**
- `conversations_dir`: Directory containing conversation JSON files (default: 'conversations')
- `duration_seconds`: How long to run the simulation in seconds (default: 60)
- `model_name`: Model to use (None = randomly select from API, or specify a fixed model)
- `api_key`: API key (required)
- `api_base`: API base URL (required)
- `temperature_range`: Min and max temperature for random selection (default: (0.5, 1.0))

**Returns:** `UserSessionStats` object containing:
- `session_start`: Start timestamp
- `session_end`: End timestamp
- `duration_seconds`: Total duration
- `total_conversations`: Total conversations attempted
- `successful_conversations`: Successfully completed conversations
- `failed_conversations`: Failed conversations
- `total_messages_sent`: Total messages across all conversations
- `total_tokens`: Total tokens used
- `total_tokens_input`: Total input tokens
- `total_tokens_output`: Total output tokens
- `average_latency_seconds`: Average latency across all messages
- `min_latency_seconds`: Fastest message
- `max_latency_seconds`: Slowest message
- `conversations_per_minute`: Rate of conversations
- `models_used`: List of models used during session
- `conversation_stats`: List of individual ConversationStats

**Behavior:**
- Runs continuously until `duration_seconds` expires
- Randomly selects a conversation file for each iteration
- Randomly selects model (if `model_name` is None)
- Randomly selects temperature within specified range
- 20% chance of limiting messages (1-3 messages)
- Aggregates statistics across all runs

### `print_session_report()`

Print a formatted report of user session statistics.

```python
def print_session_report(stats: UserSessionStats) -> None
```

Outputs a comprehensive report including:
- Session duration and timestamps
- Conversation statistics (total, successful, failed, rate)
- Token usage (total, input, output)
- Latency metrics (average, min, max)
- Models used
- Error summary (if any failures)

## Testing

Run the test suite:

```bash
# Install dev dependencies (includes pytest-cov)
uv sync --extra dev

# Run all tests (25 total)
uv run pytest test_stresser.py test_session.py -v

# Run with coverage report
uv run pytest test_stresser.py test_session.py --cov=stresser --cov-report=term-missing

# Run only session simulation tests
uv run pytest test_session.py -v
```

The test suite includes (25 tests total):

**Core functionality tests (test_stresser.py - 13 tests):**
- Sleep time calculation tests
- File handling tests
- Successful conversation simulation
- Error handling tests
- Statistics calculation verification
- Conversation history building validation

**Session simulation tests (test_session.py - 12 tests):**
- Model fetching from API
- Session duration and timing
- Random parameter selection
- Statistics aggregation
- Error tracking and reporting
- Report formatting

## Project Structure

```
BAG-Stresser/
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── pyproject.toml            # Project configuration and dependencies
├── stresser.py               # Core stress testing functionality
├── test_stresser.py          # Core functionality tests (13 tests)
├── test_session.py           # Session simulation tests (12 tests)
├── example_stresser.py       # Single conversation example
├── example_session.py        # User session simulation example
├── main.py                   # Main entry point
├── conversations/            # Sample conversation files (10 files)
│   ├── conversation_01.json
│   ├── conversation_02.json
│   └── ...
└── README.md                 # This file
```

## Sample Conversations

The project includes 10 sample conversations with varying lengths (1-5 messages):
- Simple questions and answers
- Technical tutorials (HTTP requests, FastAPI, Docker)
- Educational content (quantum computing, machine learning)
- Creative content (haiku generation)

## License

This project is open source.

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting a pull request.
