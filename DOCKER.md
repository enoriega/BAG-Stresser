# Docker Usage Guide

## Overview

BAG-Stresser can be run in a Docker container for easy deployment and isolation. The Docker image includes all dependencies and can be configured through environment variables and command-line arguments.

## Building the Docker Image

```bash
# Build the image
docker build -t bag-stresser .

# Build with a specific tag
docker build -t bag-stresser:v1.0 .
```

## Configuration

### Environment Variables

Create a `.env` file in your project root (already gitignored for security):

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-3.5-turbo  # Optional, if not set will randomly select models
```

### Model Filter File

Create a `model_filter.txt` file to exclude specific models from testing:

```bash
# Create model filter
cat > model_filter.txt << EOF
# Expensive models to exclude
gpt-4-32k
claude-opus-3

# Embedding models
text-embedding-ada-002
EOF
```

The Docker image will automatically include this file if it exists in your project directory.

## Running the Container

### Basic Usage

```bash
# Run with default settings (60 seconds, 1 session)
docker run --env-file .env bag-stresser

# Run for 120 seconds
docker run --env-file .env bag-stresser --duration 120

# Run 3 concurrent sessions
docker run --env-file .env bag-stresser --sessions 3 --duration 60
```

### Using Custom Model Filter

If you have a custom model filter file that's not in your build context:

```bash
# Mount a custom model filter file
docker run --env-file .env \
  -v $(pwd)/my-filter.txt:/app/model_filter.txt \
  bag-stresser --duration 120

# Specify a different filter file location
docker run --env-file .env \
  -v $(pwd)/my-filter.txt:/app/custom-filter.txt \
  bag-stresser --duration 120 --model-filter /app/custom-filter.txt
```

### Multi-Processing Mode

```bash
# Run with 4 worker processes, each running 3 sessions (12 total)
docker run --env-file .env bag-stresser \
  --workers 4 \
  --sessions 3 \
  --duration 60

# Maximum stress test: 5 workers × 5 sessions = 25 concurrent sessions
docker run --env-file .env bag-stresser \
  --workers 5 \
  --sessions 5 \
  --duration 120
```

### Advanced Options

```bash
# Show individual session reports
docker run --env-file .env bag-stresser \
  --sessions 3 \
  --duration 60 \
  --show-individual

# Use specific model
docker run --env-file .env bag-stresser \
  --model gpt-4 \
  --duration 60

# Custom temperature range
docker run --env-file .env bag-stresser \
  --sessions 5 \
  --temp-min 0.3 \
  --temp-max 0.9 \
  --duration 120
```

### Using Custom Conversations

If you want to use custom conversation files:

```bash
# Mount a custom conversations directory
docker run --env-file .env \
  -v $(pwd)/my-conversations:/app/conversations \
  bag-stresser --duration 60

# Use a different conversations directory inside the container
docker run --env-file .env \
  -v $(pwd)/my-conversations:/app/my-convos \
  bag-stresser \
  --conversations-dir /app/my-convos \
  --duration 60
```

## Complete Examples

### Example 1: Basic Stress Test

```bash
# Create environment file
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
OPENAI_API_BASE=https://api.openai.com/v1
EOF

# Build image
docker build -t bag-stresser .

# Run basic stress test
docker run --env-file .env bag-stresser --duration 60
```

### Example 2: Multi-Processing with Model Filter

```bash
# Create model filter
cat > model_filter.txt << EOF
# Exclude expensive models
gpt-4-32k
text-embedding-ada-002
EOF

# Build image (includes model_filter.txt)
docker build -t bag-stresser .

# Run multi-processing stress test
docker run --env-file .env bag-stresser \
  --workers 4 \
  --sessions 3 \
  --duration 120 \
  --show-individual
```

### Example 3: Custom Setup with Volume Mounts

```bash
# Run with custom conversations and filter
docker run --env-file .env \
  -v $(pwd)/my-conversations:/app/conversations \
  -v $(pwd)/production-filter.txt:/app/model_filter.txt \
  bag-stresser \
  --workers 2 \
  --sessions 5 \
  --duration 300
```

## Docker Compose

For more complex setups, you can use Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'

services:
  stresser:
    build: .
    env_file: .env
    volumes:
      - ./conversations:/app/conversations
      - ./model_filter.txt:/app/model_filter.txt
      - ./results:/app/results  # For saving results
    command: ["--workers", "4", "--sessions", "3", "--duration", "120"]
```

Run with:

```bash
docker-compose up
```

## Available Command-Line Arguments

All `multi_session.py` arguments are supported:

- `--sessions, -s`: Number of concurrent sessions per worker (default: 1)
- `--workers, -w`: Number of worker processes (default: 1)
- `--duration, -d`: Duration in seconds for each session (default: 60)
- `--conversations-dir, -c`: Directory containing conversation JSON files (default: conversations)
- `--model, -m`: Model name to use (default: random selection from API)
- `--temp-min`: Minimum temperature for random selection (default: 0.5)
- `--temp-max`: Maximum temperature for random selection (default: 1.0)
- `--show-individual`: Show detailed report for each individual session
- `--model-filter`: Path to file containing model names to exclude (default: model_filter.txt)

## Troubleshooting

### Container exits immediately

Check that your `.env` file contains valid API credentials:

```bash
docker run --env-file .env bag-stresser --duration 10
```

### Model filter not working

Verify the filter file is included in the image:

```bash
docker run --env-file .env bag-stresser ls -la /app/model_filter.txt
```

### Custom conversations not found

Check the volume mount path:

```bash
docker run --env-file .env \
  -v $(pwd)/conversations:/app/conversations \
  bag-stresser ls -la /app/conversations/
```

## Production Deployment

For production deployments, consider:

1. **Resource Limits**: Set CPU and memory limits
   ```bash
   docker run --cpus=4 --memory=4g --env-file .env bag-stresser
   ```

2. **Logging**: Capture output to a file
   ```bash
   docker run --env-file .env bag-stresser > stress-test-results.log 2>&1
   ```

3. **Automated Scheduling**: Use cron or a scheduler to run periodic tests
   ```bash
   # Example cron entry (run every hour)
   0 * * * * docker run --env-file /path/to/.env bag-stresser --duration 300
   ```

4. **Monitoring**: Consider adding health checks and monitoring
   ```bash
   docker run --health-cmd="echo 'ok'" --env-file .env bag-stresser
   ```
